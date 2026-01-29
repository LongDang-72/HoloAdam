import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. DEFINE HOLOADAM (The Actual Implementation)
# ==========================================
class HoloAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 holo_ratio=8, heads=4, v_block_size=32, min_compress_size=4096,
                 resonance_factor=0.1, warmup_steps=1000, codebook_size=256, codebook_width=8192):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        ratio=holo_ratio, heads=heads, v_block=v_block_size,
                        min_size=min_compress_size, rf=resonance_factor,
                        warmup=warmup_steps, cb_size=codebook_size, cb_width=codebook_width)
        super(HoloAdam, self).__init__(params, defaults)
        g = torch.Generator(); g.manual_seed(42)
        phases = torch.rand(codebook_size, codebook_width, generator=g) * 2 * math.pi
        self.keys_codebook = torch.polar(torch.ones_like(phases), phases)
        self.dev_keys = None

    def _get_keys(self, device, dim, idx):
        if self.dev_keys is None or self.dev_keys.device != device: self.dev_keys = self.keys_codebook.to(device)
        sel = self.dev_keys[idx % self.defaults['cb_size']]
        if dim <= self.dev_keys.shape[1]: return sel[:dim]
        return sel.repeat(math.ceil(dim / self.dev_keys.shape[1]))[:dim]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure: loss = closure()
        for group in self.param_groups:
            lr, beta1, beta2, eps = group['lr'], group['betas'][0], group['betas'][1], group['eps']
            step_rf = group['rf']
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0: state['step'] = 0
                state['step'] += 1

                # Anneal Resonance
                if state['step'] < group['warmup']: step_rf = group['rf'] * (state['step'] / group['warmup'])
                if p.grad is None: continue
                grad = p.grad

                # Check Compression Eligibility
                N = p.numel()
                is_compressed = (N > group['min_size']) and (p.dim() > 1)

                # --- STANDARD PATH (Uncompressed) ---
                if not is_compressed:
                    if 'm' not in state: state['m'] = torch.zeros_like(p); state['v'] = torch.zeros_like(p)
                    state['m'].mul_(beta1).add_(grad, alpha=1-beta1)
                    state['v'].mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    denom = state['v'].sqrt().add_(eps)
                    step_size = lr * math.sqrt(1 - beta2**state['step']) / (1 - beta1**state['step'])
                    p.data.addcdiv_(state['m'], denom, value=-step_size)
                    continue

                # --- HOLOGRAPHIC PATH ---
                # (Logic included for completeness, though Benchmark 1 uses small tensors)
                D = max(256, N // group['ratio']); D += D % 2
                if 'heads' not in state:
                    state['heads'] = [torch.zeros(D, device=p.device, dtype=torch.complex64) for _ in range(group['heads'])]
                    state['v_blocks'] = torch.zeros(math.ceil(N/group['v_block']), device=p.device)

                keys = self._get_keys(p.device, D, state['step'] % group['cb_size'])
                pad_len = (D - (N % D)) % D
                g_flat = F.pad(grad.view(-1), (0, pad_len)) if pad_len > 0 else grad.view(-1)
                g_proj = torch.fft.fft(g_flat.view(-1, D), dim=-1).sum(dim=0)

                m_accum = torch.zeros_like(g_flat)
                for h in range(group['heads']):
                    h_key = torch.roll(keys, shifts=h*100, dims=0)
                    state['heads'][h].mul_(beta1).add_(g_proj * h_key, alpha=1-beta1)
                    decoded = torch.fft.ifft(state['heads'][h] * torch.conj(h_key)).real
                    m_accum.add_(decoded.repeat(g_flat.view(-1, D).shape[0]))

                m_accum.div_(group['heads'])
                if pad_len > 0: m_accum = m_accum[:-pad_len]
                m_accum = m_accum.view_as(grad)

                # Update
                g_sq = grad.pow(2).view(-1)
                if pad_len: g_sq = F.pad(g_sq, (0, pad_len))
                g_blk = g_sq.view(-1, group['v_block']).mean(dim=1)
                if g_blk.numel() > state['v_blocks'].numel(): g_blk = g_blk[:state['v_blocks'].numel()]
                state['v_blocks'].mul_(beta2).add_(g_blk, alpha=1-beta2)

                v_exp = state['v_blocks'].repeat_interleave(group['v_block'])
                if pad_len: v_exp = v_exp[:-pad_len]

                denom = v_exp.view_as(grad).sqrt().add_(eps)
                step_size = lr * math.sqrt(1 - beta2**state['step']) / (1 - beta1**state['step'])
                p.data.addcdiv_(m_accum, denom, value=-step_size)
        return loss

# ==========================================
# 2. EXPERIMENT A: ROSENBROCK VALLEY
# ==========================================
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def run_rosenbrock_bench():
    print(">>> Running Rosenbrock Benchmark...")
    start_pos = torch.tensor([-1.5, -1.0])
    steps = 250
    lr = 0.02

    # Trackers
    paths = {'AdamW': [], 'Adafactor (No Mom)': [], 'HoloAdam': []}

    # 1. AdamW (Full Momentum)
    p1 = torch.nn.Parameter(start_pos.clone())
    opt1 = torch.optim.AdamW([p1], lr=lr)

    # 2. Simulated Adafactor (Beta1 = 0.0 removes momentum, relying only on RMSProp scaling)
    p2 = torch.nn.Parameter(start_pos.clone())
    opt2 = torch.optim.AdamW([p2], lr=lr, betas=(0.0, 0.999))

    # 3. HoloAdam (Our Method)
    p3 = torch.nn.Parameter(start_pos.clone())
    # Note: For 2 parameters, HoloAdam defaults to standard update (min_compress_size),
    # verifying that the implementation correctly falls back to high-fidelity on small tensors.
    opt3 = HoloAdam([p3], lr=lr, min_compress_size=4096)

    optimizers = [
        ('AdamW', opt1, p1),
        ('Adafactor (No Mom)', opt2, p2),
        ('HoloAdam', opt3, p3)
    ]

    for name, opt, p in optimizers:
        paths[name].append(p.detach().numpy().copy())
        for _ in range(steps):
            opt.zero_grad()
            loss = rosenbrock(p[0], p[1])
            loss.backward()
            opt.step()
            paths[name].append(p.detach().numpy().copy())

    return {k: np.array(v) for k, v in paths.items()}

# ==========================================
# 3. EXPERIMENT B: EMBEDDING LEAKAGE
# ==========================================
def run_leakage_bench():
    print(">>> Running Embedding Leakage Simulation...")
    # Setup: 100 tokens, 64 dim
    vocab_size, dim = 100, 64
    active_idx = 50  # We update "King"
    inactive_idx = 51 # We check "Queen"

    # Create Sparse Gradient (One-Hot)
    grad = torch.zeros(vocab_size, dim)
    grad[active_idx, :] = 1.0 # Strong signal

    # Compress (Ratio = 10)
    R = 10
    N = vocab_size * dim
    D = N // R

    # Simulation of HoloAdam Compression Step
    # 1. Generate Keys
    keys = torch.randn(D, dtype=torch.complex64)
    keys = keys / keys.abs()

    # 2. Encode
    g_flat = grad.view(-1)
    # Simple folding for demo (reshape and sum)
    g_chunks = g_flat.view(R, D)
    g_freq = torch.fft.fft(g_chunks).sum(dim=0) # Fold
    memory_state = g_freq * keys # Store in memory

    # 3. Decode (Retrieve)
    # To retrieve, we correlate memory with keys
    rec_freq = memory_state * torch.conj(keys)
    rec_chunks = torch.fft.ifft(rec_freq).real
    rec_flat = rec_chunks.repeat(R) # Repeat across folds (Expansion)

    # 4. Measure
    rec_grad = rec_flat.view(vocab_size, dim)

    signal_strength = rec_grad[active_idx].norm().item()
    noise_leakage = rec_grad[inactive_idx].norm().item()

    return signal_strength, noise_leakage

# ==========================================
# 4. PLOTTING FIGURE 5
# ==========================================
def plot_results(paths, leakage_stats):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot Left: Rosenbrock ---
    ax = axes[0]
    # Generate Contour
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    ax.contour(X, Y, Z, levels=np.logspace(-0.5, 3.5, 20), cmap='gray', alpha=0.3)
    ax.plot(1, 1, 'k*', markersize=15, label='Global Minimum')

    colors = {'AdamW': 'blue', 'Adafactor (No Mom)': 'red', 'HoloAdam': 'green'}
    styles = {'AdamW': '-', 'Adafactor (No Mom)': '--', 'HoloAdam': ':'}

    for name, path in paths.items():
        ax.plot(path[:,0], path[:,1], color=colors[name], linestyle=styles[name],
                linewidth=2 if name!='HoloAdam' else 3, label=name, alpha=0.8)

    ax.set_title("Momentum Dynamics (Rosenbrock Valley)")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.5, 2.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot Right: Leakage ---
    ax = axes[1]
    signal, noise = leakage_stats
    bars = ax.bar(['Active Token\n(Signal)', 'Inactive Token\n(Ghost Noise)'],
                  [signal, noise], color=['green', 'red'], alpha=0.7)

    ax.set_title(f"Sparse Gradient Leakage (Compression R=10)")
    ax.set_ylabel("Retrieved Gradient Magnitude")

    # Add text labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')

    ax.text(1, noise + 0.5, "⚠ Ghost Gradient\ncorrupts semantics!",
            ha='center', color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig("exp_4_1_microbenchmarks.png")
    print("\n✅ Micro-Benchmarks Complete. Plot saved as 'microbenchmarks.png'")
    plt.show()

if __name__ == "__main__":
    path_results = run_rosenbrock_bench()
    leakage_results = run_leakage_bench()
    plot_results(path_results, leakage_results)

# Verifying Theorem 1
def run_scaling_law_bench():
    print(">>> Verifying Theorem 1 (Noise vs Ratio)...")

    # FIX: Use a power of 2 size (131072) so it divides evenly by 2, 4, ... 128
    N = 131072
    ratios = [2, 4, 8, 16, 32, 64, 128]
    noise_levels = []

    grad = torch.randn(N)

    for R in ratios:
        D = N // R

        # Compress
        keys = torch.randn(D, dtype=torch.complex64)
        keys /= keys.abs() # Normalize to unit circle

        # View works now because N % R == 0
        g_chunks = grad.view(R, D)

        # Holographic Superposition (Sum)
        state = torch.fft.fft(g_chunks).sum(dim=0) * keys

        # Retrieve (Decode)
        # Circular Correlation via FFT
        rec = torch.fft.ifft(state * torch.conj(keys)).real

        # Expand back to full size
        rec_grad = rec.repeat(R, 1).view(N) # Correct expansion

        # Measure Noise Variance (MSE between Input and Retrieval)
        # We assume the signal is the mean, so variance is the spread of the error
        noise = (grad - rec_grad).var().item()
        noise_levels.append(noise)

        print(f"Ratio {R}: Noise Variance = {noise:.4f}")

    # Theoretical Prediction: Noise ~ (R-1)/D * ||m||^2
    # Since D = N/R, Noise ~ R^2 roughly? Let's see the plot.

    plt.figure(figsize=(8, 5))
    plt.plot(ratios, noise_levels, 'bo-', linewidth=2, label='Empirical Noise')

    # Optional: Plot linear fit to check linearity
    z = np.polyfit(ratios, noise_levels, 1)
    p = np.poly1d(z)
    plt.plot(ratios, p(ratios), 'r--', label=f'Linear Fit')

    plt.title("Theorem 1 Verification: Noise Scalability")
    plt.xlabel("Compression Ratio (R)")
    plt.ylabel("Reconstruction Noise Variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("theorem_verification.png")
    print("\n✅ Scaling Law Plot Saved as 'theorem_verification.png'")
    plt.show()

if __name__ == "__main__":
    run_scaling_law_bench()
