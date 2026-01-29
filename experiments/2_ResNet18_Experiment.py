import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys
import math

# ==========================================
# 0. THE HOLOADAM OPTIMIZER (v2.0 - Binding Fix)
# ==========================================
class HoloAdam(optim.Optimizer):
    """
    HoloAdam: Holographic Compressed Optimization
    Uses Circular Convolution (Binding) instead of Aliasing for robust compression.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, holo_ratio=8, heads=1, resonance_factor=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        holo_ratio=holo_ratio, heads=heads, resonance_factor=resonance_factor)
        super(HoloAdam, self).__init__(params, defaults)
        
        # Global Key Cache: Shared across all parameters to save VRAM
        # We generate enough keys for the largest likely layer chunk
        self.cache_size = 200000 
        self.max_ratio = 128
        self.global_keys = None
        self.register_global_keys(self.max_ratio, self.cache_size)

    def register_global_keys(self, ratio, size):
        """Generates the fixed random phasors for holographic binding"""
        print(f"Generating Holographic Keys [{ratio}x{size}]...")
        # Phases in [0, 2pi]
        phases = torch.rand(ratio, size) * 2 * math.pi
        # Convert to Complex Polar form (Magnitude=1, Phase=Random)
        self.global_keys = torch.polar(torch.ones_like(phases), phases)
        if torch.cuda.is_available():
            self.global_keys = self.global_keys.cuda()

    def get_keys(self, rows, cols, device):
        """Fetches keys from cache, handling device transfer and repeats"""
        if self.global_keys is None:
             self.register_global_keys(self.max_ratio, self.cache_size)
             
        if self.global_keys.device != device:
            self.global_keys = self.global_keys.to(device)
            
        # Slice the required keys
        keys = self.global_keys[:rows, :cols]
        
        # If the layer is wider than our cache, repeat the keys (circular)
        if cols > self.global_keys.shape[1]:
            repeats = (cols // self.global_keys.shape[1]) + 1
            keys = self.global_keys[:rows].repeat(1, repeats)[:, :cols]
            
        return keys

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ratio = group['holo_ratio']
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                
                # Check Sparse
                if grad.is_sparse:
                    raise RuntimeError('HoloAdam does not support sparse gradients')

                state = self.state[p]

                # --- Initialization ---
                if len(state) == 0:
                    state['step'] = 0
                    n_params = p.numel()
                    
                    # Compressed size is roughly N / Ratio
                    compressed_size = math.ceil(n_params / ratio)
                    
                    # Store Momentum in Frequency Domain (Complex64)
                    state['holo_m'] = torch.zeros(compressed_size, dtype=torch.complex64, device=p.device)
                    
                    # Variance (v) is kept dense for stability (Standard AdamW)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                
                # --- A. Standard Variance Update (AdamW) ---
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # --- B. Holographic Momentum Compression ---
                # 1. Flatten Gradient
                flat_grad = grad.view(-1)
                
                # 2. Pad to be divisible by target length
                target_len = state['holo_m'].shape[0] * ratio
                if flat_grad.numel() < target_len:
                    padding = torch.zeros(target_len - flat_grad.numel(), device=p.device)
                    flat_grad = torch.cat([flat_grad, padding])
                
                # 3. View as Matrix [Ratio, Compressed_Size]
                grad_segments = flat_grad.view(ratio, -1)
                
                # 4. Fetch Keys
                # We need [Ratio, Compressed_Size] keys to bind each segment uniquely
                keys = self.get_keys(ratio, grad_segments.shape[1], p.device)
                
                # 5. Binding (Circular Convolution via FFT)
                # FFT each segment -> Multiply by Key -> Sum (Superposition)
                seg_freq = torch.fft.fft(grad_segments)
                bound_freq = seg_freq * keys  # Element-wise multiply (Binding)
                
                # Average (instead of Sum) to keep magnitude consistent
                superposition = bound_freq.mean(dim=0) 
                
                # 6. Update Compressed Momentum
                state['holo_m'].mul_(beta1).add_(superposition, alpha=1 - beta1)

                # --- C. Retrieval (Unbinding) ---
                # 1. Unbind: Momentum * Conj(Key)
                # Broadcast: [1, Size] * [Ratio, Size] -> [Ratio, Size]
                rec_freq = state['holo_m'].unsqueeze(0) * torch.conj(keys)
                
                # 2. Inverse FFT
                rec_time = torch.fft.ifft(rec_freq).real
                
                # 3. Unfold back to vector
                rec_flat = rec_time.view(-1)
                retrieved_grad = rec_flat[:p.numel()].view_as(p)

                # --- D. Weight Update ---
                # Bias correction
                step_size = group['lr'] * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
                
                # Decoupled Weight Decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Apply Update
                p.data.addcdiv_(retrieved_grad, denom, value=-step_size)

        return loss

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
print(f"Torc Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    sys.exit("⚠️ STOP: You are on CPU. Please enable GPU in Kaggle Settings.")

# --- SETTINGS ---
BATCH_SIZE = 128
EPOCHS = 60           # 60 Epochs is sufficient for >90% on ResNet18
# IMPORTANT: For Dense models (ResNet), use Ratio 8. For Sparse LLMs, use 32.
HOLO_RATIO = 8        
LR = 0.1              # Initial Learning Rate (Standard for ResNet/SGD)

def get_loaders():
    print("Downloading/Loading CIFAR-10 Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader

def train_model(name, optimizer_type, heads=None):
    device = "cuda"
    print(f"\n========================================")
    print(f"STARTING EXPERIMENT: {name}")
    print(f"========================================")

    # CIFAR-10 Optimized ResNet18 (No massive downsampling at start)
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() 
    model = model.to(device)

    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    # Initialize Optimizer
    if optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    elif optimizer_type == "HoloAdam":
        optimizer = HoloAdam(
            model.parameters(),
            lr=1e-3, 
            holo_ratio=HOLO_RATIO,
            heads=heads,
            weight_decay=5e-4,
            resonance_factor=0.0 
        )

    # Cosine Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        correct_train = 0; total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        scheduler.step() 

        # Validation
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        history.append(acc)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"[{name}] Ep {epoch+1}/{EPOCHS} | Train: {100.*correct_train/total_train:.1f}% | Test: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.1e}")

    print(f"✅ {name} Finished. Peak: {max(history):.2f}%")
    return history

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    results = {}

    # 1. Baseline: Standard AdamW
    results['AdamW (Baseline)'] = train_model("AdamW", "AdamW")

    # 2. HoloAdam L=1 (Single Head)
    results['Holo (L=1)'] = train_model("Holo L=1", "HoloAdam", heads=1)

    # 3. HoloAdam L=4 (Ensemble)
    results['Holo (L=4)'] = train_model("Holo L=4", "HoloAdam", heads=4)

    # Plot
    plt.figure(figsize=(10, 6))
    for name, hist in results.items():
        style = '--' if 'Adam' in name else '-'
        width = 2.5 if 'L=4' in name or 'Adam' in name else 1.5
        alpha = 1.0 if 'L=4' in name or 'Adam' in name else 0.7
        plt.plot(hist, linestyle=style, linewidth=width, alpha=alpha, label=f"{name} (Best: {max(hist):.1f}%)")

    plt.title(f"HoloAdam Ablation: Binding Logic (Ratio={HOLO_RATIO})")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("holo_ResNet18_results.png")
    print("\n📊 Graph saved as 'holo_ResNet18_results.png'")
    plt.show()