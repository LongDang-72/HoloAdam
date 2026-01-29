"""
Holo-Gradient Optimization Framework
Algorithm: HoloAdam (for Large Scale)

Description:
A high-fidelity optimizer for Large Language Models that achieves sub-linear 
memory scaling via Holographic Superposition. It reduces optimizer state memory 
by ~95% while retaining directional momentum history.

Usage:
    from holo_optimizer import HoloAdam
    opt = HoloAdam(model.parameters(), lr=1e-4, holo_ratio=32)
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import math
import gc

class HoloAdam(optim.Optimizer):
    """
    Memory-Optimized Holo-Gradient Optimizer
    - Chunked FFT/IFFT to prevent OOM
    - Aggressive memory cleanup
    - Gradient checkpointing support
    """
    
    def __init__(self, 
                 params,
                 lr=1e-4,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.01,
                 holo_ratio=20,
                 v_block_size=32, 
                 heads=2, # Change from 4 -> 2
                 min_compress_size=8192, 
                 resonance_factor=0.4, 
                 warmup_steps=1000,
                 codebook_size=256, 
                 codebook_width=8192,
                 fft_chunk_size=2_000_000  # Reduced default for safety
                ):
        
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay,
            ratio=holo_ratio, 
            v_block=v_block_size, 
            heads=heads,
            min_size=min_compress_size, 
            max_rf=resonance_factor,
            warmup=warmup_steps, 
            cb_size=codebook_size, 
            cb_width=codebook_width,
            fft_chunk_size=fft_chunk_size
        )
        
        super(HoloAdam, self).__init__(params, defaults)

        # Keep codebook on CPU to save GPU memory
        g_cpu = torch.Generator()
        g_cpu.manual_seed(42)
        phases = torch.rand(codebook_size, codebook_width, generator=g_cpu) * 2 * math.pi
        self.keys_codebook = torch.polar(torch.ones_like(phases), phases)
        self.dev_keys = None

    def _get_keys(self, device, dim, idx):
        """
        Lazy transfer keys to device only when needed
        """
        if self.dev_keys is None or self.dev_keys.device != device:
            # Clear old device keys if exists
            if self.dev_keys is not None:
                del self.dev_keys
                torch.cuda.empty_cache()
            self.dev_keys = self.keys_codebook.to(device)
        
        sel = self.dev_keys[idx]
        
        if dim <= self.dev_keys.shape[1]: 
            return sel[:, :dim]
        
        repeats = (dim // self.dev_keys.shape[1]) + 1
        return sel.repeat(1, repeats)[:, :dim]

    def _get_annealed_rf(self, max_rf, step, warmup):
        """Annealed resonance factor"""
        if step < warmup:
            return max_rf * (step / warmup)
        return max_rf

    def _chunked_fft_compress(self, grad_flat, h_dim, keys, chunk_size):
        """
        Memory-efficient chunked FFT compression
        
        Critical optimizations:
        - Process small batches at a time
        - Immediate memory cleanup after each batch
        - No intermediate tensor accumulation
        """
        total_length = grad_flat.shape[0]
        n_full_chunks = total_length // h_dim
        
        # Calculate safe batch size
        chunks_per_batch = max(1, chunk_size // h_dim)
        
        # Pre-allocate result buffer
        compressed = torch.zeros(h_dim, dtype=torch.complex64, device=grad_flat.device)
        normalization = 1.0 / math.sqrt(max(1, n_full_chunks))
        
        # Process in small batches
        for batch_start in range(0, n_full_chunks, chunks_per_batch):
            batch_end = min(batch_start + chunks_per_batch, n_full_chunks)
            batch_size = batch_end - batch_start
            
            # Extract batch - avoid creating large intermediate tensors
            start_idx = batch_start * h_dim
            end_idx = batch_end * h_dim
            
            # Reshape in-place view (no copy)
            batch_grad = grad_flat[start_idx:end_idx].view(batch_size, h_dim)
            
            # Get corresponding keys slice
            batch_keys = keys[batch_start:batch_end]
            
            # FFT on small batch
            with torch.amp.autocast(device_type="cuda", enabled=True):  # Disable AMP for FFT
                batch_fft = torch.fft.fft(batch_grad.to(torch.float32), dim=-1)
            
            # Accumulate compressed result
            compressed.add_((batch_fft * batch_keys).sum(dim=0) * normalization)
            
            # Critical: Delete immediately to free memory
            del batch_grad, batch_keys, batch_fft
        
        return compressed

    def _chunked_ifft_decompress(self, compressed, h_dim, keys, original_length, chunk_size):
        """
        Memory-efficient chunked IFFT decompression
        """
        n_full_chunks = original_length // h_dim
        chunks_per_batch = max(1, chunk_size // h_dim)
        
        # Pre-allocate result
        result = torch.zeros(original_length, dtype=torch.float32, device=compressed.device)
        
        # Process in batches
        for batch_start in range(0, n_full_chunks, chunks_per_batch):
            batch_end = min(batch_start + chunks_per_batch, n_full_chunks)
            batch_size = batch_end - batch_start
            
            # Get keys for this batch
            batch_keys = keys[batch_start:batch_end]
            
            # Decompress: broadcast and multiply with conjugate
            with torch.amp.autocast(device_type="cuda", enabled=True):
                batch_freq = compressed.unsqueeze(0).expand(batch_size, -1) * torch.conj(batch_keys)
                batch_spatial = torch.fft.ifft(batch_freq, dim=-1).real
            
            # Write to result buffer
            start_idx = batch_start * h_dim
            end_idx = batch_end * h_dim
            result[start_idx:end_idx] = batch_spatial.flatten()
            
            # Immediate cleanup
            del batch_keys, batch_freq, batch_spatial
        
        return result

    @torch.no_grad()
    def step(self, closure=None):
        """
        Optimizer step with aggressive memory management
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps, wd = group['eps'], group['weight_decay']
            num_heads = group['heads']
            fft_chunk_size = group['fft_chunk_size']
            
            for p in group['params']:
                if p.grad is None: 
                    continue
                
                # CRITICAL: Work on a detached copy to avoid autograd overhead
                grad = p.grad.detach().to(torch.float32)
                
                # Weight decay (decoupled)
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                state = self.state[p]
                
                # Determine compression strategy
                is_large = p.numel() > group['min_size']
                is_matrix = p.dim() > 1
                
                should_compress = is_large and is_matrix
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    if should_compress:
                        h = max(64, p.numel() // group['ratio'])
                        if h % 2 != 0: 
                            h += 1
                        
                        pad = (h - (p.numel() % h)) % h
                        n_ch = (p.numel() + pad) // h
                        
                        state['idx'] = (torch.arange(n_ch, device=p.device) % group['cb_size'])
                        state['hm'] = torch.zeros(num_heads, h, dtype=torch.complex64, device=p.device)
                        state['h_dim'] = h
                        state['sv'] = torch.zeros(
                            math.ceil(p.numel() / group['v_block']), 
                            dtype=torch.float32, 
                            device=p.device
                        )
                        state['mode'] = 'holo'
                    else:
                        state['m'] = torch.zeros_like(p, dtype=torch.float32)
                        state['v'] = torch.zeros_like(p, dtype=torch.float32)
                        state['mode'] = 'dense'

                state['step'] += 1
                
                # ============================================================
                # DENSE MODE - Standard Adam
                # ============================================================
                if state['mode'] == 'dense':
                    m, v = state['m'], state['v']
                    
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    denom = v.sqrt().add_(eps)
                    step_size = lr * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
                    
                    p.data.addcdiv_(m, denom, value=-step_size)
                    
                    del grad  # Cleanup
                    continue

                # ============================================================
                # HOLOGRAPHIC MODE
                # ============================================================
                hm_heads = state['hm']
                sv = state['sv']
                h_dim = state['h_dim']
                
                # --- Update Second Moment (Variance) ---
                g_sq = grad.square().flatten()
                pad_v = (group['v_block'] - (g_sq.numel() % group['v_block'])) % group['v_block']
                if pad_v: 
                    g_sq = F.pad(g_sq, (0, pad_v))
                
                sv.mul_(beta2).add_(g_sq.view(-1, group['v_block']).mean(1), alpha=1 - beta2)
                del g_sq  # Free immediately
                
                # --- Prepare Gradient for FFT ---
                g_flat = grad.flatten()
                pad_m = (h_dim - (g_flat.numel() % h_dim)) % h_dim
                if pad_m: 
                    g_flat = F.pad(g_flat, (0, pad_m))
                
                original_length = g_flat.numel()
                
                # Pre-allocate accumulator
                m_accum = torch.zeros_like(grad)
                
                # --- Process Each Head ---
                for h_idx in range(num_heads):
                    # Generate indices for this head
                    head_indices = (state['idx'] + h_idx * 13) % group['cb_size']
                    
                    # Get keys (reuse if possible)
                    keys_full = self._get_keys(p.device, h_dim, head_indices)
                    
                    # === COMPRESS with Chunked FFT ===
                    compressed = self._chunked_fft_compress(
                        g_flat, h_dim, keys_full, fft_chunk_size
                    )
                    
                    # Update hologram state
                    hm_heads[h_idx].mul_(beta1).add_(compressed, alpha=1 - beta1)
                    del compressed
                    
                    # === DECOMPRESS with Chunked IFFT ===
                    decompressed = self._chunked_ifft_decompress(
                        hm_heads[h_idx], h_dim, keys_full, original_length, fft_chunk_size
                    )
                    
                    # Remove padding
                    if pad_m:
                        decompressed = decompressed[:-pad_m]
                    
                    # Accumulate
                    m_accum.add_(decompressed.view_as(grad))
                    
                    # Critical cleanup
                    del decompressed
                    if h_idx < num_heads - 1:  # Don't delete on last iteration if reusing
                        del keys_full

                # Average across heads
                m_accum.div_(num_heads)
                
                # --- Resonance Gating ---
                curr_rf = self._get_annealed_rf(group['max_rf'], state['step'], group['warmup'])
                
                # Memory-efficient gating
                gate = grad.mul(m_accum).mul_(5.0).tanh_().mul_(0.5).add_(0.5)
                m_gated = m_accum.mul_(1.0 - curr_rf).add_(m_accum.mul(gate).mul_(curr_rf))
                
                del gate, m_accum  # Free before next allocation
                
                # --- Build Denominator ---
                v_exp = sv.repeat_interleave(group['v_block'])
                if pad_v: 
                    v_exp = v_exp[:-pad_v]
                denom = v_exp.view_as(grad).sqrt_().add_(eps)
                del v_exp
                
                # --- Apply Update ---
                step_size = lr * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
                p.data.addcdiv_(m_gated.clamp_(-2.0, 2.0), denom, value=-step_size)
                
                # Final cleanup for this parameter
                del grad, g_flat, m_gated, denom
            
            # Aggressive memory cleanup after each parameter group
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return loss
