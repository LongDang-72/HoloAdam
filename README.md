**HoloAdam: High-Fidelity LLM Optimization**

**Overview**

HoloAdam is a production-grade optimizer designed for large-scale LLM training. It reduces optimizer state memory by **~95%** via holographic superposition while maintaining the convergence stability that other low-rank methods (like Adafactor) often lack.

**⚡ Key Features**

- **95% Memory Reduction**: Achieves 20x-50x compression via holographic encoding .
- **Multi-Head Ensemble**: 4 holographic heads reduce crosstalk noise for high fidelity.
- **Auto-Configuration**: Automatically handles sparse embeddings, small layers, and mixed-precision (FP16/BF16).
- **Resonance Gating**: Adaptive noise filtering ensures stability during the critical warmup phase.

**🚀 Quick Start**

Python

from holo_optimizer import HoloAdam

\# Standard usage (Auto-config)

optimizer = HoloAdam(model.parameters(), lr=1e-4)

\# Advanced usage (Custom compression)

optimizer = HoloAdam(

model.parameters(),

lr=1e-4,

heads=4, # More heads = less noise, slightly more memory

holo_ratio=20, # Compression ratio (20x)

resonance_factor=0.4 # Noise gating strength

)

\# Standard Training Loop

loss.backward()

optimizer.step()

optimizer.zero_grad()

**⚙️ Configuration**

| **Parameter** | **Default** | **Description** |
| --- | --- | --- |
| lr  | 1e-4 | Learning rate |
| betas | (0.9, 0.999) | Momentum coefficients |
| eps | 1e-8 | Arithmetic stability constant |
| weight_decay | 0.01 | Decoupled weight decay (AdamW) |
| holo_ratio | 20  | Compression ratio (higher = more memory savings) |
| v_block_size | 32  | Size of average block |
| heads | 4   | Number of holographic heads |
| resonance_factor | 0.4 | Noise gating strength (0.00-1.0) |
| warmup_steps | 1000 | Steps to ramp up resonance gating |
| min_compress_size | 8192 | Minimum params to trigger compression |
| fft_chunk_size | 2_000_000 | Number of param for gradient chunking |

**Custom Compression Settings**

Python

\# Maximum compression (lowest memory)

optimizer = HoloAdam(model.parameters(), holo_ratio=50, heads=2)

\# Balanced (recommended)

optimizer = HoloAdam(model.parameters(), holo_ratio=20, heads=4)

\# Conservative (highest fidelity)

optimizer = HoloAdam(model.parameters(), holo_ratio=10, heads=8)

**📊 Performance & Convergence**

HoloAdam strikes a critical balance: **sub-linear memory scaling** with **full-state convergence fidelity**.

**Memory Comparison**

| **Model Size** | **AdamW Memory** | **HoloAdam Memory** | **Reduction** |
| --- | --- | --- | --- |
| **1B params** | ~8 GB | ~0.4 GB | **95%** |
| **7B params** | ~56 GB | ~2.8 GB | **95%** |
| **70B params** | ~560 GB | ~28 GB | **95%** |

**Convergence Verification**

We conducted rigorous fine-tuning experiments on **Qwen-2.5-3B** and **Llama-3-8B** to evaluate convergence stability. While Adafactor offers aggressive memory reduction, it often sacrifices training stability. **HoloAdam** strikes a critical balance: maintaining a compact memory footprint while delivering the convergence fidelity of full-state optimizers.

**Experiment 1: Qwen-2.5-3B (Alpaca Dataset)**

_Hardware: NVIDIA RTX 3090 (24GB)_

| **Metric** | **HoloAdam (Ours)** | **Adafactor** | **Analysis** |
| --- | --- | --- | --- |
| **Peak Memory** | 10.23 GB | **5.81 GB** | HoloAdam maintains safe memory overhead |
| **Final Loss** | **0.09** | 7.00 | **HoloAdam converges; Adafactor collapsed** |

**Experiment 2: Meta-Llama-3-8B (Alpaca Dataset)**

_Hardware: NVIDIA RTX 4090 (48GB VRAM)_

| **Metric** | **HoloAdam (Ours)** | **Adafactor** | **Analysis** |
| --- | --- | --- | --- |
| **Peak Memory** | 17.80 GB | **14.98 GB** | Comparable footprint (+18%) |
| **Final Loss** | **0.08** | 7.00 | **HoloAdam achieves interested convergence** |

**Key Observation: The Stability Gap**

The data highlights a critical trade-off in low-memory optimization:

- **Convergence Failure in Aggressive Compression:** The baseline Adafactor runs resulted in a stagnant loss of ~7.00, indicating a **model collapse** or failure to escape local minima due to excessive state quantization.
- **Holographic Fidelity:** HoloAdam achieved a final loss of **< 0.1**, demonstrating that our Holographic Encoding preserves essential gradient directionality that other low-rank methods discard.

**Conclusion**

HoloAdam effectively solves the "instability bottleneck" of memory-efficient optimizers, making it the preferred choice for training sensitive LLMs where convergence reliability is paramount.

**Checkpoints & Logs:** [Download Here](https://drive.google.com/drive/folders/1598hlUIgZogDZnx7IBA0fWuexAcv7Smb?usp=sharing)

**🧪 Reproducibility**

We provide the exact scripts used to generate the results in Section 5 of the paper.

| **Experiment** | **Script** | **Description** |
| --- | --- | --- |
| **Physics Validation** | experiments/1_rosenbrock_physics.py | Verifies vector preservation and Scale Invariance (Section 5.1). |
| **CIFAR-10 ResNet-18** | experiments/2_ResNet18_Experiment.py | Reproduces ~89% convergence accuracy with spectral denoising (\$L=4\$) (Section 5.2). |
| **NanoGPT** | experiments/3_NanoGPT_Experiment.py | Trains TinyShakespeare to observe the "Crossover Phenomenon" and Gating effects (Section 5.3). |
| **LLM Fine-Tuning** | experiments/qwen_finetuning_rtx3090.ipynb | Full fine-tuning harness for Qwen-2.5-3B (Section 5.4). |
| **LLM Fine-Tuning** | experiments/llama_finetuning_rtx4090.ipynb | Full fine-tuning harness for Llama-3-8B (Section 5.4). |

**🧠 How It Works**

- **Holographic Encoding**: Gradients are bound to random phase keys and compressed via FFT.
- **Multi-Head Ensemble**: 4 independent encodings reduce reconstruction noise.
- **Spectrogram Variance**: High-resolution second moment tracking in the spatial domain.
- **Resonance Gating**: Alignment-based filter removes decoding artifacts during warmup.

**🛠 Installation**

Requires **Python 3.10+**, **PyTorch 2.0+**, and **CUDA 12+**.

**Fast Setup (using uv)**

If you are using uv for training, this is the experience to be able to easily install and run flash-attention:

Bash

\# Set up environment

uv venv .venv310 --python=python3.10

source .venv310/bin/activate

\# Set up packet for fast install

uv pip install --upgrade pip

uv pip install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cu121>

\# Register kernel

uv pip install ipykernel

python -m ipykernel install --user \\

\--name venv310 \\

\--display-name "Python 3.10 (venv310)"

\# Set up libraries

uv pip install bitsandbytes accelerate

uv pip install -U setuptools wheel packaging

uv pip install flash-attn --no-build-isolation

uv pip install transformers datasets trl tensorboardX

uv pip install -U jupyter ipywidgets tqdm

\# Clean memory if needed

rm -rf ~/.cache/huggingface

uv cache clean

pip cache purge

**📝 Citation**

If you use HoloAdam in your research, please cite:

Code snippet

@article{holo_adam_2026,

title={HoloAdam: High-Fidelity LLM Optimization via Holographic Momentum Superposition},

author={Dang, Long and Pham, Hieu},

year={2026}

}
