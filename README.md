# HoloAdam: Holographic Compressed Optimization

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

**HoloAdam** is a memory-efficient optimizer that compresses momentum states by **32x** (adjustable 2x–128x) using Holographic Reduced Representations (HRR). It enables **full pre-training and fine-tuning** of large models (e.g., Llama-3-8B) on consumer hardware without the convergence instability of Adafactor or the rank bottlenecks of LoRA.

> **Paper Title:** *HoloAdam: High-Fidelity LLM Optimization via Holographic Momentum Superposition*

---

## ⚡ Key Features

* **95% Memory Reduction:** Compresses optimizer states using Circular Convolution and Superposition, significantly reducing memory footprint compared to AdamW.
* **Full-Rank Updates:** Unlike LoRA, HoloAdam updates all parameters, allowing for "from scratch" training.
* **Resonance Gating:** A dynamic mechanism that filters out "holographic noise" during early training, ensuring stability comparable to full-precision optimizers.
* **Chunked FFT Processing:** Prevents OOM errors on limited VRAM (e.g., 24GB cards) by processing gradients in memory-safe blocks.

---

## 🚀 Installation

### 1. Clone & Dependencies
```bash
git clone [https://github.com/LongDang-72/HoloAdam.git](https://github.com/LongDang-72/HoloAdam.git)
cd HoloAdam
pip install -r requirements.txt
