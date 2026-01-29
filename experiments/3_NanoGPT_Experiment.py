import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys

# 1. Setup & Imports
if not torch.cuda.is_available():
    sys.exit("⚠️ STOP: Enable GPU (Runtime > Change runtime type > T4).")

try:
    from holo_optimizer import HoloAdam
except ImportError:
    print("⚠️ WARNING: 'holo_optimizer.py' not found. Using AdamW as placeholder if needed, but HoloAdam is required for paper results.")

# ==========================================
# HYPERPARAMETERS (The "Rescue" Config)
# ==========================================
BATCH_SIZE = 64
BLOCK_SIZE = 128  # Context length
MAX_ITERS = 2000  # Enough for convergence on TinyShakespeare
LEARNING_RATE = 6e-4 # Lower than 1e-3 is crucial for Transformers
DEVICE = 'cuda'
EVAL_INTERVAL = 250
HOLO_RATIO = 8   # Standard paper setting

# ==========================================
# 2. Data Loader (TinyShakespeare)
# ==========================================
import requests
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_src[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_src[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==========================================
# 3. NanoGPT Model (Andrej Karpathy Style)
# ==========================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(384, head_size, bias=False)
        self.query = nn.Linear(384, head_size, bias=False)
        self.value = nn.Linear(384, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,16)
        q = self.query(x) # (B,T,16)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, 384)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, 384)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, 384)
        self.blocks = nn.Sequential(*[Block(384, 6) for _ in range(6)])
        self.ln_f = nn.LayerNorm(384)
        self.lm_head = nn.Linear(384, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# ==========================================
# 4. Training Loop
# ==========================================
def train_experiment(opt_name):
    model = GPT().to(DEVICE)
    
    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    else:
        # HoloAdam Config
        optimizer = HoloAdam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            holo_ratio=HOLO_RATIO, 
            heads=2, 
            resonance_factor=0.0 # Start with gating OFF for stability
        )
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS)
    
    loss_history = []
    print(f"\n>>> Starting {opt_name} Training...")
    
    for iter in range(MAX_ITERS):
        # Sample batch
        xb, yb = get_batch('train')
        
        # Forward
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient Clipping (CRITICAL FOR TRANSFORMERS)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            loss_history.append(losses['val'].item())
            
    print(f"✅ {opt_name} Final Val Loss: {loss_history[-1]:.4f}")
    return loss_history

# Run
loss_adam = train_experiment("AdamW")
loss_holo = train_experiment("HoloAdam")

# Plot
import matplotlib.pyplot as plt
plt.plot(loss_adam, label='AdamW', linestyle='--')
plt.plot(loss_holo, label='HoloAdam (R=8)')
plt.legend()
plt.title("NanoGPT Convergence")
plt.savefig("nanogpt.png")
plt.show()