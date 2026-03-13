"""
Autoresearch Pretraining Script - RTX 5090 Optimized
Stable Version: Fixed asdict import and final checkpoint save.
"""

import os
# Updated to modern variable name to remove the warning
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
from dataclasses import dataclass, asdict # FIX: Added asdict here
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

# Standard constants for your 5090 setup
MAX_SEQ_LEN = 2048
TIME_BUDGET = 3600 # 1 hour default
VOCAB_SIZE = 8192 # Keep this at 8192 to match your existing checkpoint weights

def make_dataloader(tokenizer, batch_size, seq_len, split):
    # Direct binary loader to bypass 'prepare' module
    filename = f"{split}.bin"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing {filename}. Run aiprep.py first.")
    
    data = np.fromfile(filename, dtype=np.uint16)
    
    def get_batch():
        # Random offsets for training
        ix = torch.randint(len(data) - seq_len, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
        return x.to('cuda'), y.to('cuda'), None

    while True:
        yield get_batch()

# Minimal Tokenizer class to keep the script running
class Tokenizer:
    @staticmethod
    def from_directory():
        return Tokenizer()
    def get_vocab_size(self):
        return VOCAB_SIZE
# ... [GPT Architecture classes remain exactly the same] ...
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 512

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head, self.n_kv_head = config.n_head, config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = nn.Linear(32, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin):
        B, T, C = x.size()
        cos, sin = cos_sin
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))
            v = v + gate.transpose(1, 2).unsqueeze(-1) * ve
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(norm(q), norm(k), v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        )
    def forward(self, x, ve, cos_sin):
        x = x + self.attn(norm(x), ve, cos_sin)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, config.n_kv_head * (config.n_embd // config.n_head))
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(config.sequence_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :].bfloat16(), persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :].bfloat16(), persistent=False)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        cos_sin = (self.cos[:, :, :T, :], self.sin[:, :, :T, :])
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin)
        logits = self.lm_head(norm(x)).float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else logits

def train():
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    tokenizer = Tokenizer.from_directory()
    config = GPTConfig(vocab_size=tokenizer.get_vocab_size())
    model = GPT(config).to(device)
    model = torch.compile(model)
    
    TOTAL_BATCH_SIZE = 524288
    DEVICE_BATCH_SIZE = 32 
    grad_accum_steps = TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, fused=True)
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

    checkpoint_path = "checkpoint.pth"
    start_step = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resuming from Step {start_step}")

    print(f"Ready on Thing 2 | Device Batch: {DEVICE_BATCH_SIZE} | Accum Steps: {grad_accum_steps}")

    t_start = time.time()
    step = start_step
    while True:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            x, y, _ = next(train_loader)
            with autocast_ctx:
                loss = model(x, y)
            (loss / grad_accum_steps).backward()
        
        optimizer.step()
        dt = (time.time() - t0) * 1000
        print(f"\rStep {step:04d} | Loss {loss.item():.4f} | {dt:.0f}ms", end="")
        
        if step > 0 and step % 100 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': asdict(config) # This will work now!
            }, checkpoint_path)
            print(f" | Checkpoint Saved")
        
        if step == start_step: gc.collect()
        step += 1
        if (time.time() - t_start) > TIME_BUDGET: break

    # Final Save Block
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': asdict(config)
    }, checkpoint_path)
    print(f"\nFinal state saved to {checkpoint_path}. VRAM Peak: {torch.cuda.max_memory_allocated()/1e6:.1f}MB")

if __name__ == "__main__":
    train()