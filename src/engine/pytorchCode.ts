import { ModelConfig } from '../types';

export function generatePyTorchScript(config: ModelConfig, sampleDataName: string = 'input.txt'): string {
  return `"""
train_tiny_gpt.py
=================
TinyGPT Pretraining Lab: Standalone 1-File PyTorch Pretraining Script
File location: python/train_tiny_gpt.py

Pretrains a minimal 1-block causal language model (Decoder-Only Transformer) on any text,
evaluates train/val loss, saves checkpoint weights, generates autoregressive text,
and exports browser-compatible JSON weights for the TinyGPT visualizer.

Prerequisites:
    pip install torch

Usage:
    python python/train_tiny_gpt.py
    # or inside python/ directory:
    cd python && python train_tiny_gpt.py
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Hyperparameters (Matching TinyGPT Pretraining Lab config)
# -----------------------------------------------------------------------------
batch_size = ${config.batchSize}          # Sequences processed in parallel per step
block_size = ${config.blockSize}         # Maximum context length (tokens)
n_embd = ${config.nEmbd}             # Embedding dimension
n_head = ${config.nHead}              # Number of attention heads
n_layer = 1             # 1 Transformer block (MicroGPT architecture)
learning_rate = ${config.lr}   # Learning rate for AdamW
max_iters = 1000        # Total optimization steps
eval_interval = 100     # Evaluation frequency
eval_iters = 40         # Batches to average for train/val loss
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(1337)
print(f"--> [TinyGPT] Compute device: {device.upper()}")

# -----------------------------------------------------------------------------
# 1. Dataset & Character Tokenizer with 90/10 Train/Validation Split
# -----------------------------------------------------------------------------
corpus_path = "${sampleDataName}"
if not os.path.exists(corpus_path):
    print("--> Creating default Technical Interview corpus (${sampleDataName})...")
    starter_corpus = """INTERVIEWER: Welcome to the evaluation session. Could you describe your background?
CANDIDATE: I specialize in distributed systems, foundation models, and attention mechanisms.
INTERVIEWER: What occurs during the forward pass of a causal transformer block?
CANDIDATE: Input tokens are converted into token embeddings and augmented with positional vectors.
INTERVIEWER: And inside the multi-head attention layer?
CANDIDATE: Queries and keys compute dot-product attention scores, normalized by sqrt(d_k), masked causally, and multiplied by values.
INTERVIEWER: Why does the browser trainer decouple fast heuristics from PyTorch?
CANDIDATE: JavaScript computes forward activations and linear backprop for real-time visualization, while PyTorch performs full analytic backprop through attention.
INTERVIEWER: How do we attach persistent memory to a frozen foundation model?
CANDIDATE: We store explicit facts in an external vector index and retrieve relevant snippets into the prompt context window.
""" * 40
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(starter_corpus)

with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"--> Corpus: {len(text):,} characters | Unique Vocabulary: {vocab_size} tokens")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# 90% train, 10% validation split
data = torch.tensor(encode(text), dtype=torch.long)
n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]
print(f"--> Split: {len(train_data):,} train tokens (90%) | {len(val_data):,} val tokens (10%)")

def get_batch(split: str):
    source = train_data if split == 'train' else val_data
    if len(source) <= block_size + 1:
        source = train_data
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model: nn.Module):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# -----------------------------------------------------------------------------
# 2. Transformer Architecture (1 Transformer Block Decoder-Only GPT)
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=True)

    def forward(self, x: torch.Tensor):
        return self.c_proj(F.relu(self.c_fc(x)))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.block = TransformerBlock(n_embd, n_head, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.block(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 10):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(0.1, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# 3. Training & Validation Loop
# -----------------------------------------------------------------------------
model = TinyGPT(vocab_size, block_size, n_embd, n_head).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"--> Architecture: 1 Transformer Block | Parameters: {param_count:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

initial_loss = math.log(vocab_size)
print(f"--> Initial Theoretical Random Loss: -ln(1/{vocab_size}) = {initial_loss:.4f}\\n")

print(f"{'Step':>6} | {'Train Loss':>11} | {'Val Loss':>10} | {'Perplexity':>11}")
print("-" * 48)

for step in range(max_iters + 1):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        perp = math.exp(min(10, losses['val']))
        print(f"{step:6d} | {losses['train']:11.4f} | {losses['val']:10.4f} | {perp:11.2f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

# -----------------------------------------------------------------------------
# 4. Save PyTorch .pt Checkpoint & Browser-Compatible .json Weights
# -----------------------------------------------------------------------------
checkpoint_pt_path = "tiny_gpt_weights.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': chars,
    'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': 1
    }
}, checkpoint_pt_path)
print(f"\\n--> Saved PyTorch binary checkpoint: {checkpoint_pt_path}")

def export_for_browser_visualizer(output_path="tiny_gpt_weights.json"):
    sd = model.state_dict()
    export_data = {
        "config": {
            "vocabSize": vocab_size,
            "blockSize": block_size,
            "nEmbd": n_embd,
            "nHead": n_head,
            "nLayer": 1
        },
        "vocab": chars,
        "weights": {
            "wte": sd["wte.weight"].detach().cpu().numpy().flatten().tolist(),
            "wpe": sd["wpe.weight"].detach().cpu().numpy().flatten().tolist(),
            "wq": sd["block.attn.q_proj.weight"].detach().cpu().numpy().flatten().tolist(),
            "wk": sd["block.attn.k_proj.weight"].detach().cpu().numpy().flatten().tolist(),
            "wv": sd["block.attn.v_proj.weight"].detach().cpu().numpy().flatten().tolist(),
            "wo": sd["block.attn.out_proj.weight"].detach().cpu().numpy().flatten().tolist(),
            "w1": sd["block.mlp.c_fc.weight"].detach().cpu().t().numpy().flatten().tolist(),
            "b1": sd["block.mlp.c_fc.bias"].detach().cpu().numpy().flatten().tolist(),
            "w2": sd["block.mlp.c_proj.weight"].detach().cpu().t().numpy().flatten().tolist(),
            "b2": sd["block.mlp.c_proj.bias"].detach().cpu().numpy().flatten().tolist(),
            "lmHead": sd["lm_head.weight"].detach().cpu().t().numpy().flatten().tolist()
        }
    }
    with open(output_path, "w") as f:
        json.dump(export_data, f)
    print(f"--> Saved browser-compatible visualizer weights: {output_path}")
    print(f"--> You can now drag-and-drop '{output_path}' directly into the TinyGPT web visualizer!")

export_for_browser_visualizer()

# -----------------------------------------------------------------------------
# 5. Autoregressive Sample Generation
# -----------------------------------------------------------------------------
print("\\n" + "=" * 55)
print("Sample Model Generation from Trained Checkpoint:")
print("=" * 55)
prompt = "INTERVIEWER:"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
output_indices = model.generate(context, max_new_tokens=180, temperature=0.7)[0].tolist()
print(decode(output_indices))
print("=" * 55 + "\\n")
`;
}

export function generateColabNotebookJSON(config: ModelConfig): string {
  const pythonScript = generatePyTorchScript(config);
  const notebook = {
    nbformat: 4,
    nbformat_minor: 0,
    metadata: {
      colab: { name: "TinyGPT_Pretraining_From_Scratch.ipynb", provenance: [] },
      kernelspec: { name: "python3", display_name: "Python 3" },
      language_info: { name: "python" }
    },
    cells: [
      {
        cell_type: "markdown",
        metadata: {},
        source: [
          "# TinyGPT: Smallest 'From Scratch' Pretraining Experiment\n",
          "This notebook implements a complete Decoder-Only Transformer (like GPT-2) in single-file PyTorch.\n",
          "1. Pretrains on raw text\n",
          "2. Watches loss drop\n",
          "3. Saves `tiny_gpt_weights.pt`\n",
          "4. Generates text autoregressively"
        ]
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          "!nvidia-smi\n",
          "!pip install -q torch"
        ]
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [pythonScript]
      }
    ]
  };
  return JSON.stringify(notebook, null, 2);
}
