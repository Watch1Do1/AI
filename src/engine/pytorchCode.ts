import { ModelConfig } from '../types';

export function generatePyTorchScript(config: ModelConfig, sampleDataName: string = 'input.txt'): string {
  return `"""
train_tiny_gpt.py
=================
The smallest, complete "from scratch" GPT pretraining experiment in PyTorch.
Pretrains a minimal causal language model (Decoder-Only Transformer) on any text file,
saves a checkpoint, and generates new text.

Prerequisites:
  pip install torch

Usage:
  python train_tiny_gpt.py
"""

import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
batch_size = ${config.batchSize}         # Sequences processed in parallel
block_size = ${config.blockSize}         # Maximum context length (tokens)
max_iters = 1000          # Training steps
eval_interval = 100       # How often to check validation loss
learning_rate = ${config.lr}      # AdamW learning rate
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
eval_iters = 50
n_embd = ${config.nEmbd}            # Embedding vector dimension
n_head = ${config.nHead}             # Attention heads
n_layer = ${config.nLayer}            # Transformer blocks
dropout = 0.1
checkpoint_file = 'tiny_gpt_weights.pt'

torch.manual_seed(1337)
print(f"--> Using compute device: {device.upper()}")

# -----------------------------------------------------------------------------
# 1. Dataset & Character Tokenizer
# -----------------------------------------------------------------------------
# Put your text in 'input.txt'. If not found, a starter interview corpus is generated.
data_path = '${sampleDataName}'
if not os.path.exists(data_path):
    print(f"Creating default sample training corpus: {data_path}")
    sample_text = """INTERVIEWER: Welcome. Could you describe your background?
CANDIDATE: I specialize in distributed systems and neural networks.
INTERVIEWER: What is the forward pass of a transformer?
CANDIDATE: Token embeddings and positional vectors pass through multi-head self-attention and MLP layers.
INTERVIEWER: How do we attach memory to a foundation model?
CANDIDATE: We store explicit facts in a key-value or vector index and retrieve them into the context window.
""" * 40
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Corpus length: {len(text):,} characters | Unique vocabulary: {vocab_size} characters")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    if len(d) <= block_size:
        d = train_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# 2. Transformer Architecture (Decoder-Only GPT)
# -----------------------------------------------------------------------------
class Head(nn.Module):
    """ One head of masked causal self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# 3. Training Loop & Checkpoint Saving
# -----------------------------------------------------------------------------
model = TinyGPT().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {param_count:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\\n--> Starting Pretraining from Scratch...")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save checkpoint weights
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': chars,
    'config': {
        'block_size': block_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer
    }
}, checkpoint_file)
print(f"\\n--> Checkpoint weights successfully saved to: {checkpoint_file}")

# -----------------------------------------------------------------------------
# 4. Generate Text from Saved Model
# -----------------------------------------------------------------------------
print("\\n--> Sample Generation from Trained Model:")
context = torch.tensor([encode("INTERVIEWER:")], dtype=torch.long, device=device)
output_tokens = model.generate(context, max_new_tokens=200, temperature=0.8)[0].tolist()
print("-" * 50)
print(decode(output_tokens))
print("-" * 50)
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
