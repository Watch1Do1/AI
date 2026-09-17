# TinyGPT Pretraining Lab

An educational, interactive laboratory and reference implementation for the smallest "from scratch" causal language model pretraining experiment.

---

## 🎯 The "Smallest From Scratch" Experiment

If the goal is to understand how foundational Large Language Models (LLMs) are built, the minimal genuine experiment is:
1. **Clone or write a standalone 1-file GPT trainer in PyTorch.**
2. **Train a 1-block causal Transformer on a text corpus.**
3. **Train until train loss and validation loss drop significantly.**
4. **Generate autoregressive text token-by-token from the saved checkpoint.**

At this point, you have a foundation model in the technical sense: pretrained from scratch without any external API or teacher model.

---

## 🧠 Architecture: 1 Transformer Block

The model is strictly configured as a **1-block Decoder-Only Transformer** (GPT architecture):
* **Context window ($T$):** 16–64 tokens (default: 32).
* **Embedding dimension ($d_{\text{model}}$):** 24–48 (default: 32).
* **Attention heads ($n_{\text{head}}$):** 2 (head dimension: $d_{\text{head}} = 16$).
* **Transformer depth ($n_{\text{layer}}$):** **1 Block (Hardcoded)**.
* **Feedforward MLP:** Dimension expansion $d_{\text{model}} \to 4 \times d_{\text{model}} \to d_{\text{model}}$ with ReLU / GELU activation.
* **Causal Masking:** Strict lower-triangular attention mask ensuring autoregressive token generation.
* **Loss Function:** Categorical Cross-Entropy (Negative Log Likelihood) over vocabulary logits.

---

## 🔬 Browser Engine vs. PyTorch Source of Truth

TinyGPT Lab includes two complementary implementations:

### 1. PyTorch (`python/train_tiny_gpt.py`) — Source of Truth
* Standalone ~150-line PyTorch script requiring only `pip install torch`.
* **Full analytic backpropagation** through all components, including multi-head causal self-attention matrices ($W_q, W_k, W_v, W_o$).
* **90% Train / 10% Validation split** with regular validation loss logging.
* Exports standard PyTorch `.pt` weights and automatically dumps browser-compatible `tiny_gpt_weights.json`.

### 2. Browser Engine (`src/engine/transformer.ts`) — Real-Time Visualizer
* Runs 100% locally in the browser with zero server dependencies.
* Evaluates both **Train Loss (90% split)** and **Validation Loss (10% split)** in real time.
* Real-time attention weight matrix heatmaps ($Q \times K^T / \sqrt{d}$) across all heads.
* **Implementation Note:** The browser `trainStep` updates token embeddings, positional embeddings, MLP projections ($W_1, b_1, W_2, b_2$), and the language model head for real-time visualization. It **does not fully backprop through self-attention**. Full attention gradient backpropagation is executed in PyTorch.

---

## 🔒 Strict Weight Validation

To ensure mathematical consistency between PyTorch and the browser:
* All uploaded `tiny_gpt_weights.json` checkpoints undergo **atomic shape validation** before updating model state.
* If any tensor shape or dimension mismatches the 1-block MicroGPT configuration (e.g. `wte`, `wpe`, `wq`, `wk`, `wv`, `wo`, `w1`, `b1`, `w2`, `b2`, `lmHead`), the import is **strictly rejected** with an informative error message explaining the expected vs. received shape.

---

## 🚀 Quickstart: Running the PyTorch Script

```bash
# 1. Create a clean virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install PyTorch
pip install torch

# 3. Run the standalone 1-block pretraining script
python python/train_tiny_gpt.py

# 4. Drag & drop 'tiny_gpt_weights.json' into the web UI's "Import .pt Weights" modal!
```

---

## 📈 Roadmap & Educational Progression

1. **Byte-Pair Encoding (BPE):** Transition from character-level tokens to subword BPE tokens.
2. **PyTorch Pretraining:** Run `python/train_tiny_gpt.py` on GPU/Colab for deep convergence.
3. **Supervised Fine-Tuning (SFT) & Memory:** Understand that conversational interview scripts are *product capabilities* achieved via instruction tuning or RAG/memory architectures over 1B–8B parameter models, not raw pretraining.
4. **Scale Up:** Expand to multi-layer nanoGPT on TinyStories and FineWeb-scale corpora.
