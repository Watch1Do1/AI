import React, { useState } from 'react';
import { TrendingUp, Cpu, HardDrive, Database, Zap, Download, Copy, Check, ArrowRight } from 'lucide-react';

export const ScalingRoadmap: React.FC = () => {
  const [paramsM, setParamsM] = useState<number>(33); // 33 Million params (e.g. TinyStories GPT-4-mini)
  const [contextLen, setContextLen] = useState<number>(256);
  const [datasetTokensM, setDatasetTokensM] = useState<number>(500); // 500M tokens
  const [copiedCode, setCopiedCode] = useState(false);

  // Compute FLOPs = 6 * N * D (forward + backward)
  const totalFlops = 6 * (paramsM * 1e6) * (datasetTokensM * 1e6);
  const t4FlopsPerSec = 65e12; // ~65 TFLOPS FP16 on T4
  const a100FlopsPerSec = 312e12; // ~312 TFLOPS BF16 on A100
  
  const hoursOnT4 = totalFlops / (t4FlopsPerSec * 0.35 * 3600); // assume 35% MFU
  const hoursOnA100 = totalFlops / (a100FlopsPerSec * 0.45 * 3600); // assume 45% MFU

  // Memory footprint: Weights (FP16 = 2B/param) + Adam states (8B/param) + activations
  const modelVramGB = (paramsM * 1e6 * 16) / (1024 ** 3);

  const tinyStoriesPrepScript = `"""
prepare_tinystories.py
======================
Downloads and tokenizes the TinyStories dataset (~500M tokens) for nanoGPT training.
TinyStories was created by Ronen Eldan and Yuanzhi Li to show that 10M-33M parameter
models can learn fluent grammar and reasoning on synthetic children stories.

Usage:
  pip install datasets tiktoken numpy tqdm
  python prepare_tinystories.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# 1. Download TinyStories dataset from Hugging Face
print("--> Downloading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories")

# 2. Initialize GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token) # End of text
    out = {'ids': ids, 'len': len(ids)}
    return out

# 3. Tokenize train and validation splits
print("--> Tokenizing splits into binary memory-mapped files...")
for split in ['train', 'validation']:
    dset = dataset[split]
    filename = f"{split}.bin"
    
    total_tokens = sum(len(enc.encode_ordinary(t['text'])) + 1 for t in dset)
    print(f"Split {split}: {total_tokens:,} tokens")
    
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(total_tokens,))
    idx = 0
    for example in tqdm(dset):
        tokens = enc.encode_ordinary(example['text'])
        tokens.append(enc.eot_token)
        arr[idx : idx + len(tokens)] = tokens
        idx += len(tokens)
    arr.flush()
    print(f"Saved: {filename}")

print("\\nDone! You can now point nanoGPT's train.py to this directory.")`;

  const handleCopy = () => {
    navigator.clipboard.writeText(tinyStoriesPrepScript);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([tinyStoriesPrepScript], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'prepare_tinystories.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      
      {/* Overview Banner */}
      <div className="bg-sky-950 text-white rounded-2xl p-6 border border-sky-900 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-sky-300 text-xs font-mono font-semibold">
          <TrendingUp className="w-4 h-4 text-sky-400" />
          <span>Step 4 in the Roadmap</span>
        </div>
        <h2 className="text-xl font-bold text-sky-50">
          Scaling the Architecture — The nanoGPT & nanochat Path
        </h2>
        <p className="text-sm text-sky-200/90 leading-relaxed">
          Once you have verified the pretraining mechanics and built your fine-tuned/memory product, you can scale this exact decoder-only architecture: increase layers, widen embedding dimensions, extend the context window to 512–1024 tokens, and train on real datasets like <strong>TinyStories</strong> (~500M tokens) or <strong>FineWeb</strong>.
        </p>
      </div>

      {/* 4 Scaling Stages Card */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <h3 className="text-sm font-semibold text-zinc-900">The 4 Evolutionary Scale Stages</h3>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          
          <div className="p-3.5 rounded-xl bg-zinc-50 border border-zinc-200 space-y-1.5">
            <div className="flex items-center justify-between font-bold text-zinc-800">
              <span>Stage 1: Toy Lab</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-zinc-200 text-zinc-700">Current</span>
            </div>
            <div className="font-mono text-zinc-600 text-[11px]">~100K Params · 32 Context</div>
            <p className="text-zinc-500 text-[11px] leading-relaxed">
              Trains in browser or CPU in minutes. Verifies next-token loss drop and attention mechanics.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-sky-50/80 border border-sky-200 space-y-1.5">
            <div className="flex items-center justify-between font-bold text-sky-950">
              <span>Stage 2: TinyStories</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-sky-200 text-sky-800 font-bold">Recommended</span>
            </div>
            <div className="font-mono text-sky-900 text-[11px]">10M–33M Params · 256 Context</div>
            <p className="text-sky-800 text-[11px] leading-relaxed">
              Trains on 1x free Colab T4 GPU in ~2 hours. Generates coherent English stories with character dialogue.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-indigo-50/80 border border-indigo-200 space-y-1.5">
            <div className="flex items-center justify-between font-bold text-indigo-950">
              <span>Stage 3: nanoGPT</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-indigo-200 text-indigo-800 font-medium">GPT-2 Scale</span>
            </div>
            <div className="font-mono text-indigo-900 text-[11px]">124M Params · 1024 Context</div>
            <p className="text-indigo-800 text-[11px] leading-relaxed">
              Trained on OpenWebText or FineWeb-edu sample. Requires ~1-2 days on an 8x A100 node (~$40).
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-emerald-50/80 border border-emerald-200 space-y-1.5">
            <div className="flex items-center justify-between font-bold text-emerald-950">
              <span>Stage 4: Open Model</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-200 text-emerald-800 font-medium">Production</span>
            </div>
            <div className="font-mono text-emerald-900 text-[11px]">1B–8B Params · 4096 Context</div>
            <p className="text-emerald-800 text-[11px] leading-relaxed">
              Full Llama / Mistral foundation scale. Pretrained on trillions of tokens across compute clusters.
            </p>
          </div>

        </div>
      </div>

      {/* Interactive Compute & VRAM Budget Calculator */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-zinc-100">
          <div>
            <h3 className="text-sm font-semibold text-zinc-900">Compute Budget & Training Time Calculator</h3>
            <p className="text-xs text-zinc-500">Based on Chinchilla compute law: Compute = 6 × N × D FLOPs</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-1 text-xs">
            <div className="flex justify-between text-zinc-700 font-medium">
              <span>Parameters:</span>
              <span className="font-mono font-bold text-zinc-900">{paramsM} Million</span>
            </div>
            <input
              type="range"
              min="5"
              max="125"
              step="5"
              value={paramsM}
              onChange={(e) => setParamsM(Number(e.target.value))}
              className="w-full accent-zinc-900"
            />
          </div>

          <div className="space-y-1 text-xs">
            <div className="flex justify-between text-zinc-700 font-medium">
              <span>Context Length:</span>
              <span className="font-mono font-bold text-zinc-900">{contextLen} Tokens</span>
            </div>
            <input
              type="range"
              min="64"
              max="1024"
              step="64"
              value={contextLen}
              onChange={(e) => setContextLen(Number(e.target.value))}
              className="w-full accent-zinc-900"
            />
          </div>

          <div className="space-y-1 text-xs">
            <div className="flex justify-between text-zinc-700 font-medium">
              <span>Dataset Size:</span>
              <span className="font-mono font-bold text-zinc-900">{datasetTokensM} Million Tokens</span>
            </div>
            <input
              type="range"
              min="50"
              max="2000"
              step="50"
              value={datasetTokensM}
              onChange={(e) => setDatasetTokensM(Number(e.target.value))}
              className="w-full accent-zinc-900"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 bg-zinc-50 rounded-xl border border-zinc-200 font-mono text-xs">
          <div>
            <div className="text-zinc-500 text-[10px] uppercase">Total Compute</div>
            <div className="text-sm font-bold text-zinc-900">{(totalFlops / 1e18).toFixed(2)} ExaFLOPs</div>
          </div>
          <div>
            <div className="text-zinc-500 text-[10px] uppercase">Training VRAM</div>
            <div className="text-sm font-bold text-zinc-900">~{modelVramGB.toFixed(1)} GB</div>
          </div>
          <div>
            <div className="text-zinc-500 text-[10px] uppercase">Colab T4 GPU Time</div>
            <div className="text-sm font-bold text-emerald-600">~{hoursOnT4.toFixed(1)} Hours</div>
          </div>
          <div>
            <div className="text-zinc-500 text-[10px] uppercase">1× A100 GPU Time</div>
            <div className="text-sm font-bold text-indigo-600">~{hoursOnA100.toFixed(1)} Hours</div>
          </div>
        </div>
      </div>

      {/* TinyStories Data Prep Code */}
      <div className="bg-zinc-950 rounded-2xl border border-zinc-800 overflow-hidden shadow-sm">
        <div className="flex items-center justify-between px-4 py-3 bg-zinc-900 border-b border-zinc-800 text-xs font-mono">
          <span className="font-semibold text-zinc-300">prepare_tinystories.py</span>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
            >
              {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copiedCode ? 'Copied' : 'Copy'}</span>
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download</span>
            </button>
          </div>
        </div>

        <div className="p-4 max-h-[380px] overflow-y-auto text-zinc-300 font-mono text-xs leading-relaxed">
          <pre>{tinyStoriesPrepScript}</pre>
        </div>
      </div>

    </div>
  );
};
