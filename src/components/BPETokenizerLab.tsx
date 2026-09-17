import React, { useState, useMemo } from 'react';
import { Type, Cpu, Sparkles, ArrowRight, Zap, RefreshCw, BarChart2, Layers, Check } from 'lucide-react';
import { BPETokenizer } from '../engine/bpe';

interface BPETokenizerLabProps {
  corpus: string;
}

export const BPETokenizerLab: React.FC<BPETokenizerLabProps> = ({ corpus }) => {
  const [numMerges, setNumMerges] = useState<number>(35);
  const [testText, setTestText] = useState<string>(
    "INTERVIEWER: Could you explain why causal masking prevents attending to future tokens in autoregressive models?"
  );

  // Train BPE with current merge count
  const bpe = useMemo(() => {
    return new BPETokenizer(corpus, numMerges);
  }, [corpus, numMerges]);

  const bpeTokens = useMemo(() => {
    return bpe.tokenize(testText);
  }, [bpe, testText]);

  const charTokens = testText.split('');
  const compressionRatio = charTokens.length / Math.max(1, bpeTokens.length);
  const attentionSavings = Math.pow(compressionRatio, 2);

  // Color palette for token chips
  const colors = [
    'bg-emerald-100 text-emerald-900 border-emerald-300',
    'bg-indigo-100 text-indigo-900 border-indigo-300',
    'bg-amber-100 text-amber-900 border-amber-300',
    'bg-purple-100 text-purple-900 border-purple-300',
    'bg-rose-100 text-rose-900 border-rose-300',
    'bg-sky-100 text-sky-900 border-sky-300',
    'bg-teal-100 text-teal-900 border-teal-300'
  ];

  return (
    <div className="space-y-6">
      
      {/* Banner */}
      <div className="bg-zinc-900 text-white rounded-2xl p-6 border border-zinc-800 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-emerald-400 text-xs font-mono font-semibold">
          <Type className="w-4 h-4" />
          <span>Step 1 in the Roadmap</span>
        </div>
        <h2 className="text-xl font-bold text-zinc-100">
          Byte-Pair Encoding (BPE) — "Same Model, Less of a Toy"
        </h2>
        <p className="text-sm text-zinc-300 leading-relaxed">
          Character tokenizers treat every single letter as a token, forcing your tiny model to waste context length on spelling words letter-by-letter. BPE algorithmically merges the most frequent adjacent character pairs into subword tokens. This compresses sequences by ~3.5×, allowing the exact same transformer architecture to see <strong>3.5× more content</strong> and reducing attention quadratic compute by <strong>~12×</strong>.
        </p>
      </div>

      {/* Comparison Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">Character Tokens</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">{charTokens.length}</div>
          <div className="text-[11px] text-zinc-600 mt-1">1 token per character</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">BPE Subword Tokens</div>
          <div className="text-2xl font-bold font-mono text-emerald-600">{bpeTokens.length}</div>
          <div className="text-[11px] text-zinc-600 mt-1">{numMerges} pair merges active</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">Context Compression</div>
          <div className="text-2xl font-bold font-mono text-indigo-600">{compressionRatio.toFixed(1)}×</div>
          <div className="text-[11px] text-zinc-600 mt-1">More text in same context window</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">O(T²) Attention Cost Drop</div>
          <div className="text-2xl font-bold font-mono text-amber-600">{attentionSavings.toFixed(0)}×</div>
          <div className="text-[11px] text-zinc-600 mt-1">Quadratic compute efficiency</div>
        </div>
      </div>

      {/* Interactive Tokenizer Comparison */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-zinc-100">
          <div>
            <h3 className="text-base font-semibold text-zinc-900">Side-by-Side Tokenization Visualizer</h3>
            <p className="text-xs text-zinc-500">Inspect how a raw sentence is chunked into tokens</p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-600 font-medium">BPE Merges:</span>
            <input
              type="range"
              min="5"
              max="80"
              step="5"
              value={numMerges}
              onChange={(e) => setNumMerges(Number(e.target.value))}
              className="w-24 accent-zinc-900"
            />
            <span className="font-mono text-xs font-bold text-zinc-900 w-8">{numMerges}</span>
          </div>
        </div>

        {/* Test Sentence Input */}
        <div>
          <label className="block text-xs font-medium text-zinc-700 mb-1">Test Sentence:</label>
          <input
            type="text"
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            className="w-full bg-zinc-50 border border-zinc-300 rounded-lg px-3 py-2 text-xs font-mono text-zinc-900 focus:outline-none focus:ring-2 focus:ring-zinc-900"
          />
        </div>

        {/* Visual Token Chips: BPE Subwords */}
        <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-zinc-800 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-emerald-600" />
              <span>BPE Subword Tokens ({bpeTokens.length} tokens):</span>
            </span>
            <span className="text-[11px] text-emerald-700 font-mono font-medium">
              Compressed by {compressionRatio.toFixed(1)}×
            </span>
          </div>

          <div className="flex flex-wrap gap-1.5 font-mono text-xs">
            {bpeTokens.map((tok, idx) => (
              <span
                key={idx}
                className={`px-2 py-1 rounded-md border font-medium ${colors[idx % colors.length]}`}
              >
                {tok === ' ' ? '␣' : tok}
              </span>
            ))}
          </div>
        </div>

        {/* Visual Token Chips: Character-Level */}
        <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-zinc-700">
              Character-Level Tokens ({charTokens.length} tokens):
            </span>
            <span className="text-[11px] text-zinc-500 font-mono">
              Uncompressed baseline
            </span>
          </div>

          <div className="flex flex-wrap gap-1 font-mono text-[11px]">
            {charTokens.map((ch, idx) => (
              <span
                key={idx}
                className="px-1.5 py-0.5 rounded bg-white text-zinc-700 border border-zinc-200"
              >
                {ch === ' ' ? '␣' : ch}
              </span>
            ))}
          </div>
        </div>

      </div>

      {/* Active BPE Merges Table */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-zinc-900 flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-600" />
            <span>Top Algorithmically Discovered Pair Merges</span>
          </h3>
          <span className="text-xs text-zinc-500 font-mono">
            Total Vocabulary: {bpe.vocab.length} subwords
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 max-h-48 overflow-y-auto pr-1">
          {bpe.merges.slice(0, 30).map((merge, idx) => (
            <div
              key={idx}
              className="p-2 rounded-lg bg-zinc-50 border border-zinc-200 font-mono text-xs flex items-center justify-between"
            >
              <div className="flex items-center gap-1 text-zinc-800">
                <span className="bg-white px-1 rounded border border-zinc-200">{merge.pair[0]}</span>
                <span className="text-zinc-600">+</span>
                <span className="bg-white px-1 rounded border border-zinc-200">{merge.pair[1]}</span>
                <ArrowRight className="w-3 h-3 text-zinc-600" />
                <span className="font-bold text-emerald-600">{merge.merged}</span>
              </div>
              <span className="text-[10px] text-zinc-600">×{merge.count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* PyTorch Integration Code */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-3">
        <h3 className="text-sm font-semibold text-zinc-900">
          How to Swap Character Tokenizer with BPE in PyTorch (tiktoken)
        </h3>
        <p className="text-xs text-zinc-600">
          In your Python script, simply replace the character encoder with OpenAI's standard fast BPE tokenizer (`tiktoken`):
        </p>

        <div className="p-4 rounded-xl bg-zinc-950 text-zinc-200 font-mono text-xs overflow-x-auto">
          <pre>{`# 1. Install standard fast BPE:
# pip install tiktoken

import tiktoken

# Use standard GPT-2 byte-pair encoding
enc = tiktoken.get_encoding("gpt2")

# Tokenize and decode:
encoded_tokens = enc.encode("INTERVIEWER: Could you explain causal masking?")
decoded_text = enc.decode(encoded_tokens)

print(f"Characters: {len('INTERVIEWER: Could you explain causal masking?')}")
print(f"BPE Tokens: {len(encoded_tokens)}")
print(f"Vocab size: {enc.n_vocab:,}") # 50,257 subwords`}</pre>
        </div>
      </div>

    </div>
  );
};
