import React, { useState } from 'react';
import { Layers, ArrowDown, Info, Cpu, Zap, Activity } from 'lucide-react';
import { ModelConfig } from '../types';

interface ArchitectureDiagramProps {
  config: ModelConfig;
}

interface ComponentDetail {
  title: string;
  tensorShape: string;
  formula: string;
  description: string;
}

export const ArchitectureDiagram: React.FC<ArchitectureDiagramProps> = ({ config }) => {
  const [selectedComponent, setSelectedComponent] = useState<string>('attention');

  const { blockSize, nEmbd, nHead, vocabSize, nLayer } = config;
  const headDim = Math.floor(nEmbd / nHead);

  const details: Record<string, ComponentDetail> = {
    tokens: {
      title: 'Input Token Sequence',
      tensorShape: `(Batch, Time) = (${config.batchSize}, ${blockSize})`,
      formula: 'x \\in \\{0, 1, ..., V-1\\}^T',
      description: 'Raw integer token indices encoded by the tokenizer from the input text.'
    },
    embedding: {
      title: 'Token + Positional Embeddings',
      tensorShape: `(Batch, Time, n_embd) = (${config.batchSize}, ${blockSize}, ${nEmbd})`,
      formula: 'h_0 = W_{te}[x] + W_{pe}[:T]',
      description: 'Learned lookup matrices convert discrete token IDs into dense vector representations and inject temporal sequence order.'
    },
    attention: {
      title: 'Masked Causal Multi-Head Attention',
      tensorShape: `Heads: ${nHead} × (Time, Time) = ${nHead} × (${blockSize}, ${blockSize})`,
      formula: 'A = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right) V',
      description: 'Calculates dynamic relevance between all past tokens in the context. The upper triangular causal mask M guarantees tokens cannot peek into future steps.'
    },
    mlp: {
      title: 'Feed-Forward Network (MLP)',
      tensorShape: `Hidden: (Batch, Time, 4 × n_embd) = (${config.batchSize}, ${blockSize}, ${4 * nEmbd})`,
      formula: '\\text{FFN}(x) = \\text{GELU}(x W_1 + b_1) W_2 + b_2',
      description: 'Position-wise non-linear feature transformation that expands embedding dimension by 4x to store associative patterns and syntactic transformations.'
    },
    head: {
      title: 'Language Model Head (LM Head)',
      tensorShape: `Logits: (Batch, Time, vocab_size) = (${config.batchSize}, ${blockSize}, ${vocabSize})`,
      formula: 'Z = h_{final} W_{lm\\_head}',
      description: 'Projects the final hidden states into unnormalized log-probabilities over the entire character vocabulary for next-token prediction.'
    }
  };

  const activeDetail = details[selectedComponent] || details.attention;

  return (
    <div className="space-y-6">
      
      {/* Header Overview */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-2">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-600" />
          <h2 className="text-base font-semibold text-zinc-900">
            Decoder-Only Transformer Architecture (GPT)
          </h2>
        </div>
        <p className="text-xs text-zinc-500 leading-relaxed">
          The exact neural network topology implemented in both our browser MicroGPT engine and the downloadable PyTorch script. Click any architectural stage below to inspect its mathematical formulation and tensor dimensions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Visual Stack Flow */}
        <div className="lg:col-span-7 bg-zinc-950 rounded-2xl p-6 border border-zinc-800 text-white space-y-3 font-mono text-xs">
          
          {/* Stage 1: Input Tokens */}
          <div
            onClick={() => setSelectedComponent('tokens')}
            className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
              selectedComponent === 'tokens'
                ? 'border-emerald-400 bg-zinc-900 ring-1 ring-emerald-400'
                : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'
            }`}
          >
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              <span className="font-semibold text-zinc-200">1. Input Tokens: idx</span>
            </div>
            <span className="text-[11px] text-zinc-400">[{config.batchSize}, {blockSize}]</span>
          </div>

          <div className="flex justify-center text-zinc-600">
            <ArrowDown className="w-4 h-4" />
          </div>

          {/* Stage 2: Embeddings */}
          <div
            onClick={() => setSelectedComponent('embedding')}
            className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
              selectedComponent === 'embedding'
                ? 'border-emerald-400 bg-zinc-900 ring-1 ring-emerald-400'
                : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'
            }`}
          >
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-indigo-400" />
              <span className="font-semibold text-zinc-200">2. Token Embed (wte) + Pos Embed (wpe)</span>
            </div>
            <span className="text-[11px] text-zinc-400">[{config.batchSize}, {blockSize}, {nEmbd}]</span>
          </div>

          <div className="flex justify-center text-zinc-600">
            <ArrowDown className="w-4 h-4" />
          </div>

          {/* Stage 3: Transformer Block */}
          <div className="p-4 rounded-xl border-2 border-dashed border-zinc-700 bg-zinc-900/40 space-y-3">
            <div className="text-[11px] uppercase tracking-wider text-emerald-400 font-bold flex items-center justify-between">
              <span>Transformer Decoder Block (×{nLayer})</span>
              <span className="text-zinc-500 font-normal">Residual Connections Enabled</span>
            </div>

            {/* Sub: Attention */}
            <div
              onClick={() => setSelectedComponent('attention')}
              className={`p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between ${
                selectedComponent === 'attention'
                  ? 'border-emerald-400 bg-zinc-800 ring-1 ring-emerald-400'
                  : 'border-zinc-700 bg-zinc-800/80 hover:border-zinc-600'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                <span className="font-semibold text-zinc-100">LayerNorm + Causal Multi-Head Attention</span>
              </div>
              <span className="text-[11px] text-emerald-400 font-semibold">{nHead} Heads (dim={headDim})</span>
            </div>

            {/* Sub: MLP */}
            <div
              onClick={() => setSelectedComponent('mlp')}
              className={`p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between ${
                selectedComponent === 'mlp'
                  ? 'border-emerald-400 bg-zinc-800 ring-1 ring-emerald-400'
                  : 'border-zinc-700 bg-zinc-800/80 hover:border-zinc-600'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-400" />
                <span className="font-semibold text-zinc-100">LayerNorm + Position-Wise MLP (GELU)</span>
              </div>
              <span className="text-[11px] text-amber-400 font-semibold">4 × {nEmbd} = {4 * nEmbd}</span>
            </div>
          </div>

          <div className="flex justify-center text-zinc-600">
            <ArrowDown className="w-4 h-4" />
          </div>

          {/* Stage 4: LM Head & Logits */}
          <div
            onClick={() => setSelectedComponent('head')}
            className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
              selectedComponent === 'head'
                ? 'border-emerald-400 bg-zinc-900 ring-1 ring-emerald-400'
                : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'
            }`}
          >
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-rose-400" />
              <span className="font-semibold text-zinc-200">3. Final LayerNorm + LM Linear Head</span>
            </div>
            <span className="text-[11px] text-zinc-400">Logits [{config.batchSize}, {blockSize}, {vocabSize}]</span>
          </div>

        </div>

        {/* Stage Inspector Card */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
            <div className="flex items-center gap-2">
              <Info className="w-4 h-4 text-emerald-600" />
              <h3 className="text-sm font-semibold text-zinc-900">Component Details</h3>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="text-zinc-500 font-medium">Stage Name:</label>
                <div className="text-sm font-bold text-zinc-900 mt-0.5">{activeDetail.title}</div>
              </div>

              <div>
                <label className="text-zinc-500 font-medium">Tensor Dimension:</label>
                <div className="p-2 rounded bg-zinc-100 font-mono text-zinc-900 font-semibold mt-0.5">
                  {activeDetail.tensorShape}
                </div>
              </div>

              <div>
                <label className="text-zinc-500 font-medium">Mathematical Formula:</label>
                <div className="p-2 rounded bg-zinc-900 font-mono text-emerald-400 text-xs mt-0.5">
                  {activeDetail.formula}
                </div>
              </div>

              <div>
                <label className="text-zinc-500 font-medium">Operational Role:</label>
                <p className="text-zinc-600 leading-relaxed mt-0.5">
                  {activeDetail.description}
                </p>
              </div>
            </div>
          </div>

          {/* Parameter Count Math */}
          <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-2 text-xs">
            <h4 className="font-semibold text-zinc-900">Parameter Formula Breakdown</h4>
            <div className="space-y-1 font-mono text-[11px] text-zinc-600">
              <div className="flex justify-between">
                <span>wte: Vocab × d_model</span>
                <span>{vocabSize} × {nEmbd} = {(vocabSize * nEmbd).toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>wpe: Block × d_model</span>
                <span>{blockSize} × {nEmbd} = {(blockSize * nEmbd).toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Attention (Q,K,V,O)</span>
                <span>4 × {nEmbd}² = {(4 * nEmbd * nEmbd).toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>MLP (W1, W2, B1, B2)</span>
                <span>8 × {nEmbd}² + 5 × {nEmbd} = {(8 * nEmbd * nEmbd + 5 * nEmbd).toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>LM Head: d_model × Vocab</span>
                <span>{nEmbd} × {vocabSize} = {(nEmbd * vocabSize).toLocaleString()}</span>
              </div>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
};
