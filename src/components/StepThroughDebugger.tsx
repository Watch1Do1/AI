import React, { useState, useMemo } from 'react';
import { MicroGPT } from '../engine/transformer';
import { CharTokenizer } from '../engine/tokenizer';
import { ModelConfig } from '../types';
import { Layers, ArrowRight, ArrowLeft, Eye, Sparkles, Hash, Info, Play, Sliders, Check } from 'lucide-react';

interface StepThroughDebuggerProps {
  model: MicroGPT;
  tokenizer: CharTokenizer;
  config: ModelConfig;
}

export const StepThroughDebugger: React.FC<StepThroughDebuggerProps> = ({
  model,
  tokenizer,
  config
}) => {
  const [inputText, setInputText] = useState('AI');
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [selectedTokenPos, setSelectedTokenPos] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);

  // Encode tokens
  const tokenIds = useMemo(() => {
    const raw = inputText.trim() || 'A';
    return tokenizer.encode(raw).slice(0, Math.min(raw.length, config.blockSize));
  }, [inputText, tokenizer, config.blockSize]);

  // Keep selectedTokenPos in bounds
  const activeTokenPos = Math.min(selectedTokenPos, Math.max(0, tokenIds.length - 1));

  // Run detailed forward pass
  const stepData = useMemo(() => {
    if (tokenIds.length === 0) return null;
    return model.stepThroughDetailed(tokenIds);
  }, [model, tokenIds]);

  const tokenChars = useMemo(() => {
    return tokenIds.map(id => tokenizer.decode([id]));
  }, [tokenIds, tokenizer]);

  const STAGES = [
    {
      id: 0,
      name: '1. Embeddings',
      title: 'Token Embedding (wte) + Positional Encoding (wpe)',
      formula: 'x[t] = W_{te}[token_t] + W_{pe}[t]',
      shape: `(${tokenIds.length}, ${config.nEmbd})`,
      explanation: 'Discrete character token IDs are mapped to dense d_model vectors and added with learned sequence position encodings.'
    },
    {
      id: 1,
      name: '2. Q, K, V & Raw Logits',
      title: 'Attention Projections & Unmasked Dot-Product Scores',
      formula: 'Q = x W_q, \\; K = x W_k, \\; \\text{Scores} = \\frac{Q K^T}{\\sqrt{d_k}}',
      shape: `Heads: ${config.nHead} × (${tokenIds.length}, ${tokenIds.length})`,
      explanation: 'Each token queries other tokens by multiplying Query and Key vectors. Notice unmasked scores can be negative or large before softmax.'
    },
    {
      id: 2,
      name: '3. Causal Mask & Softmax',
      title: 'Causal Masking & Row-Normalized Softmax Probabilities',
      formula: 'A_{t, i} = \\text{softmax}\\left(\\frac{Q_t K_i^T}{\\sqrt{d_k}} + M_{t, i}\\right)',
      shape: `Heads: ${config.nHead} × (${tokenIds.length}, ${tokenIds.length})`,
      explanation: 'Future positions (i > t) are set to -∞ before softmax so the model cannot cheat. Every row sums strictly to 1.0.'
    },
    {
      id: 3,
      name: '4. Value Aggregation & Wo',
      title: 'Attention-Weighted Value Sum & Linear Output Projection',
      formula: '\\text{AttnOut}[t] = \\left( \\sum_{i \\le t} A_{t, i} V_i \\right) W_o',
      shape: `(${tokenIds.length}, ${config.nEmbd})`,
      explanation: 'Softmax attention weights mix the Value vectors (V). The multi-head outputs are concatenated and multiplied by Wo.'
    },
    {
      id: 4,
      name: '5. Residual & LayerNorm',
      title: 'First Skip Residual Connection',
      formula: 'x_{mid}[t] = x[t] + \\text{AttnOut}[t]',
      shape: `(${tokenIds.length}, ${config.nEmbd})`,
      explanation: 'The original input vector x[t] is added directly to the attention output. This residual gradient highway prevents vanishing gradients.'
    },
    {
      id: 5,
      name: '6. Position-Wise MLP',
      title: 'Feed-Forward Expansion (4× d_model) & ReLU/GELU Activation',
      formula: '\\text{MLP}(x_{mid}) = \\text{ReLU}(x_{mid} W_1 + b_1) W_2 + b_2',
      shape: `Hidden: (${tokenIds.length}, ${4 * config.nEmbd}) \\to (${tokenIds.length}, ${config.nEmbd})`,
      explanation: 'The MLP acts as key-value associative storage for learned patterns. Activations expand 4× and compress back.'
    },
    {
      id: 6,
      name: '7. LM Head & Logits',
      title: 'Unnormalized Next-Token Logits & Probabilities',
      formula: 'Z = x_{final}[T-1] W_{lm\\_head}, \\quad p = \\text{softmax}(Z)',
      shape: `(${config.vocabSize}) over vocabulary`,
      explanation: 'The final token representation is projected into a score for every possible character in the vocabulary to pick the next token.'
    }
  ];

  const activeStage = STAGES[currentStepIndex];

  // Helper to render a numeric vector grid
  const renderVectorSlice = (vec: Float32Array | number[], maxCols = 16, label?: string) => {
    const slice = Array.from(vec).slice(0, maxCols);
    return (
      <div className="space-y-1">
        {label && <div className="text-[11px] font-mono text-zinc-500 font-medium">{label}</div>}
        <div className="grid grid-cols-4 sm:grid-cols-8 gap-1 font-mono text-xs">
          {slice.map((val, idx) => {
            const isPos = val >= 0;
            const intensity = Math.min(1, Math.abs(val) * 1.5);
            const bg = isPos
              ? `rgba(16, 185, 129, ${Math.max(0.08, intensity * 0.4)})`
              : `rgba(239, 68, 68, ${Math.max(0.08, intensity * 0.4)})`;
            return (
              <div
                key={idx}
                style={{ backgroundColor: bg }}
                className="p-1.5 rounded border border-zinc-200 text-center select-all font-mono text-[11px] text-zinc-800"
                title={`dim [${idx}]: ${val.toFixed(5)}`}
              >
                {val.toFixed(2)}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      
      {/* Top Banner */}
      <div className="bg-zinc-900 text-white rounded-2xl p-6 border border-zinc-800 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-emerald-400 text-xs font-mono font-semibold">
            <Layers className="w-4 h-4" />
            <span>Interactive Forward Pass Inspector</span>
          </div>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">
            Step {currentStepIndex + 1} of {STAGES.length}
          </span>
        </div>
        <h2 className="text-xl font-bold text-zinc-100">
          Step Through a Transformer
        </h2>
        <p className="text-sm text-zinc-300 leading-relaxed max-w-3xl">
          Walk step-by-step through a genuine forward pass of the Decoder-Only Transformer. Inspect raw activation values, tensor shapes, attention matrices, and logit probability distributions at every layer.
        </p>

        {/* Input Bar & Presets */}
        <div className="pt-2 flex flex-col sm:flex-row sm:items-center gap-3">
          <div className="flex-1 flex items-center gap-2 bg-zinc-950 px-3 py-2 rounded-xl border border-zinc-700">
            <span className="text-xs text-zinc-400 font-mono">Input:</span>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              maxLength={config.blockSize}
              className="bg-transparent text-white font-mono text-sm focus:outline-none flex-1"
              placeholder="Type characters to step through..."
            />
          </div>
          <div className="flex items-center gap-1.5 text-xs text-zinc-400">
            <span>Examples:</span>
            {['AI', 'def', 'Hi', 'Why'].map(sample => (
              <button
                key={sample}
                onClick={() => setInputText(sample)}
                className="px-2.5 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-mono transition-colors"
              >
                {sample}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Stepper Navigation Bar */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-4 shadow-xs space-y-3">
        <div className="flex items-center justify-between gap-2 overflow-x-auto pb-1 no-scrollbar">
          {STAGES.map((s, idx) => {
            const isActive = currentStepIndex === idx;
            const isCompleted = currentStepIndex > idx;
            return (
              <button
                key={s.id}
                onClick={() => setCurrentStepIndex(idx)}
                className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-zinc-900 text-white font-semibold shadow-xs'
                    : isCompleted
                    ? 'bg-emerald-50 text-emerald-800 border border-emerald-200 hover:bg-emerald-100'
                    : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200'
                }`}
              >
                <span>{s.name}</span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-zinc-100">
          <button
            onClick={() => setCurrentStepIndex(Math.max(0, currentStepIndex - 1))}
            disabled={currentStepIndex === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-200 text-xs font-medium text-zinc-700 hover:bg-zinc-100 disabled:opacity-40 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Previous Stage</span>
          </button>

          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500 font-mono">
              Tokens:
            </span>
            <div className="flex items-center gap-1">
              {tokenChars.map((ch, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedTokenPos(idx)}
                  className={`px-2 py-0.5 rounded text-xs font-mono transition-colors ${
                    activeTokenPos === idx
                      ? 'bg-emerald-600 text-white font-bold'
                      : 'bg-zinc-100 text-zinc-700 hover:bg-zinc-200'
                  }`}
                >
                  t={idx} '{ch === ' ' ? '␣' : ch}'
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={() => setCurrentStepIndex(Math.min(STAGES.length - 1, currentStepIndex + 1))}
            disabled={currentStepIndex === STAGES.length - 1}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-xs font-medium text-white disabled:opacity-40 transition-colors"
          >
            <span>Next Stage</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Step Detail Stage */}
      {stepData && (
        <div className="bg-white rounded-2xl border border-zinc-200 p-6 shadow-xs space-y-6">
          
          {/* Stage Header Info */}
          <div className="space-y-2 pb-4 border-b border-zinc-100">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-lg font-bold text-zinc-900">{activeStage.title}</h3>
              <span className="px-2.5 py-1 rounded bg-zinc-100 font-mono text-xs text-zinc-700 font-medium">
                Shape: {activeStage.shape}
              </span>
            </div>
            <div className="p-2.5 rounded-lg bg-zinc-900 text-emerald-400 font-mono text-xs">
              {activeStage.formula}
            </div>
            <p className="text-xs text-zinc-600 leading-relaxed">
              {activeStage.explanation}
            </p>
          </div>

          {/* Dynamic Content Per Stage */}
          {currentStepIndex === 0 && (
            <div className="space-y-4">
              <div className="text-xs font-semibold text-zinc-800 flex items-center justify-between">
                <span>Inspecting Token at Position t={activeTokenPos} ('{tokenChars[activeTokenPos]}', ID={tokenIds[activeTokenPos]})</span>
                <span className="text-zinc-500 font-normal">First 16 dimensions shown</span>
              </div>
              <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 space-y-4">
                {renderVectorSlice(stepData.xTokens[activeTokenPos], 16, `Token Lookup: W_te[${tokenIds[activeTokenPos]}]`)}
                {renderVectorSlice(stepData.xPositions[activeTokenPos], 16, `Positional Encoding: W_pe[${activeTokenPos}]`)}
                <div className="pt-2 border-t border-zinc-200">
                  {renderVectorSlice(stepData.xCombined[activeTokenPos], 16, `Sum Combined: x[${activeTokenPos}] = W_te + W_pe`)}
                </div>
              </div>
            </div>
          )}

          {currentStepIndex === 1 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-800">
                  Raw Dot-Product Attention Scores (QKᵀ / √d) — Head {selectedHead}
                </span>
                <div className="flex items-center gap-1">
                  {Array.from({ length: config.nHead }).map((_, h) => (
                    <button
                      key={h}
                      onClick={() => setSelectedHead(h)}
                      className={`px-2 py-0.5 rounded text-xs font-mono ${
                        selectedHead === h ? 'bg-zinc-900 text-white font-bold' : 'bg-zinc-100 text-zinc-700'
                      }`}
                    >
                      Head {h}
                    </button>
                  ))}
                </div>
              </div>

              <div className="overflow-x-auto p-4 bg-zinc-50 rounded-xl border border-zinc-200">
                <table className="font-mono text-xs border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2 text-zinc-400 font-normal">Q \ K</th>
                      {tokenChars.map((ch, idx) => (
                        <th key={idx} className="p-2 text-zinc-700 text-center font-bold">
                          '{ch}' (t={idx})
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {stepData.rawAttentionScores[selectedHead]?.map((row, rIdx) => (
                      <tr key={rIdx}>
                        <td className="p-2 text-zinc-700 font-bold whitespace-nowrap">
                          '{tokenChars[rIdx]}' (t={rIdx})
                        </td>
                        {row.map((score, cIdx) => {
                          const isCausalMasked = cIdx > rIdx;
                          return (
                            <td
                              key={cIdx}
                              className={`p-2.5 text-center rounded border border-zinc-200 ${
                                isCausalMasked
                                  ? 'bg-zinc-200/50 text-zinc-400 line-through'
                                  : 'bg-white text-zinc-900 font-bold'
                              }`}
                            >
                              {isCausalMasked ? 'Masked' : score.toFixed(2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {currentStepIndex === 2 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-800">
                  Causal Softmax Matrix A — Head {selectedHead} (Row Sum = 1.0)
                </span>
                <div className="flex items-center gap-1">
                  {Array.from({ length: config.nHead }).map((_, h) => (
                    <button
                      key={h}
                      onClick={() => setSelectedHead(h)}
                      className={`px-2 py-0.5 rounded text-xs font-mono ${
                        selectedHead === h ? 'bg-zinc-900 text-white font-bold' : 'bg-zinc-100 text-zinc-700'
                      }`}
                    >
                      Head {h}
                    </button>
                  ))}
                </div>
              </div>

              <div className="overflow-x-auto p-4 bg-zinc-50 rounded-xl border border-zinc-200">
                <table className="font-mono text-xs border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2 text-zinc-400 font-normal">Query \ Key</th>
                      {tokenChars.map((ch, idx) => (
                        <th key={idx} className="p-2 text-zinc-700 text-center font-bold">
                          '{ch}' (t={idx})
                        </th>
                      ))}
                      <th className="p-2 text-zinc-400 text-center">Row Sum</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stepData.softmaxProbabilities[selectedHead]?.map((row, rIdx) => {
                      const sum = row.reduce((a, b) => a + b, 0);
                      return (
                        <tr key={rIdx}>
                          <td className="p-2 text-zinc-700 font-bold whitespace-nowrap">
                            '{tokenChars[rIdx]}' (t={rIdx})
                          </td>
                          {row.map((prob, cIdx) => {
                            const isMasked = cIdx > rIdx;
                            const alpha = Math.min(1, prob * 1.5);
                            return (
                              <td
                                key={cIdx}
                                style={{
                                  backgroundColor: isMasked
                                    ? undefined
                                    : `rgba(99, 102, 241, ${Math.max(0.08, alpha * 0.5)})`
                                }}
                                className={`p-2.5 text-center border border-zinc-200 ${
                                  isMasked ? 'bg-zinc-100 text-zinc-400' : 'text-zinc-900 font-bold'
                                }`}
                              >
                                {isMasked ? '0.00' : prob.toFixed(2)}
                              </td>
                            );
                          })}
                          <td className="p-2 text-center font-bold text-emerald-600">
                            {sum.toFixed(2)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {currentStepIndex === 3 && (
            <div className="space-y-4">
              <span className="text-xs font-semibold text-zinc-800 block">
                Value Aggregation & Linear Output (Wo) Projection at t={activeTokenPos}
              </span>
              <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 space-y-4">
                {renderVectorSlice(stepData.attnProj[activeTokenPos], 16, `Attention Output: AttnOut[${activeTokenPos}]`)}
              </div>
            </div>
          )}

          {currentStepIndex === 4 && (
            <div className="space-y-4">
              <span className="text-xs font-semibold text-zinc-800 block">
                Residual Connection Addition at t={activeTokenPos}
              </span>
              <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 space-y-3">
                {renderVectorSlice(stepData.xCombined[activeTokenPos], 16, `Input x[${activeTokenPos}]`)}
                <div className="text-xs font-mono text-zinc-400 text-center">+ (Residual Highway)</div>
                {renderVectorSlice(stepData.attnProj[activeTokenPos], 16, `Attention Output AttnOut[${activeTokenPos}]`)}
                <div className="pt-2 border-t border-zinc-200">
                  {renderVectorSlice(stepData.xMid[activeTokenPos], 16, `Output: x_mid[${activeTokenPos}] = x + AttnOut`)}
                </div>
              </div>
            </div>
          )}

          {currentStepIndex === 5 && (
            <div className="space-y-4">
              <span className="text-xs font-semibold text-zinc-800 block">
                Position-Wise Feed-Forward MLP at t={activeTokenPos}
              </span>
              <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 space-y-4">
                {renderVectorSlice(stepData.mlpActivations[activeTokenPos], 16, `MLP ReLU Activations (Expanded to 4× d_model = ${4 * config.nEmbd})`)}
                {renderVectorSlice(stepData.mlpOut[activeTokenPos], 16, `MLP Projection: W2 + b2`)}
                <div className="pt-2 border-t border-zinc-200">
                  {renderVectorSlice(stepData.xFinal[activeTokenPos], 16, `Final Layer Representation: x_final[${activeTokenPos}]`)}
                </div>
              </div>
            </div>
          )}

          {currentStepIndex === 6 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-800">
                  LM Head Logits & Top Next-Token Probability Candidates
                </span>
                <span className="text-xs font-mono text-indigo-600 font-semibold">
                  Shannon Entropy H = {stepData.entropy.toFixed(3)} bits
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-zinc-950 text-zinc-200 rounded-xl font-mono text-xs space-y-2 border border-zinc-800">
                  <div className="text-zinc-400 font-semibold mb-2">Top 5 Next-Character Candidates</div>
                  {stepData.topCandidates.slice(0, 5).map((cand, idx) => {
                    const char = tokenizer.decode([cand.idx]);
                    const pct = (cand.prob * 100).toFixed(1);
                    return (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between items-center text-xs">
                          <span className="font-bold text-white">
                            '{char === ' ' ? '␣' : char === '\n' ? '↵' : char}'
                            <span className="text-zinc-500 font-normal ml-2">ID: {cand.idx}</span>
                          </span>
                          <span className="text-emerald-400 font-bold">{pct}%</span>
                        </div>
                        <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-emerald-400 h-full rounded-full transition-all"
                            style={{ width: `${Math.max(2, cand.prob * 100)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="p-4 bg-zinc-50 rounded-xl border border-zinc-200 text-xs space-y-2.5">
                  <h4 className="font-bold text-zinc-900">How Next Token is Sampled</h4>
                  <p className="text-zinc-600 leading-relaxed">
                    1. The final vector <code className="font-mono text-zinc-800">x_final[T-1]</code> is multiplied by <code className="font-mono text-zinc-800">lmHead</code> to generate {config.vocabSize} raw logits.
                  </p>
                  <p className="text-zinc-600 leading-relaxed">
                    2. Applying <code className="font-mono text-zinc-800">softmax(logits / temperature)</code> creates a categorical probability distribution.
                  </p>
                  <p className="text-zinc-600 leading-relaxed">
                    3. Lower entropy (~1.0–2.0) means high certainty; higher entropy (&gt; 3.5) means diffuse uncertainty across tokens.
                  </p>
                </div>
              </div>
            </div>
          )}

        </div>
      )}

    </div>
  );
};
