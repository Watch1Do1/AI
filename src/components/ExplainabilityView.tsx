import React, { useState } from 'react';
import { MicroGPT } from '../engine/transformer';
import { CharTokenizer } from '../engine/tokenizer';
import { Search, Eye, Sparkles, HelpCircle, BarChart3, AlertCircle, Check } from 'lucide-react';

interface ExplainabilityViewProps {
  model: MicroGPT;
  tokenizer: CharTokenizer;
}

export const ExplainabilityView: React.FC<ExplainabilityViewProps> = ({ model, tokenizer }) => {
  const [testPrompt, setTestPrompt] = useState('def train_');

  const tokens = tokenizer.encode(testPrompt.slice(0, model.config.blockSize));
  const detailed = model.stepThroughDetailed(tokens);

  // Entropy calculation
  const entropy = detailed.entropy;
  const topCandidates = detailed.topCandidates;
  const topChosen = topCandidates[0];
  const chosenChar = tokenizer.decode([topChosen?.idx ?? 0]);

  // Derive pseudo-saliency / attribution for earlier tokens using attention weights of the last token
  const lastTokenIdx = Math.max(0, tokens.length - 1);
  const head0 = detailed.softmaxProbabilities[0]?.[lastTokenIdx] || [];
  const head1 = detailed.softmaxProbabilities[1]?.[lastTokenIdx] || [];
  const saliency = tokens.map((_, i) => ((head0[i] || 0) + (head1[i] || 0)) / 2);

  // Certainty gauge
  const getCertaintyLabel = (ent: number) => {
    if (ent < 1.5) return { label: 'High Certainty', color: 'text-emerald-600 bg-emerald-50 border-emerald-200' };
    if (ent < 3.0) return { label: 'Moderate Confidence', color: 'text-sky-600 bg-sky-50 border-sky-200' };
    return { label: 'Diffuse / High Uncertainty', color: 'text-amber-600 bg-amber-50 border-amber-200' };
  };

  const certainty = getCertaintyLabel(entropy);

  return (
    <div className="space-y-6">
      
      {/* Banner */}
      <div className="bg-zinc-900 text-white rounded-2xl p-6 border border-zinc-800 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-emerald-400 text-xs font-mono font-semibold">
          <Eye className="w-4 h-4" />
          <span>Mechanistic Interpretability</span>
        </div>
        <h2 className="text-xl font-bold text-zinc-100">
          Model Explainability & Token Attribution
        </h2>
        <p className="text-sm text-zinc-300 leading-relaxed max-w-3xl">
          Deconstruct why Ted predicted a specific token. Inspect Shannon entropy, evaluate top-k candidate probability bars, and trace token attribution back to prior prompt tokens through attention routing.
        </p>

        {/* Prompt Input */}
        <div className="pt-2 flex flex-col sm:flex-row sm:items-center gap-3">
          <div className="flex-1 flex items-center gap-2 bg-zinc-950 px-3 py-2 rounded-xl border border-zinc-700">
            <span className="text-xs text-zinc-400 font-mono">Prompt Context:</span>
            <input
              type="text"
              value={testPrompt}
              onChange={(e) => setTestPrompt(e.target.value)}
              className="bg-transparent text-white font-mono text-sm focus:outline-none flex-1"
            />
          </div>
          <div className="flex items-center gap-1.5 text-xs text-zinc-400">
            {['def train_', 'INTERVIEWER: W', 'First Citizen:'].map(sample => (
              <button
                key={sample}
                onClick={() => setTestPrompt(sample)}
                className="px-2.5 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-mono transition-colors"
              >
                {sample}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Top Telemetry Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        
        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs space-y-1">
          <div className="text-xs text-zinc-500 font-medium">Predicted Next Token</div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-zinc-900">
              '{chosenChar === ' ' ? '␣' : chosenChar === '\n' ? '↵' : chosenChar}'
            </span>
            <span className="text-xs text-emerald-600 font-semibold font-mono">
              ({(topChosen.prob * 100).toFixed(1)}% prob)
            </span>
          </div>
          <div className="text-[11px] text-zinc-500">Argmax of LM Head logits</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs space-y-1">
          <div className="text-xs text-zinc-500 font-medium">Shannon Logit Entropy H(p)</div>
          <div className="text-3xl font-bold font-mono text-zinc-900">{entropy.toFixed(3)} bits</div>
          <div className="text-[11px] text-zinc-500">-Σ p_i log2(p_i) over V={model.config.vocabSize}</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs space-y-2">
          <div className="text-xs text-zinc-500 font-medium">Model Certainty Level</div>
          <div className={`px-3 py-1.5 rounded-lg border text-xs font-semibold inline-block ${certainty.color}`}>
            {certainty.label}
          </div>
          <div className="text-[11px] text-zinc-500">
            {entropy < 2.0 ? 'Sharp probability concentration' : 'Diffuse probability mass across multiple tokens'}
          </div>
        </div>

      </div>

      {/* Token Importance Saliency Attribution */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-zinc-100">
          <div>
            <h3 className="text-sm font-bold text-zinc-900">Context Saliency Attribution</h3>
            <p className="text-xs text-zinc-500">Which input characters had the strongest causal attention driving the next prediction?</p>
          </div>
          <span className="text-xs text-zinc-400 font-mono">Darker = Higher Influence</span>
        </div>

        <div className="flex flex-wrap gap-1.5 p-4 bg-zinc-50 rounded-xl border border-zinc-200 font-mono text-sm">
          {tokens.map((tokId, idx) => {
            const ch = tokenizer.decode([tokId]);
            const score = saliency[idx] || 0;
            const bgIntensity = Math.min(0.9, score * 2.5);
            return (
              <div
                key={idx}
                style={{
                  backgroundColor: `rgba(16, 185, 129, ${Math.max(0.08, bgIntensity)})`
                }}
                className="px-2 py-1 rounded border border-zinc-200 text-center font-bold text-zinc-900 relative group cursor-pointer"
              >
                <span>{ch === ' ' ? '␣' : ch === '\n' ? '↵' : ch}</span>
                <span className="absolute -top-7 left-1/2 -translate-x-1/2 hidden group-hover:block bg-zinc-900 text-white text-[10px] px-1.5 py-0.5 rounded shadow whitespace-nowrap z-10">
                  Influence: {(score * 100).toFixed(0)}%
                </span>
              </div>
            );
          })}
          <div className="px-2 py-1 rounded border-2 border-dashed border-emerald-500 text-emerald-700 font-bold bg-emerald-50 animate-pulse">
            👉 '{chosenChar === ' ' ? '␣' : chosenChar}'
          </div>
        </div>
      </div>

      {/* Top Candidate Distribution Bars */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <h3 className="text-sm font-bold text-zinc-900">Top-8 Candidate Probabilities</h3>

        <div className="space-y-2.5 font-mono text-xs">
          {topCandidates.slice(0, 8).map((cand, idx) => {
            const ch = tokenizer.decode([cand.idx]);
            const pct = (cand.prob * 100).toFixed(1);
            return (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-zinc-900 text-sm w-8 text-center bg-zinc-100 rounded py-0.5">
                      '{ch === ' ' ? '␣' : ch === '\n' ? '↵' : ch}'
                    </span>
                    <span className="text-zinc-500 font-sans text-xs">Logit: {cand.logit.toFixed(2)}</span>
                  </div>
                  <span className="font-bold text-zinc-900">{pct}%</span>
                </div>
                <div className="w-full bg-zinc-100 h-2 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      idx === 0 ? 'bg-emerald-500' : 'bg-zinc-400'
                    }`}
                    style={{ width: `${Math.max(1, cand.prob * 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
};
