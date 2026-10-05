import React, { useState } from 'react';
import { MicroGPT } from '../engine/transformer';
import { Activity, AlertTriangle, CheckCircle2, TrendingUp, BarChart2, Eye, ShieldCheck, Zap } from 'lucide-react';
import { GradientTelemetry } from '../types';

interface GradientActivationVisualizerProps {
  model: MicroGPT;
  telemetry: GradientTelemetry | null;
  step: number;
}

export const GradientActivationVisualizer: React.FC<GradientActivationVisualizerProps> = ({
  model,
  telemetry,
  step
}) => {
  const [activeHeatmapMode, setActiveHeatmapMode] = useState<'raw' | 'softmax'>('softmax');
  const [selectedLayer, setSelectedLayer] = useState<number>(0);

  const stats = telemetry?.activationStats || [
    { name: 'Token Embeds (x)', mean: 0.02, variance: 0.09, min: -0.65, max: 0.72, bins: [12, 28, 45, 60, 52, 35, 18, 8] },
    { name: 'Post-LayerNorm (xNorm)', mean: 0.00, variance: 1.01, min: -1.82, max: 2.14, bins: [8, 22, 54, 75, 70, 48, 20, 6] },
    { name: 'Post-Attn Residual (xMid)', mean: 0.04, variance: 0.88, min: -1.45, max: 1.95, bins: [10, 25, 48, 68, 65, 42, 19, 7] },
    { name: 'MLP Output (xFinal)', mean: 0.05, variance: 1.12, min: -2.10, max: 2.30, bins: [6, 18, 42, 70, 72, 45, 21, 9] }
  ];

  const totalNorm = telemetry?.totalNorm ?? 0.38;
  const isVanishing = telemetry?.isVanishing ?? (totalNorm < 1e-4);
  const isExploding = telemetry?.isExploding ?? (totalNorm > 4.5);

  const activeStats = stats[selectedLayer] || stats[0];
  const maxBinVal = Math.max(...activeStats.bins, 1);

  // Checkpoint attention heatmap
  const rawLogits = telemetry?.attentionLogitsHeatmap?.rawLogits || [];
  const softmaxProbs = telemetry?.attentionLogitsHeatmap?.softmaxProbs || [];

  return (
    <div className="space-y-6">
      
      {/* Banner */}
      <div className="bg-zinc-900 text-white rounded-2xl p-6 border border-zinc-800 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-rose-400 text-xs font-mono font-semibold">
          <Activity className="w-4 h-4" />
          <span>Internal Optimization Diagnostics</span>
        </div>
        <h2 className="text-xl font-bold text-zinc-100">
          Gradient & Activation Visualizer
        </h2>
        <p className="text-sm text-zinc-300 leading-relaxed max-w-3xl">
          Inspect layer-by-layer activation distributions, monitor L2 gradient norms over time, catch vanishing or exploding gradients, and compare pre-softmax attention logits against normalized probabilities.
        </p>
      </div>

      {/* Stability Health Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        
        <div className={`p-4 rounded-xl border flex items-center gap-3 ${
          isExploding
            ? 'bg-rose-50 border-rose-300 text-rose-950'
            : isVanishing
            ? 'bg-amber-50 border-amber-300 text-amber-950'
            : 'bg-emerald-50 border-emerald-300 text-emerald-950'
        }`}>
          {isExploding ? (
            <AlertTriangle className="w-6 h-6 text-rose-600 shrink-0" />
          ) : isVanishing ? (
            <AlertTriangle className="w-6 h-6 text-amber-600 shrink-0" />
          ) : (
            <ShieldCheck className="w-6 h-6 text-emerald-600 shrink-0" />
          )}
          <div>
            <div className="text-xs font-bold uppercase tracking-wider">Gradient Health</div>
            <div className="text-sm font-semibold">
              {isExploding ? 'Exploding Gradients Warning' : isVanishing ? 'Vanishing Gradients Warning' : 'Stable Gradient Flow'}
            </div>
            <div className="text-[11px] opacity-80 mt-0.5">
              {isExploding ? '||g|| > 4.5! Clipping active.' : isVanishing ? '||g|| < 1e-4! Updates stalled.' : 'Norm within nominal range [0.01 - 2.5]'}
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">Total Gradient L2 Norm ||g||</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">{totalNorm.toFixed(4)}</div>
          <div className="text-[11px] text-zinc-500 mt-1">Clipped at threshold [-1.0, 1.0]</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">Active Step Telemetry</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">Step {step}</div>
          <div className="text-[11px] text-zinc-500 mt-1">Updated on each backprop pass</div>
        </div>

      </div>

      {/* Layer-by-Layer Parameter Gradient Norm Breakdown */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <h3 className="text-sm font-bold text-zinc-900">Per-Layer Gradient Norm Breakdown (||∇W||)</h3>
        
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 font-mono text-xs">
          {[
            { label: 'Token Embed (W_te)', norm: telemetry?.wteNorm ?? 0.12 },
            { label: 'Pos Embed (W_pe)', norm: telemetry?.wpeNorm ?? 0.04 },
            { label: 'MLP Hidden (W_1)', norm: telemetry?.w1Norm ?? 0.28 },
            { label: 'MLP Proj (W_2)', norm: telemetry?.w2Norm ?? 0.21 },
            { label: 'LM Head (W_lm)', norm: telemetry?.lmHeadNorm ?? 0.35 }
          ].map((item, idx) => (
            <div key={idx} className="p-3 rounded-xl bg-zinc-50 border border-zinc-200 space-y-1.5">
              <span className="text-[11px] text-zinc-500 block truncate font-sans">{item.label}</span>
              <div className="text-lg font-bold text-zinc-900">{item.norm.toFixed(3)}</div>
              <div className="w-full bg-zinc-200 h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-zinc-900 h-full rounded-full"
                  style={{ width: `${Math.min(100, (item.norm / 1.0) * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Activation Distributions & Histograms */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100">
          <div>
            <h3 className="text-sm font-bold text-zinc-900">Activation Distributions</h3>
            <p className="text-xs text-zinc-500">Inspect mean, variance, and value spread across layers to diagnose dead ReLUs or saturation.</p>
          </div>

          <div className="flex items-center gap-1 overflow-x-auto">
            {stats.map((st, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedLayer(idx)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors whitespace-nowrap ${
                  selectedLayer === idx
                    ? 'bg-zinc-900 text-white font-bold'
                    : 'bg-zinc-100 text-zinc-700 hover:bg-zinc-200'
                }`}
              >
                {st.name}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 p-4 bg-zinc-950 text-white rounded-xl border border-zinc-800 space-y-3">
            <div className="flex justify-between items-center text-xs text-zinc-400 font-mono">
              <span>Value Histogram (8 Bins)</span>
              <span>Min: {activeStats.min.toFixed(2)} | Max: {activeStats.max.toFixed(2)}</span>
            </div>

            <div className="h-40 flex items-end gap-2 pt-4 px-2">
              {activeStats.bins.map((count, bIdx) => {
                const heightPct = (count / maxBinVal) * 100;
                return (
                  <div key={bIdx} className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end group">
                    <span className="text-[10px] font-mono text-zinc-500 opacity-0 group-hover:opacity-100 transition-opacity">
                      {count}
                    </span>
                    <div
                      style={{ height: `${Math.max(6, heightPct)}%` }}
                      className="w-full bg-emerald-500 hover:bg-emerald-400 rounded-t transition-all"
                    />
                    <span className="text-[9px] font-mono text-zinc-500">
                      B{bIdx + 1}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="p-4 bg-zinc-50 rounded-xl border border-zinc-200 text-xs space-y-3 font-mono">
            <h4 className="font-bold text-zinc-900 font-sans text-sm">Layer Summary: {activeStats.name}</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-zinc-500">Empirical Mean (μ):</span>
                <span className="font-bold text-zinc-900">{activeStats.mean.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Variance (σ²):</span>
                <span className="font-bold text-zinc-900">{activeStats.variance.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Std Deviation (σ):</span>
                <span className="font-bold text-zinc-900">{Math.sqrt(Math.max(0, activeStats.variance)).toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Extreme Range:</span>
                <span className="font-bold text-zinc-900">[{activeStats.min.toFixed(2)}, {activeStats.max.toFixed(2)}]</span>
              </div>
            </div>

            <p className="text-[11px] font-sans text-zinc-600 leading-normal pt-2 border-t border-zinc-200">
              {activeStats.variance > 2.5
                ? 'High variance indicates large internal features; LayerNorm will rescale before attention.'
                : activeStats.variance < 0.05
                ? 'Low variance suggests activations are compressed near zero (watch for dead neurons).'
                : 'Nominal activation variance around 1.0 indicates healthy residual scale.'}
            </p>
          </div>
        </div>
      </div>

      {/* Raw Attention Logits vs Softmax */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-zinc-100">
          <div>
            <h3 className="text-sm font-bold text-zinc-900">Attention Logits vs. Softmax Probabilities</h3>
            <p className="text-xs text-zinc-500">See raw dot-product scores (QKᵀ / √d) vs row-normalized probabilities.</p>
          </div>

          <div className="flex items-center gap-1 bg-zinc-100 p-1 rounded-lg text-xs font-medium">
            <button
              onClick={() => setActiveHeatmapMode('raw')}
              className={`px-3 py-1 rounded-md transition-all ${
                activeHeatmapMode === 'raw' ? 'bg-white text-zinc-900 font-bold shadow-xs' : 'text-zinc-600'
              }`}
            >
              Raw Logits (QKᵀ / √d)
            </button>
            <button
              onClick={() => setActiveHeatmapMode('softmax')}
              className={`px-3 py-1 rounded-md transition-all ${
                activeHeatmapMode === 'softmax' ? 'bg-white text-zinc-900 font-bold shadow-xs' : 'text-zinc-600'
              }`}
            >
              Softmax Probabilities
            </button>
          </div>
        </div>

        {softmaxProbs.length > 0 ? (
          <div className="overflow-x-auto p-4 bg-zinc-50 rounded-xl border border-zinc-200">
            <div className="font-mono text-xs text-zinc-600 mb-2">
              Viewing: {activeHeatmapMode === 'raw' ? 'Raw unscaled dot-product logits' : 'Row-normalized probabilities (sums to 1.0)'}
            </div>
            <div className="grid grid-cols-8 gap-1 max-w-sm">
              {(activeHeatmapMode === 'raw' ? rawLogits : softmaxProbs).slice(0, 8).map((row, r) =>
                row.slice(0, 8).map((val, c) => {
                  const isMasked = c > r;
                  const bg = isMasked
                    ? '#f4f4f5'
                    : activeHeatmapMode === 'raw'
                    ? val >= 0 ? `rgba(16, 185, 129, ${Math.min(1, Math.abs(val) * 0.3)})` : `rgba(239, 68, 68, ${Math.min(1, Math.abs(val) * 0.3)})`
                    : `rgba(99, 102, 241, ${Math.max(0.1, val * 1.5)})`;
                  return (
                    <div
                      key={`${r}-${c}`}
                      style={{ backgroundColor: bg }}
                      className="p-2 rounded border border-zinc-200 text-center font-mono text-[10px] text-zinc-800"
                    >
                      {isMasked ? '—' : val.toFixed(1)}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        ) : (
          <div className="p-8 text-center text-zinc-500 font-mono text-xs">
            Start training or execute a step in the visualizer to populate attention logits.
          </div>
        )}
      </div>

    </div>
  );
};
