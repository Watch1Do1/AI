import React, { useState } from 'react';
import { MicroGPT } from '../engine/transformer';
import { CharTokenizer } from '../engine/tokenizer';
import { CheckpointSnapshot } from '../types';
import { History, Play, RotateCcw, Sparkles, Check, ChevronRight, BookmarkPlus } from 'lucide-react';

interface CheckpointTimelineProps {
  model: MicroGPT;
  tokenizer: CharTokenizer;
  snapshots: CheckpointSnapshot[];
  onTakeSnapshot: () => void;
  onRestoreSnapshot: (snapshot: CheckpointSnapshot) => void;
  currentStep: number;
}

export const CheckpointTimeline: React.FC<CheckpointTimelineProps> = ({
  model,
  tokenizer,
  snapshots,
  onTakeSnapshot,
  onRestoreSnapshot,
  currentStep
}) => {
  const [selectedIndex, setSelectedIndex] = useState<number>(Math.max(0, snapshots.length - 1));

  // If no snapshots yet, provide a mock/default baseline set for educational scrubber exploration
  const effectiveSnapshots: CheckpointSnapshot[] = snapshots.length > 0 ? snapshots : [
    {
      step: 0,
      timestamp: Date.now() - 300000,
      loss: 4.17,
      valLoss: 4.21,
      perplexity: 64.7,
      samplePrompt: 'INTERVIEWER:',
      sampleText: 'INTERVIEWER: zq#9! vK.. ?x[[ 8w09a m;LLq',
      attentionMaps: [{ headIndex: 0, matrix: [[1, 0], [0.5, 0.5]] }]
    },
    {
      step: 100,
      timestamp: Date.now() - 200000,
      loss: 3.12,
      valLoss: 3.25,
      perplexity: 22.6,
      samplePrompt: 'INTERVIEWER:',
      sampleText: 'INTERVIEWER: and the the the in and to to model in the',
      attentionMaps: [{ headIndex: 0, matrix: [[1, 0], [0.3, 0.7]] }]
    },
    {
      step: 300,
      timestamp: Date.now() - 100000,
      loss: 2.38,
      valLoss: 2.45,
      perplexity: 10.8,
      samplePrompt: 'INTERVIEWER:',
      sampleText: 'INTERVIEWER: What is the model attention layers in the system?',
      attentionMaps: [{ headIndex: 0, matrix: [[1, 0], [0.1, 0.9]] }]
    },
    {
      step: 500,
      timestamp: Date.now(),
      loss: 1.84,
      valLoss: 1.95,
      perplexity: 6.3,
      samplePrompt: 'INTERVIEWER:',
      sampleText: 'INTERVIEWER: Could you explain why causal masking prevents attending to future tokens?',
      attentionMaps: [{ headIndex: 0, matrix: [[1, 0], [0.05, 0.95]] }]
    }
  ];

  const activeSnapshot = effectiveSnapshots[Math.min(selectedIndex, effectiveSnapshots.length - 1)];

  return (
    <div className="space-y-6">
      
      {/* Banner */}
      <div className="bg-zinc-900 text-white rounded-2xl p-6 border border-zinc-800 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sky-400 text-xs font-mono font-semibold">
            <History className="w-4 h-4" />
            <span>Training Checkpoint Timeline & Evolution</span>
          </div>
          <button
            onClick={onTakeSnapshot}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-medium transition-colors border border-zinc-700"
          >
            <BookmarkPlus className="w-3.5 h-3.5 text-emerald-400" />
            <span>Save Snapshot at Step {currentStep}</span>
          </button>
        </div>
        <h2 className="text-xl font-bold text-zinc-100">
          Checkpoint Timeline & Scrubbing
        </h2>
        <p className="text-sm text-zinc-300 leading-relaxed max-w-3xl">
          Drag the timeline scrubber to watch language emerge across training. Witness the classic phase transition from random uniform noise to consonant repetitions, common words, and syntax fluency.
        </p>
      </div>

      {/* Interactive Timeline Scrubber Card */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-6 shadow-xs space-y-6">
        
        {/* Slider Controls */}
        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-zinc-500">Timeline Scrubber:</span>
            <span className="font-bold text-zinc-900">
              Viewing Step {activeSnapshot.step} ({selectedIndex + 1} of {effectiveSnapshots.length})
            </span>
          </div>

          <input
            type="range"
            min={0}
            max={effectiveSnapshots.length - 1}
            step={1}
            value={selectedIndex}
            onChange={(e) => setSelectedIndex(Number(e.target.value))}
            className="w-full accent-zinc-900 cursor-pointer h-2 bg-zinc-200 rounded-lg"
          />

          {/* Step markers below slider */}
          <div className="flex justify-between text-[11px] font-mono text-zinc-500 pt-1">
            {effectiveSnapshots.map((sn, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedIndex(idx)}
                className={`transition-colors ${
                  selectedIndex === idx ? 'text-zinc-900 font-bold underline' : 'hover:text-zinc-800'
                }`}
              >
                Step {sn.step}
              </button>
            ))}
          </div>
        </div>

        {/* Telemetry Stats for Selected Snapshot */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
            <span className="text-[11px] text-zinc-500 block mb-1">Pretraining Step</span>
            <div className="text-xl font-bold font-mono text-zinc-900">Step {activeSnapshot.step}</div>
          </div>
          <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
            <span className="text-[11px] text-zinc-500 block mb-1">Train Loss</span>
            <div className="text-xl font-bold font-mono text-emerald-600">{activeSnapshot.loss.toFixed(3)}</div>
          </div>
          <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
            <span className="text-[11px] text-zinc-500 block mb-1">Val Loss (10%)</span>
            <div className="text-xl font-bold font-mono text-sky-600">{activeSnapshot.valLoss.toFixed(3)}</div>
          </div>
          <div className="p-3 bg-zinc-50 rounded-xl border border-zinc-200">
            <span className="text-[11px] text-zinc-500 block mb-1">Perplexity</span>
            <div className="text-xl font-bold font-mono text-zinc-900">{activeSnapshot.perplexity.toFixed(1)}</div>
          </div>
        </div>

        {/* Output Replay at Selected Step */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-semibold text-zinc-800">
            <span>Model Output Sample at Step {activeSnapshot.step}:</span>
            <span className="text-zinc-500 font-normal">
              {activeSnapshot.step === 0
                ? 'Random Weights Phase'
                : activeSnapshot.step < 200
                ? 'Early Grammar Phase'
                : 'Syntax Fluency Phase'}
            </span>
          </div>
          <div className="p-4 bg-zinc-950 text-zinc-100 rounded-xl font-mono text-sm leading-relaxed border border-zinc-800 min-h-[90px] whitespace-pre-wrap">
            <span className="text-emerald-400 font-bold">{activeSnapshot.samplePrompt}</span>
            <span className="text-zinc-300">{activeSnapshot.sampleText.slice(activeSnapshot.samplePrompt.length)}</span>
          </div>
        </div>

        {/* Restore Action */}
        {activeSnapshot.weightsData && (
          <div className="pt-3 border-t border-zinc-100 flex justify-end">
            <button
              onClick={() => onRestoreSnapshot(activeSnapshot)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-semibold transition-colors shadow-xs"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Restore Weights from Step {activeSnapshot.step}</span>
            </button>
          </div>
        )}

      </div>

    </div>
  );
};
