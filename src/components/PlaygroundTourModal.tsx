import React from 'react';
import { Sparkles, Layers, Cpu, Terminal, Scale, BookOpen, ArrowRight, X, Play } from 'lucide-react';
import { AppTab } from './Header';

interface PlaygroundTourModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectTab: (tab: AppTab) => void;
  onOpenWizard: () => void;
}

export const PlaygroundTourModal: React.FC<PlaygroundTourModalProps> = ({
  isOpen,
  onClose,
  onSelectTab,
  onOpenWizard
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/65 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl max-w-3xl w-full border border-zinc-200 shadow-2xl overflow-hidden flex flex-col max-h-[92vh]">
        
        {/* Hero Banner */}
        <div className="p-6 md:p-8 bg-zinc-950 text-white relative overflow-hidden">
          <div className="relative z-10 flex items-start justify-between">
            <div className="space-y-2 max-w-xl">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950 border border-emerald-800 text-emerald-300 text-xs font-mono font-medium">
                <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                <span>TinyGPT Pretraining Lab</span>
              </div>
              <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
                The Smallest "From Scratch" Transformer Playground
              </h2>
              <p className="text-zinc-300 text-xs md:text-sm leading-relaxed">
                An interactive laboratory for the minimal causal language model pretraining experiment. From raw text bytes to multi-head self-attention, loss curves, and autoregressive token generation.
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="absolute -right-10 -bottom-10 w-64 h-64 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none" />
        </div>

        {/* 4 Interactive Tracks */}
        <div className="p-6 md:p-8 overflow-y-auto flex-1 space-y-6 text-xs">
          <div>
            <h3 className="text-sm font-bold text-zinc-900">Explore Interactive Tracks</h3>
            <p className="text-zinc-500">Pick any pathway to start exploring or training right now.</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            
            {/* Track 1: Wizard */}
            <div
              onClick={() => {
                onClose();
                onOpenWizard();
              }}
              className="p-4 rounded-2xl border border-emerald-200 bg-emerald-50/40 hover:bg-emerald-50 hover:border-emerald-300 transition-all cursor-pointer space-y-2 group shadow-xs"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 font-bold text-zinc-900">
                  <Play className="w-4 h-4 text-emerald-600 fill-current" />
                  <span>1. Train Your Own GPT</span>
                </div>
                <ArrowRight className="w-4 h-4 text-emerald-600 group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-zinc-600 leading-relaxed text-[11px]">
                A guided 4-step wizard: paste text, choose 1-click presets (NanoGPT, GPT-2 Tiny), train live, and sample tokens.
              </p>
            </div>

            {/* Track 2: Step-Through Debugger */}
            <div
              onClick={() => {
                onClose();
                onSelectTab('step-debugger');
              }}
              className="p-4 rounded-2xl border border-zinc-200 bg-white hover:border-zinc-300 hover:bg-zinc-50 transition-all cursor-pointer space-y-2 group shadow-xs"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 font-bold text-zinc-900">
                  <Layers className="w-4 h-4 text-indigo-600" />
                  <span>2. Step Through a Transformer</span>
                </div>
                <ArrowRight className="w-4 h-4 text-zinc-400 group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-zinc-600 leading-relaxed text-[11px]">
                Inspect intermediate vectors across all 7 stages: Embeddings → Attention Logits → Softmax → Weighted Sum → MLP → Residual → Logits.
              </p>
            </div>

            {/* Track 3: Compare GPT-2 */}
            <div
              onClick={() => {
                onClose();
                onSelectTab('compare-gpt2');
              }}
              className="p-4 rounded-2xl border border-zinc-200 bg-white hover:border-zinc-300 hover:bg-zinc-50 transition-all cursor-pointer space-y-2 group shadow-xs"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 font-bold text-zinc-900">
                  <Scale className="w-4 h-4 text-purple-600" />
                  <span>3. Compare Against GPT-2</span>
                </div>
                <ArrowRight className="w-4 h-4 text-zinc-400 group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-zinc-600 leading-relaxed text-[11px]">
                Side-by-side benchmark comparing 1-block TinyGPT against OpenAI's 124M 12-layer GPT-2 with Shannon entropy metrics.
              </p>
            </div>

            {/* Track 4: PyTorch & Colab */}
            <div
              onClick={() => {
                onClose();
                onSelectTab('pytorch');
              }}
              className="p-4 rounded-2xl border border-zinc-200 bg-white hover:border-zinc-300 hover:bg-zinc-50 transition-all cursor-pointer space-y-2 group shadow-xs"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 font-bold text-zinc-900">
                  <Terminal className="w-4 h-4 text-emerald-600" />
                  <span>4. PyTorch & Colab Source</span>
                </div>
                <ArrowRight className="w-4 h-4 text-zinc-400 group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-zinc-600 leading-relaxed text-[11px]">
                Standalone 1-file Python pretraining script (<code className="font-mono text-zinc-800">train_tiny_gpt.py</code>) and 1-click Google Colab notebook export.
              </p>
            </div>

          </div>

          {/* Quickstart Action Bar */}
          <div className="p-4 bg-zinc-100 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
            <div className="text-zinc-600 font-mono">
              Ready to jump into the lab?
            </div>
            <button
              onClick={() => {
                onClose();
                onSelectTab('workbench');
              }}
              className="px-5 py-2.5 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-white font-semibold transition-colors shadow-xs"
            >
              Open Visualizer Workbench
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};
