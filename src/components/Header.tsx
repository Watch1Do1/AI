import React from 'react';
import {
  Cpu,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  Terminal,
  BookOpen,
  Layers,
  Database,
  Type,
  MessageSquare,
  TrendingUp,
  Activity,
  Scale,
  History,
  Eye,
  HelpCircle,
  Code2
} from 'lucide-react';

export type AppTab =
  | 'workbench'
  | 'step-debugger'
  | 'compare-gpt2'
  | 'gradient-viz'
  | 'checkpoints'
  | 'explainability'
  | 'browser-vs-pytorch'
  | 'why-this-works'
  | 'bpe'
  | 'pytorch'
  | 'sft'
  | 'memory'
  | 'scaling'
  | 'dataset'
  | 'architecture';

interface HeaderProps {
  activeTab: AppTab;
  setActiveTab: (tab: AppTab) => void;
  isTraining: boolean;
  onToggleTraining: () => void;
  onReset: () => void;
  paramCount: number;
  currentLoss: number | null;
  step: number;
  weightSource?: 'browser' | 'pytorch';
  onOpenWizard?: () => void;
  onOpenTour?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  isTraining,
  onToggleTraining,
  onReset,
  paramCount,
  currentLoss,
  step,
  weightSource = 'browser',
  onOpenWizard,
  onOpenTour
}) => {
  return (
    <header className="border-b border-zinc-200 bg-white/95 backdrop-blur sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          
          {/* Logo & Title */}
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 rounded-xl bg-zinc-900 text-white flex items-center justify-center font-mono font-bold text-lg shadow-sm">
              <Cpu className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-zinc-900 tracking-tight text-lg">TinyGPT</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-medium border border-emerald-200">
                  Pretraining Lab
                </span>
                {weightSource === 'pytorch' && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 font-semibold border border-indigo-200 font-mono">
                    .pt Checkpoint Loaded
                  </span>
                )}
              </div>
              <p className="text-xs text-zinc-500 hidden sm:block">
                Interactive Causal Language Model Laboratory & Mechanics Inspector
              </p>
            </div>
          </div>

          {/* Quick Telemetry Pills */}
          <div className="hidden lg:flex items-center gap-2 text-xs font-mono">
            <div className="px-2.5 py-1 rounded-md bg-zinc-100 text-zinc-700 border border-zinc-200/80">
              <span className="text-zinc-500">Params:</span> <span className="font-semibold text-zinc-900">{paramCount.toLocaleString()}</span>
            </div>
            <div className="px-2.5 py-1 rounded-md bg-zinc-100 text-zinc-700 border border-zinc-200/80">
              <span className="text-zinc-500">Step:</span> <span className="font-semibold text-zinc-900">{step}</span>
            </div>
            <div className="px-2.5 py-1 rounded-md bg-zinc-100 text-zinc-700 border border-zinc-200/80">
              <span className="text-zinc-500">Loss:</span>{' '}
              <span className={`font-semibold ${currentLoss && currentLoss < 2.5 ? 'text-emerald-600' : 'text-zinc-900'}`}>
                {currentLoss !== null ? currentLoss.toFixed(3) : '—'}
              </span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            {onOpenTour && (
              <button
                onClick={onOpenTour}
                className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-zinc-700 bg-zinc-100 hover:bg-zinc-200 transition-colors border border-zinc-200"
                title="Open Interactive Tour & Pathways"
              >
                <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
                <span>Tour</span>
              </button>
            )}

            {onOpenWizard && (
              <button
                onClick={onOpenWizard}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold text-emerald-950 bg-emerald-400 hover:bg-emerald-300 transition-colors shadow-xs"
                title="Launch guided 4-step training wizard"
              >
                <Sparkles className="w-3.5 h-3.5 text-emerald-950" />
                <span>Train Wizard</span>
              </button>
            )}

            <button
              id="header-toggle-training-btn"
              onClick={onToggleTraining}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all shadow-sm ${
                isTraining
                  ? 'bg-amber-600 hover:bg-amber-700 text-white'
                  : 'bg-zinc-900 hover:bg-zinc-800 text-white'
              }`}
            >
              {isTraining ? (
                <>
                  <Pause className="w-3.5 h-3.5" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 text-emerald-300" />
                  <span>{step > 0 ? 'Resume' : 'Pretrain'}</span>
                </>
              )}
            </button>

            <button
              id="header-reset-btn"
              onClick={onReset}
              title="Reset weights to random initial state"
              className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-800 hover:bg-zinc-100 transition-colors border border-zinc-200"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1 overflow-x-auto no-scrollbar -mb-px border-t border-zinc-100 pt-1 text-xs">
          
          <button
            id="nav-tab-workbench"
            onClick={() => setActiveTab('workbench')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'workbench'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Visualizer Lab</span>
          </button>

          <button
            id="nav-tab-step-debugger"
            onClick={() => setActiveTab('step-debugger')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'step-debugger'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-emerald-600" />
            <span>Step Debugger</span>
          </button>

          <button
            id="nav-tab-compare-gpt2"
            onClick={() => setActiveTab('compare-gpt2')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'compare-gpt2'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Scale className="w-3.5 h-3.5 text-purple-600" />
            <span>vs. GPT-2</span>
          </button>

          <button
            id="nav-tab-gradient-viz"
            onClick={() => setActiveTab('gradient-viz')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'gradient-viz'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Activity className="w-3.5 h-3.5 text-rose-500" />
            <span>Gradients & Norms</span>
          </button>

          <button
            id="nav-tab-checkpoints"
            onClick={() => setActiveTab('checkpoints')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'checkpoints'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <History className="w-3.5 h-3.5 text-sky-600" />
            <span>Checkpoints</span>
          </button>

          <button
            id="nav-tab-explainability"
            onClick={() => setActiveTab('explainability')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'explainability'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Eye className="w-3.5 h-3.5 text-indigo-500" />
            <span>Explainability</span>
          </button>

          <button
            id="nav-tab-browser-vs-pytorch"
            onClick={() => setActiveTab('browser-vs-pytorch')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'browser-vs-pytorch'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Code2 className="w-3.5 h-3.5 text-teal-600" />
            <span>Browser vs PyTorch</span>
          </button>

          <button
            id="nav-tab-why-this-works"
            onClick={() => setActiveTab('why-this-works')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'why-this-works'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <HelpCircle className="w-3.5 h-3.5 text-amber-600" />
            <span>Why This Works</span>
          </button>

          <button
            id="nav-tab-bpe"
            onClick={() => setActiveTab('bpe')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'bpe'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Type className="w-3.5 h-3.5 text-emerald-600" />
            <span>BPE Tokenizer</span>
          </button>

          <button
            id="nav-tab-pytorch"
            onClick={() => setActiveTab('pytorch')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'pytorch'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Terminal className="w-3.5 h-3.5 text-indigo-600" />
            <span>PyTorch Script</span>
          </button>

          <button
            id="nav-tab-dataset"
            onClick={() => setActiveTab('dataset')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'dataset'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Corpus</span>
          </button>

          <button
            id="nav-tab-architecture"
            onClick={() => setActiveTab('architecture')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'architecture'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Architecture</span>
          </button>

          <button
            id="nav-tab-sft"
            onClick={() => setActiveTab('sft')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'sft'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5 text-amber-600" />
            <span>SFT LoRA</span>
          </button>

          <button
            id="nav-tab-memory"
            onClick={() => setActiveTab('memory')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'memory'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <Database className="w-3.5 h-3.5 text-indigo-500" />
            <span>Memory Bridge</span>
          </button>

          <button
            id="nav-tab-scaling"
            onClick={() => setActiveTab('scaling')}
            className={`flex items-center gap-1.5 px-3 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === 'scaling'
                ? 'border-zinc-900 text-zinc-900 font-semibold'
                : 'border-transparent text-zinc-500 hover:text-zinc-800 hover:border-zinc-300'
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5 text-sky-600" />
            <span>Scaling Roadmap</span>
          </button>

        </div>

      </div>
    </header>
  );
};
