import React, { useState } from 'react';
import { Sparkles, ArrowRight, ArrowLeft, Play, Pause, Check, Upload, BookOpen, Sliders, Cpu, X, FileText, RotateCcw } from 'lucide-react';
import { ModelConfig, CorpusPreset, ConfigPresetItem } from '../types';
import { CORPUS_PRESETS } from '../engine/datasets';

interface TrainWizardModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: ModelConfig;
  onUpdateConfig: (config: ModelConfig) => void;
  onSelectCorpus: (text: string, title: string) => void;
  currentCorpusTitle: string;
  onRunSteps: (count: number) => void;
  onGenerate: (prompt: string, temp: number, topK: number, maxTokens: number) => { text: string };
  currentLoss: number | null;
  step: number;
}

export const PRESET_CONFIGS: ConfigPresetItem[] = [
  {
    id: 'nanogpt',
    name: 'NanoGPT (Karpathy Default)',
    badge: '1 Block / Classic',
    description: 'The standard Karpathy tutorial setup: 32 context window, 32 embed dim, 2 heads.',
    blockSize: 32,
    nEmbd: 32,
    nHead: 2,
    lr: 0.003,
    batchSize: 4
  },
  {
    id: 'gpt2_tiny',
    name: 'GPT-2 Tiny Config',
    badge: 'Denser / 4 Heads',
    description: 'Wider context and higher capacity: 64 context window, 48 embed dim, 4 heads.',
    blockSize: 64,
    nEmbd: 48,
    nHead: 4,
    lr: 0.002,
    batchSize: 4
  },
  {
    id: 'gptj_micro',
    name: 'GPT-J Micro Config',
    badge: 'Ultra Fast',
    description: 'Lightweight and nimble: 32 context, 24 embed dim, 2 heads for maximum frames/sec.',
    blockSize: 32,
    nEmbd: 24,
    nHead: 2,
    lr: 0.005,
    batchSize: 4
  },
  {
    id: 'tinystories',
    name: 'TinyStories Config',
    badge: 'High Reasoning',
    description: '64 context length, 32 embed dim, 4 heads modeled after Eldan & Li synthetic story benchmark.',
    blockSize: 64,
    nEmbd: 32,
    nHead: 4,
    lr: 0.002,
    batchSize: 4
  }
];

export const TrainWizardModal: React.FC<TrainWizardModalProps> = ({
  isOpen,
  onClose,
  config,
  onUpdateConfig,
  onSelectCorpus,
  currentCorpusTitle,
  onRunSteps,
  onGenerate,
  currentLoss,
  step
}) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedCorpusId, setSelectedCorpusId] = useState(CORPUS_PRESETS[0].id);
  const [customText, setCustomText] = useState('');
  const [customTitle, setCustomTitle] = useState('My Custom Corpus');
  const [targetTrainSteps, setTargetTrainSteps] = useState(50);
  const [isTrainingInModal, setIsTrainingInModal] = useState(false);
  const [promptText, setPromptText] = useState('INTERVIEWER:');
  const [generatedSample, setGeneratedSample] = useState('');

  if (!isOpen) return null;

  const handleApplyPreset = (preset: ConfigPresetItem) => {
    onUpdateConfig({
      ...config,
      blockSize: preset.blockSize,
      nEmbd: preset.nEmbd,
      nHead: preset.nHead,
      lr: preset.lr,
      batchSize: preset.batchSize
    });
  };

  const handleSelectCorpusInternal = (c: CorpusPreset) => {
    setSelectedCorpusId(c.id);
    onSelectCorpus(c.text, c.title);
  };

  const handleApplyCustomCorpus = () => {
    if (!customText.trim()) return;
    onSelectCorpus(customText, customTitle);
    setSelectedCorpusId('custom');
  };

  const handleRunBatchTraining = () => {
    setIsTrainingInModal(true);
    onRunSteps(targetTrainSteps);
    setTimeout(() => {
      setIsTrainingInModal(false);
    }, 600);
  };

  const handleTestGenerate = () => {
    const res = onGenerate(promptText, 0.7, 8, 80);
    setGeneratedSample(res.text);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-2xl w-full border border-zinc-200 shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        
        {/* Header */}
        <div className="p-5 border-b border-zinc-100 flex items-center justify-between bg-zinc-950 text-white">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-500 text-zinc-950 flex items-center justify-center font-bold">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-bold">Train Your Own GPT Wizard</h2>
              <p className="text-xs text-zinc-400">Guided 4-step setup from raw text to autoregressive generation</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* 4 Step Progress Header */}
        <div className="grid grid-cols-4 border-b border-zinc-200 text-xs font-medium">
          {[
            { num: 1, name: '1. Corpus' },
            { num: 2, name: '2. Config' },
            { num: 3, name: '3. Pretrain' },
            { num: 4, name: '4. Generate' }
          ].map(s => (
            <div
              key={s.num}
              className={`py-3 px-2 text-center border-b-2 transition-colors ${
                currentStep === s.num
                  ? 'border-emerald-500 text-emerald-700 bg-emerald-50/50 font-bold'
                  : currentStep > s.num
                  ? 'border-zinc-300 text-zinc-700 font-semibold'
                  : 'border-transparent text-zinc-400'
              }`}
            >
              {s.name}
            </div>
          ))}
        </div>

        {/* Step Body */}
        <div className="p-6 overflow-y-auto flex-1 text-xs space-y-4">
          
          {/* STEP 1: Corpus */}
          {currentStep === 1 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-bold text-zinc-900">Step 1: Choose or Paste Training Text</h3>
                <p className="text-zinc-500">Pick a curated domain corpus or paste your own custom text to train the language model.</p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {CORPUS_PRESETS.map(c => (
                  <div
                    key={c.id}
                    onClick={() => handleSelectCorpusInternal(c)}
                    className={`p-3 rounded-xl border cursor-pointer transition-all ${
                      selectedCorpusId === c.id
                        ? 'border-emerald-500 bg-emerald-50/40 ring-1 ring-emerald-500'
                        : 'border-zinc-200 hover:border-zinc-300 bg-zinc-50/50'
                    }`}
                  >
                    <div className="font-bold text-zinc-900 flex items-center justify-between">
                      <span>{c.title}</span>
                      {selectedCorpusId === c.id && <Check className="w-3.5 h-3.5 text-emerald-600" />}
                    </div>
                    <p className="text-[11px] text-zinc-500 mt-1 line-clamp-2">{c.description}</p>
                  </div>
                ))}
              </div>

              <div className="pt-2 border-t border-zinc-100 space-y-2">
                <span className="font-semibold text-zinc-800">Or Paste Custom Text:</span>
                <textarea
                  value={customText}
                  onChange={(e) => setCustomText(e.target.value)}
                  placeholder="Paste interview dialogue, technical documents, or custom text here..."
                  rows={3}
                  className="w-full p-2.5 rounded-lg border border-zinc-300 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-zinc-900"
                />
                {customText.trim() && (
                  <button
                    onClick={handleApplyCustomCorpus}
                    className="px-3 py-1.5 rounded-lg bg-zinc-900 text-white font-medium hover:bg-zinc-800 transition-colors"
                  >
                    Apply Custom Text ({customText.length} chars)
                  </button>
                )}
              </div>

              <div className="p-3 bg-zinc-100 rounded-xl font-mono text-[11px] text-zinc-700">
                Active Selection: <span className="font-bold text-zinc-900">{currentCorpusTitle}</span>
              </div>
            </div>
          )}

          {/* STEP 2: Config & Presets */}
          {currentStep === 2 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-bold text-zinc-900">Step 2: Choose Hyperparameter Preset</h3>
                <p className="text-zinc-500">Select a 1-click architecture preset or fine-tune individual dimensions.</p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {PRESET_CONFIGS.map(preset => {
                  const isMatching =
                    config.blockSize === preset.blockSize &&
                    config.nEmbd === preset.nEmbd &&
                    config.nHead === preset.nHead;

                  return (
                    <div
                      key={preset.id}
                      onClick={() => handleApplyPreset(preset)}
                      className={`p-3 rounded-xl border cursor-pointer transition-all ${
                        isMatching
                          ? 'border-emerald-500 bg-emerald-50/50 ring-1 ring-emerald-500'
                          : 'border-zinc-200 hover:border-zinc-300 bg-white'
                      }`}
                    >
                      <div className="flex items-center justify-between font-bold text-zinc-900">
                        <span>{preset.name}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 font-normal">
                          {preset.badge}
                        </span>
                      </div>
                      <p className="text-[11px] text-zinc-500 mt-1">{preset.description}</p>
                      <div className="mt-2 font-mono text-[10px] text-zinc-600 flex gap-2">
                        <span>T={preset.blockSize}</span>
                        <span>d={preset.nEmbd}</span>
                        <span>h={preset.nHead}</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="p-4 bg-zinc-50 rounded-xl border border-zinc-200 grid grid-cols-3 gap-3 font-mono text-xs">
                <div>
                  <label className="text-zinc-500 block mb-1">Block Size (T)</label>
                  <select
                    value={config.blockSize}
                    onChange={(e) => onUpdateConfig({ ...config, blockSize: Number(e.target.value) })}
                    className="w-full bg-white border border-zinc-300 rounded p-1.5 text-zinc-900"
                  >
                    <option value={16}>16 tokens</option>
                    <option value={32}>32 tokens</option>
                    <option value={64}>64 tokens</option>
                  </select>
                </div>

                <div>
                  <label className="text-zinc-500 block mb-1">Embed Dim (d)</label>
                  <select
                    value={config.nEmbd}
                    onChange={(e) => onUpdateConfig({ ...config, nEmbd: Number(e.target.value) })}
                    className="w-full bg-white border border-zinc-300 rounded p-1.5 text-zinc-900"
                  >
                    <option value={24}>24 dim</option>
                    <option value={32}>32 dim</option>
                    <option value={48}>48 dim</option>
                  </select>
                </div>

                <div>
                  <label className="text-zinc-500 block mb-1">Heads (h)</label>
                  <select
                    value={config.nHead}
                    onChange={(e) => onUpdateConfig({ ...config, nHead: Number(e.target.value) })}
                    className="w-full bg-white border border-zinc-300 rounded p-1.5 text-zinc-900"
                  >
                    <option value={2}>2 Heads</option>
                    <option value={4}>4 Heads</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* STEP 3: Pretrain */}
          {currentStep === 3 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-bold text-zinc-900">Step 3: Run Live Pretraining</h3>
                <p className="text-zinc-500">Execute training steps right in your browser to minimize cross-entropy loss.</p>
              </div>

              <div className="p-4 bg-zinc-950 text-white rounded-xl border border-zinc-800 flex items-center justify-between font-mono">
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-zinc-400">Current Step</div>
                  <div className="text-2xl font-bold">{step}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-zinc-400">Current Loss</div>
                  <div className={`text-2xl font-bold ${currentLoss && currentLoss < 2.5 ? 'text-emerald-400' : 'text-zinc-200'}`}>
                    {currentLoss !== null ? currentLoss.toFixed(3) : '—'}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-zinc-400">Target</div>
                  <div className="text-xs text-emerald-400 font-semibold">&lt; 2.200 (Syntax fluency)</div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="font-semibold text-zinc-700">Train Iterations:</span>
                {[20, 50, 100, 200].map(cnt => (
                  <button
                    key={cnt}
                    onClick={() => setTargetTrainSteps(cnt)}
                    className={`px-3 py-1.5 rounded-lg border font-mono transition-colors ${
                      targetTrainSteps === cnt
                        ? 'border-zinc-900 bg-zinc-900 text-white font-bold'
                        : 'border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-100'
                    }`}
                  >
                    +{cnt} Steps
                  </button>
                ))}
              </div>

              <button
                onClick={handleRunBatchTraining}
                disabled={isTrainingInModal}
                className="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-sm flex items-center justify-center gap-2 transition-all shadow-md disabled:opacity-50"
              >
                {isTrainingInModal ? (
                  <>
                    <RotateCcw className="w-4 h-4 animate-spin" />
                    <span>Executing {targetTrainSteps} Backprop Steps...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Run +{targetTrainSteps} Optimization Steps</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* STEP 4: Generate */}
          {currentStep === 4 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-bold text-zinc-900">Step 4: Sample Autoregressively</h3>
                <p className="text-zinc-500">Provide a seed prompt and watch your trained model predict character-by-character.</p>
              </div>

              <div className="space-y-2">
                <label className="font-semibold text-zinc-800">Seed Prompt:</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={promptText}
                    onChange={(e) => setPromptText(e.target.value)}
                    className="flex-1 p-2 rounded-lg border border-zinc-300 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-zinc-900"
                  />
                  <button
                    onClick={handleTestGenerate}
                    className="px-4 py-2 bg-zinc-900 hover:bg-zinc-800 text-white font-medium rounded-lg transition-colors flex items-center gap-1.5"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Generate</span>
                  </button>
                </div>
              </div>

              <div className="p-4 bg-zinc-950 text-zinc-100 rounded-xl font-mono text-xs border border-zinc-800 min-h-[90px] whitespace-pre-wrap leading-relaxed">
                {generatedSample ? (
                  <div>
                    <span className="text-emerald-400 font-bold">{promptText}</span>
                    <span className="text-zinc-200">{generatedSample.slice(promptText.length)}</span>
                  </div>
                ) : (
                  <span className="text-zinc-500 italic">Click "Generate" above to test your trained model.</span>
                )}
              </div>
            </div>
          )}

        </div>

        {/* Footer Navigation */}
        <div className="p-4 border-t border-zinc-100 bg-zinc-50 flex items-center justify-between text-xs">
          <button
            onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
            disabled={currentStep === 1}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-200 text-zinc-700 hover:bg-white disabled:opacity-40 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back</span>
          </button>

          {currentStep < 4 ? (
            <button
              onClick={() => setCurrentStep(currentStep + 1)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-white font-medium transition-colors"
            >
              <span>Next Step</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          ) : (
            <button
              onClick={onClose}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-bold transition-colors shadow-xs"
            >
              <Check className="w-4 h-4" />
              <span>Finish & Open in Visualizer</span>
            </button>
          )}
        </div>

      </div>
    </div>
  );
};
