import React, { useState, useMemo, useRef } from 'react';
import { Play, Pause, FastForward, RotateCcw, Sparkles, Sliders, ChevronRight, Eye, RefreshCw, Upload, FileCode, Check, AlertCircle } from 'lucide-react';
import { ModelConfig, TrainingMetrics, AttentionHeadData } from '../types';
import { generateWeightExportPythonCode } from '../engine/weightBridge';

interface PretrainingWorkbenchProps {
  config: ModelConfig;
  setConfig: React.Dispatch<React.SetStateAction<ModelConfig>>;
  metricsHistory: TrainingMetrics[];
  isTraining: boolean;
  onToggleTraining: () => void;
  onStep: (stepsCount: number) => void;
  onReset: () => void;
  onGenerate: (prompt: string, temp: number, topK: number, maxTokens: number) => { text: string; tokens: number[]; attention: AttentionHeadData[] };
  datasetTitle: string;
  initialTheoreticalLoss: number;
  onImportWeights?: (data: any) => boolean;
  weightSource?: 'browser' | 'pytorch';
}

export const PretrainingWorkbench: React.FC<PretrainingWorkbenchProps> = ({
  config,
  setConfig,
  metricsHistory,
  isTraining,
  onToggleTraining,
  onStep,
  onReset,
  onGenerate,
  datasetTitle,
  initialTheoreticalLoss,
  onImportWeights,
  weightSource = 'browser'
}) => {
  const [prompt, setPrompt] = useState('INTERVIEWER:');
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(8);
  const [maxTokens, setMaxTokens] = useState(70);
  const [generatedResult, setGeneratedResult] = useState<string>('');
  const [activeHead, setActiveHead] = useState<number>(0);
  const [attentionData, setAttentionData] = useState<AttentionHeadData[]>([]);
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(5);
  const [showHyperparams, setShowHyperparams] = useState<boolean>(false);

  const [showImportModal, setShowImportModal] = useState<boolean>(false);
  const [importStatus, setImportStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target?.result as string);
        if (onImportWeights && onImportWeights(json)) {
          setImportStatus('success');
          setTimeout(() => {
            setShowImportModal(false);
            setImportStatus(null);
          }, 1500);
        } else {
          setImportStatus('error');
        }
      } catch (err) {
        setImportStatus('error');
      }
    };
    reader.readAsText(file);
  };

  const pythonExportCode = generateWeightExportPythonCode();
  const latestMetric = metricsHistory[metricsHistory.length - 1];
  const currentLoss = latestMetric ? latestMetric.loss : initialTheoreticalLoss;
  const currentStep = latestMetric ? latestMetric.step : 0;
  const initialLoss = metricsHistory[0]?.loss || initialTheoreticalLoss;
  const lossDropPct = initialLoss > 0 ? Math.max(0, ((initialLoss - currentLoss) / initialLoss) * 100) : 0;

  // Compute SVG coordinates for the loss curve
  const chartSvgPath = useMemo(() => {
    if (metricsHistory.length < 2) return '';
    const width = 600;
    const height = 180;
    const padding = 20;

    const minLoss = Math.min(...metricsHistory.map(m => m.loss), 1.0);
    const maxLoss = Math.max(...metricsHistory.map(m => m.loss), initialTheoreticalLoss);
    const lossRange = Math.max(0.5, maxLoss - minLoss);

    const points = metricsHistory.map((m, idx) => {
      const x = padding + (idx / (metricsHistory.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((m.loss - minLoss) / lossRange) * (height - 2 * padding);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    return `M ${points.join(' L ')}`;
  }, [metricsHistory, initialTheoreticalLoss]);

  const handleRunGenerate = () => {
    const res = onGenerate(prompt, temperature, topK, maxTokens);
    setGeneratedResult(res.text);
    setAttentionData(res.attention || []);
  };

  return (
    <div className="space-y-6">
      
      {/* Top Banner: Context and Experiment Overview */}
      <div className="bg-zinc-900 text-white rounded-2xl p-6 border border-zinc-800 relative overflow-hidden shadow-sm">
        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-zinc-800/90 text-zinc-300 text-xs font-mono">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>Experiment Phase: {currentStep === 0 ? 'Fresh Initialization' : isTraining ? 'Pretraining in Progress' : 'Checkpoint Paused'}</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-zinc-100">
              Smallest "From Scratch" LLM Pretrainer
            </h1>
            <p className="text-sm text-zinc-400 leading-relaxed">
              Witness next-token causal language model pretraining right in your browser. Watch the cross-entropy loss descend from random chance (~{initialTheoreticalLoss.toFixed(2)}) as the transformer discovers character sequences, bigrams, and syntax on <span className="text-zinc-200 font-medium">"{datasetTitle}"</span>.
            </p>
          </div>

          <div className="flex flex-wrap lg:flex-nowrap items-center gap-3">
            <button
              id="workbench-main-train-btn"
              onClick={onToggleTraining}
              className={`flex items-center gap-2 px-5 py-3 rounded-xl font-medium text-sm transition-all shadow-md ${
                isTraining
                  ? 'bg-amber-500 hover:bg-amber-600 text-zinc-950 font-semibold'
                  : 'bg-emerald-500 hover:bg-emerald-400 text-zinc-950 font-semibold'
              }`}
            >
              {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
              <span>{isTraining ? 'Pause Training' : currentStep > 0 ? 'Continue Training' : 'Start Pretraining'}</span>
            </button>

            <button
              id="workbench-step-10-btn"
              onClick={() => onStep(10)}
              disabled={isTraining}
              className="flex items-center gap-1.5 px-3.5 py-3 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-sm font-medium border border-zinc-700 disabled:opacity-50 transition-colors"
              title="Execute 10 backprop steps manually"
            >
              <FastForward className="w-4 h-4 text-emerald-400" />
              <span>+10 Steps</span>
            </button>

            <button
              id="workbench-import-weights-btn"
              onClick={() => setShowImportModal(true)}
              className="flex items-center gap-1.5 px-3.5 py-3 rounded-xl bg-indigo-950/80 hover:bg-indigo-900 text-indigo-200 text-sm font-medium border border-indigo-700 transition-colors"
              title="Import trained PyTorch weights into visualizer"
            >
              <Upload className="w-4 h-4 text-indigo-400" />
              <span className="hidden sm:inline">Import .pt Weights</span>
            </button>

            <button
              id="workbench-reset-weights-btn"
              onClick={onReset}
              className="p-3 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 border border-zinc-700 transition-colors"
              title="Re-initialize weights with random normal noise"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Subtle background gradient */}
        <div className="absolute -right-20 -bottom-20 w-80 h-80 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none" />
      </div>

      {/* Modal: Import PyTorch Checkpoint */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-xl w-full border border-zinc-200 p-6 shadow-2xl space-y-4 text-xs">
            <div className="flex items-center justify-between pb-2 border-b border-zinc-100">
              <div className="flex items-center gap-2">
                <FileCode className="w-5 h-5 text-indigo-600" />
                <h3 className="text-base font-bold text-zinc-900">Import Trained PyTorch Weights</h3>
              </div>
              <button
                onClick={() => setShowImportModal(false)}
                className="text-zinc-400 hover:text-zinc-700 font-bold p-1"
              >
                ✕
              </button>
            </div>

            <p className="text-zinc-600 leading-relaxed">
              Step 2 in the roadmap: <em>"Train the .pt file in Colab/PyTorch as the source of truth; keep the browser engine as a visualizer."</em>
            </p>

            <div className="p-4 bg-zinc-50 rounded-xl border-2 border-dashed border-zinc-300 text-center space-y-2">
              <Upload className="w-8 h-8 text-zinc-400 mx-auto" />
              <div className="font-semibold text-zinc-800">
                Select tiny_gpt_weights.json exported from PyTorch
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg shadow-xs transition-colors"
              >
                Browse File
              </button>
            </div>

            {importStatus === 'success' && (
              <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-800 flex items-center gap-2">
                <Check className="w-4 h-4 text-emerald-600" />
                <span>PyTorch checkpoint successfully loaded! Visualizer updated.</span>
              </div>
            )}

            {importStatus === 'error' && (
              <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-rose-800 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600" />
                <span>Invalid weights format. Ensure you used export_model_for_browser_visualizer()</span>
              </div>
            )}

            <div className="space-y-1.5">
              <div className="text-[11px] font-semibold text-zinc-700">How to export from Python:</div>
              <div className="p-3 bg-zinc-950 text-zinc-300 font-mono text-[11px] rounded-lg overflow-x-auto max-h-36">
                <pre>{pythonExportCode}</pre>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Telemetry Dashboard Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        
        <div className="bg-white p-4 rounded-xl border border-zinc-200/80 shadow-xs">
          <div className="text-xs font-medium text-zinc-500 mb-1">Pretraining Step</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">{currentStep.toLocaleString()}</div>
          <div className="text-[11px] text-zinc-600 mt-1 flex items-center gap-1">
            <span>Speed: {speedMultiplier} steps/tick</span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200/80 shadow-xs">
          <div className="text-xs font-medium text-zinc-500 mb-1">Cross-Entropy Loss</div>
          <div className="flex items-baseline gap-2">
            <div className={`text-2xl font-bold font-mono ${currentLoss < 2.5 ? 'text-emerald-600' : 'text-zinc-900'}`}>
              {currentLoss.toFixed(4)}
            </div>
            {lossDropPct > 0 && (
              <span className="text-xs text-emerald-600 font-semibold">
                -{lossDropPct.toFixed(0)}%
              </span>
            )}
          </div>
          <div className="text-[11px] text-zinc-600 mt-1">
            Random Baseline: ~{initialTheoreticalLoss.toFixed(2)}
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200/80 shadow-xs">
          <div className="text-xs font-medium text-zinc-500 mb-1">Perplexity (exp(Loss))</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">
            {Math.min(9999, Math.exp(currentLoss)).toFixed(1)}
          </div>
          <div className="text-[11px] text-zinc-600 mt-1">
            Lower is better (vocab uncertainty)
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200/80 shadow-xs">
          <div className="text-xs font-medium text-zinc-500 mb-1">Tokens Processed</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">
            {latestMetric ? latestMetric.tokensProcessed.toLocaleString() : '0'}
          </div>
          <div className="text-[11px] text-zinc-600 mt-1">
            Batch: {config.batchSize} × {config.blockSize} context
          </div>
        </div>

      </div>

      {/* Main Graph & Control Center */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100">
          <div>
            <h2 className="text-base font-semibold text-zinc-900">Pretraining Loss Curve</h2>
            <p className="text-xs text-zinc-500">Real-time negative log likelihood during stochastic gradient descent</p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-xs text-zinc-600">
              <span>Steps / frame:</span>
              <input
                type="range"
                min="1"
                max="20"
                value={speedMultiplier}
                onChange={(e) => setSpeedMultiplier(Number(e.target.value))}
                className="w-20 accent-zinc-900"
              />
              <span className="font-mono text-zinc-800 w-6 font-semibold">{speedMultiplier}x</span>
            </div>

            <button
              id="workbench-toggle-hyperparams-btn"
              onClick={() => setShowHyperparams(!showHyperparams)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-zinc-100 hover:bg-zinc-200 text-zinc-700 transition-colors"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>Hyperparams</span>
            </button>
          </div>
        </div>

        {/* Expandable Hyperparameters Drawer */}
        {showHyperparams && (
          <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div>
              <label className="block text-zinc-600 font-medium mb-1">Learning Rate</label>
              <select
                value={config.lr}
                onChange={(e) => setConfig({ ...config, lr: Number(e.target.value) })}
                className="w-full bg-white border border-zinc-300 rounded-md px-2.5 py-1.5 text-zinc-800"
              >
                <option value={0.005}>0.005 (Fast)</option>
                <option value={0.002}>0.002 (Balanced)</option>
                <option value={0.0008}>0.0008 (Gentle)</option>
              </select>
            </div>

            <div>
              <label className="block text-zinc-600 font-medium mb-1">Context Length (Block)</label>
              <select
                value={config.blockSize}
                onChange={(e) => setConfig({ ...config, blockSize: Number(e.target.value) })}
                className="w-full bg-white border border-zinc-300 rounded-md px-2.5 py-1.5 text-zinc-800"
              >
                <option value={16}>16 tokens</option>
                <option value={32}>32 tokens (Default)</option>
                <option value={64}>64 tokens</option>
              </select>
            </div>

            <div>
              <label className="block text-zinc-600 font-medium mb-1">Embedding Dim (d_model)</label>
              <select
                value={config.nEmbd}
                onChange={(e) => setConfig({ ...config, nEmbd: Number(e.target.value) })}
                className="w-full bg-white border border-zinc-300 rounded-md px-2.5 py-1.5 text-zinc-800"
              >
                <option value={24}>24 dim</option>
                <option value={32}>32 dim (Default)</option>
                <option value={48}>48 dim</option>
              </select>
            </div>

            <div>
              <label className="block text-zinc-600 font-medium mb-1">Attention Heads</label>
              <select
                value={config.nHead}
                onChange={(e) => setConfig({ ...config, nHead: Number(e.target.value) })}
                className="w-full bg-white border border-zinc-300 rounded-md px-2.5 py-1.5 text-zinc-800"
              >
                <option value={2}>2 Heads</option>
                <option value={4}>4 Heads</option>
              </select>
            </div>
          </div>
        )}

        {/* SVG Loss Curve Display */}
        <div className="relative w-full h-48 bg-zinc-950 rounded-xl p-3 overflow-hidden font-mono text-xs flex flex-col justify-between">
          <div className="flex justify-between items-center text-zinc-500 text-[11px] z-10">
            <span>Theoretical Random (~{initialTheoreticalLoss.toFixed(2)})</span>
            <span className="text-emerald-400">Current: {currentLoss.toFixed(3)}</span>
          </div>

          {metricsHistory.length < 2 ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-500 gap-2">
              <RefreshCw className="w-5 h-5 animate-spin text-zinc-600" />
              <span>Click "Start Pretraining" to record steps & plot real loss</span>
            </div>
          ) : (
            <svg viewBox="0 0 600 180" className="w-full h-full preserve-3d" preserveAspectRatio="none">
              {/* Baseline reference line */}
              <line x1="20" y1="25" x2="580" y2="25" stroke="#3f3f46" strokeDasharray="4 4" strokeWidth="1" />
              
              {/* Target threshold line at ~2.0 */}
              <line x1="20" y1="120" x2="580" y2="120" stroke="#10b981" strokeOpacity="0.3" strokeDasharray="3 3" strokeWidth="1" />

              {/* Loss Area and Line */}
              <path
                d={chartSvgPath}
                fill="none"
                stroke="#10b981"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}

          <div className="flex justify-between items-center text-zinc-500 text-[10px] z-10">
            <span>Step 0</span>
            <span>Target: &lt; 2.0 (syntax fluency)</span>
            <span>Step {currentStep}</span>
          </div>
        </div>
      </div>

      {/* Interactive Text Generation & Sampling Station */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-zinc-100">
          <div>
            <h2 className="text-base font-semibold text-zinc-900 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-500" />
              <span>Autoregressive Text Generation</span>
            </h2>
            <p className="text-xs text-zinc-500">Sample from the trained model checkpoint token-by-token</p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">Starters:</span>
            {['INTERVIEWER:', 'CANDIDATE:', 'def ', 'First'].map((starter) => (
              <button
                key={starter}
                onClick={() => setPrompt(starter)}
                className="px-2 py-0.5 rounded text-[11px] bg-zinc-100 hover:bg-zinc-200 text-zinc-700 transition-colors font-mono"
              >
                {starter}
              </button>
            ))}
          </div>
        </div>

        {/* Generation Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2 space-y-1">
            <label className="text-xs font-medium text-zinc-700">Seed Prompt</label>
            <div className="flex gap-2">
              <input
                id="generation-prompt-input"
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Type prompt here..."
                className="flex-1 bg-zinc-50 border border-zinc-300 rounded-lg px-3 py-2 text-sm font-mono text-zinc-900 focus:outline-none focus:ring-2 focus:ring-zinc-900"
              />
              <button
                id="generate-text-btn"
                onClick={handleRunGenerate}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-white text-sm font-medium transition-colors shadow-xs"
              >
                <Sparkles className="w-4 h-4 text-emerald-300" />
                <span>Generate</span>
              </button>
            </div>
          </div>

          {/* Sampling Sliders */}
          <div className="grid grid-cols-2 gap-3 p-3 bg-zinc-50 rounded-xl border border-zinc-200 text-xs">
            <div>
              <div className="flex justify-between text-zinc-600 mb-1">
                <span>Temp:</span>
                <span className="font-mono font-semibold">{temperature.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full accent-zinc-900"
                title="Higher = more diverse, Lower = deterministic"
              />
            </div>

            <div>
              <div className="flex justify-between text-zinc-600 mb-1">
                <span>Top-K:</span>
                <span className="font-mono font-semibold">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="25"
                step="1"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full accent-zinc-900"
                title="Limits sampling to top K probable tokens"
              />
            </div>
          </div>
        </div>

        {/* Output Box */}
        <div className="p-4 rounded-xl bg-zinc-950 text-zinc-100 font-mono text-sm leading-relaxed border border-zinc-800 min-h-[110px] relative">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-bold mb-2 flex items-center justify-between">
            <span>Model Output (Autoregressive Logits Sampling)</span>
            <span className="text-zinc-600">{currentStep === 0 ? 'Random Weights (Gibberish expected)' : `Trained at Step ${currentStep}`}</span>
          </div>

          {generatedResult ? (
            <div className="whitespace-pre-wrap">
              <span className="text-emerald-400 font-bold">{prompt}</span>
              <span className="text-zinc-200">{generatedResult.slice(prompt.length)}</span>
            </div>
          ) : (
            <div className="text-zinc-600 italic">
              Click "Generate" to autoregressively predict subsequent characters using the transformer's current weights.
            </div>
          )}
        </div>

        {/* Multi-Head Attention Map Visualizer */}
        {attentionData.length > 0 && (
          <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4 text-indigo-600" />
                <span className="text-xs font-semibold text-zinc-900">Transformer Attention Map Inspector</span>
              </div>
              <div className="flex items-center gap-1">
                {attentionData.map((head, idx) => (
                  <button
                    key={idx}
                    onClick={() => setActiveHead(idx)}
                    className={`px-2.5 py-1 text-xs rounded font-mono ${
                      activeHead === idx
                        ? 'bg-zinc-900 text-white font-semibold'
                        : 'bg-zinc-200 text-zinc-700 hover:bg-zinc-300'
                    }`}
                  >
                    Head {idx}
                  </button>
                ))}
              </div>
            </div>

            <p className="text-xs text-zinc-600">
              Hover over generated tokens to see causal attention intensity (softmax(Q·Kᵀ / √d)). Darker highlights denote higher attention weight.
            </p>

            <div className="flex flex-wrap gap-1 p-3 bg-white rounded-lg border border-zinc-200 font-mono text-xs">
              {generatedResult.split('').slice(0, 40).map((char, idx) => {
                const isHovered = hoveredTokenIdx === idx;
                const matrix = attentionData[activeHead]?.matrix;
                let attentionFromHovered = 0;
                if (hoveredTokenIdx !== null && matrix && matrix[hoveredTokenIdx]) {
                  attentionFromHovered = matrix[hoveredTokenIdx][idx] || 0;
                }

                const bgAlpha = hoveredTokenIdx !== null ? Math.min(0.9, attentionFromHovered * 2) : 0;

                return (
                  <span
                    key={idx}
                    onMouseEnter={() => setHoveredTokenIdx(idx)}
                    onMouseLeave={() => setHoveredTokenIdx(null)}
                    style={{
                      backgroundColor: isHovered
                        ? '#6366f1'
                        : attentionFromHovered > 0.05
                        ? `rgba(99, 102, 241, ${bgAlpha})`
                        : undefined,
                      color: isHovered ? '#ffffff' : undefined
                    }}
                    className={`px-1.5 py-0.5 rounded cursor-pointer transition-colors border border-transparent ${
                      isHovered ? 'shadow-xs font-bold' : 'hover:border-zinc-300'
                    }`}
                  >
                    {char === ' ' ? '␣' : char === '\n' ? '↵' : char}
                  </span>
                );
              })}
            </div>
          </div>
        )}

      </div>

    </div>
  );
};
