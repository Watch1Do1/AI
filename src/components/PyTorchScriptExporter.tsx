import React, { useState } from 'react';
import { Terminal, Copy, Check, Download, ExternalLink, Play, HardDrive, Cpu, CheckCircle } from 'lucide-react';
import { ModelConfig } from '../types';
import { generatePyTorchScript, generateColabNotebookJSON } from '../engine/pytorchCode';

interface PyTorchScriptExporterProps {
  config: ModelConfig;
}

export const PyTorchScriptExporter: React.FC<PyTorchScriptExporterProps> = ({ config }) => {
  const [copied, setCopied] = useState(false);
  const [copiedBash, setCopiedBash] = useState(false);

  const pythonScript = generatePyTorchScript(config);

  const handleCopy = () => {
    navigator.clipboard.writeText(pythonScript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadPy = () => {
    const blob = new Blob([pythonScript], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'train_tiny_gpt.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadColab = () => {
    const colabJSON = generateColabNotebookJSON(config);
    const blob = new Blob([colabJSON], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'TinyGPT_Pretraining_Colab.ipynb';
    a.click();
    URL.revokeObjectURL(url);
  };

  const bashInstructions = `# 1. Create a clean project folder & virtual environment
mkdir tiny-gpt-lab && cd tiny-gpt-lab
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate

# 2. Install standard PyTorch (CPU or GPU)
pip install torch

# 3. Run the complete training script
python train_tiny_gpt.py`;

  const handleCopyBash = () => {
    navigator.clipboard.writeText(bashInstructions);
    setCopiedBash(true);
    setTimeout(() => setCopiedBash(false), 2000);
  };

  return (
    <div className="space-y-6">
      
      {/* Overview Banner */}
      <div className="bg-emerald-950 text-white rounded-2xl p-6 border border-emerald-900 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-emerald-300 text-xs font-mono font-semibold">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span>Smallest Real Experiment Blueprint</span>
        </div>
        <h2 className="text-xl font-bold text-emerald-50">
          Standalone 1-File PyTorch Pretraining Script
        </h2>
        <p className="text-sm text-emerald-200/90 leading-relaxed">
          Zero complex framework dependencies. This standalone ~150-line script is a fully functional causal Decoder-Only Transformer (GPT architecture). It runs on CPU, Apple Silicon MPS, or NVIDIA CUDA, creates a dataset file, trains until loss drops, saves <code className="bg-emerald-900/80 px-1.5 py-0.5 rounded text-white font-mono text-xs">tiny_gpt_weights.pt</code>, and generates autoregressive text.
        </p>

        <div className="flex flex-wrap gap-3 pt-2">
          <button
            onClick={handleDownloadPy}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-zinc-950 text-xs font-semibold transition-colors shadow-sm"
          >
            <Download className="w-4 h-4" />
            <span>Download train_tiny_gpt.py</span>
          </button>

          <button
            onClick={handleDownloadColab}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-emerald-900/80 hover:bg-emerald-800 text-emerald-100 text-xs font-semibold border border-emerald-700 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            <span>Download Google Colab Notebook (.ipynb)</span>
          </button>

          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-emerald-900/80 hover:bg-emerald-800 text-emerald-100 text-xs font-semibold border border-emerald-700 transition-colors"
          >
            {copied ? <Check className="w-4 h-4 text-emerald-300" /> : <Copy className="w-4 h-4" />}
            <span>{copied ? 'Copied Script!' : 'Copy Python Code'}</span>
          </button>
        </div>
      </div>

      {/* Terminal Quickstart Guide */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-zinc-900 flex items-center gap-2">
            <Play className="w-4 h-4 text-emerald-600" />
            <span>How to Run in Terminal (3 Minutes)</span>
          </h3>
          <button
            onClick={handleCopyBash}
            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-800 transition-colors font-mono"
          >
            {copiedBash ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copiedBash ? 'Copied' : 'Copy Commands'}</span>
          </button>
        </div>

        <div className="p-3 bg-zinc-950 rounded-xl text-zinc-200 font-mono text-xs overflow-x-auto">
          <pre>{bashInstructions}</pre>
        </div>
      </div>

      {/* What to Expect During Pretraining */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs">
        <div className="p-3.5 bg-white rounded-xl border border-zinc-200 shadow-xs space-y-1">
          <div className="text-zinc-500 font-medium">Step 0 (Random)</div>
          <div className="text-base font-mono font-bold text-zinc-900">Loss ~4.15</div>
          <p className="text-zinc-500 text-[11px]">Uncertainty over character vocab (-ln(1/64)). Generates random character noise.</p>
        </div>

        <div className="p-3.5 bg-white rounded-xl border border-zinc-200 shadow-xs space-y-1">
          <div className="text-zinc-500 font-medium">Step 100 - 200</div>
          <div className="text-base font-mono font-bold text-zinc-900">Loss ~2.80</div>
          <p className="text-zinc-500 text-[11px]">Model discovers space patterns, vowel frequencies, and common bigrams (th, in, er).</p>
        </div>

        <div className="p-3.5 bg-white rounded-xl border border-zinc-200 shadow-xs space-y-1">
          <div className="text-zinc-500 font-medium">Step 500</div>
          <div className="text-base font-mono font-bold text-emerald-600">Loss ~2.10</div>
          <p className="text-zinc-500 text-[11px]">Real words emerge with consistent spacing and simple grammatical rhythms.</p>
        </div>

        <div className="p-3.5 bg-white rounded-xl border border-zinc-200 shadow-xs space-y-1">
          <div className="text-zinc-500 font-medium">Step 1000 (Checkpoint)</div>
          <div className="text-base font-mono font-bold text-emerald-600">Loss ~1.65</div>
          <p className="text-zinc-500 text-[11px]">Structured syntax (e.g. Q&A dialogue turns, code blocks, or character names).</p>
        </div>
      </div>

      {/* Full Script Code Viewer */}
      <div className="bg-zinc-950 rounded-2xl border border-zinc-800 overflow-hidden shadow-sm">
        <div className="flex items-center justify-between px-4 py-3 bg-zinc-900 border-b border-zinc-800 text-xs font-mono">
          <div className="flex items-center gap-2 text-zinc-300">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <span className="font-semibold">train_tiny_gpt.py</span>
            <span className="text-zinc-500 text-[11px]">({pythonScript.split('\n').length} lines)</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
            <button
              onClick={handleDownloadPy}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download</span>
            </button>
          </div>
        </div>

        <div className="p-4 max-h-[500px] overflow-y-auto text-zinc-300 font-mono text-xs leading-relaxed">
          <pre>{pythonScript}</pre>
        </div>
      </div>

    </div>
  );
};
