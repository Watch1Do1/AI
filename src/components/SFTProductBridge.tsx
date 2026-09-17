import React, { useState } from 'react';
import { Sparkles, MessageSquare, Download, Copy, Check, ExternalLink, ArrowRight, ShieldCheck, Terminal, Bot } from 'lucide-react';
import { generateSFTFineTuningScript, generateSFTColabJSON } from '../engine/sftGuide';

export const SFTProductBridge: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('unsloth/Llama-3.2-1B-Instruct');
  const [copiedScript, setCopiedScript] = useState(false);
  const [userPrompt, setUserPrompt] = useState('How would you evaluate a candidate who struggles with distributed consensus?');

  const sftScript = generateSFTFineTuningScript(selectedModel);

  const handleCopy = () => {
    navigator.clipboard.writeText(sftScript);
    setCopiedScript(true);
    setTimeout(() => setCopiedScript(false), 2000);
  };

  const handleDownloadPy = () => {
    const blob = new Blob([sftScript], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'fine_tune_interview_model.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadColab = () => {
    const jsonStr = generateSFTColabJSON();
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Fine_Tune_Interview_Model_Colab.ipynb';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      
      {/* Overview Banner */}
      <div className="bg-amber-950 text-white rounded-2xl p-6 border border-amber-900 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-amber-300 text-xs font-mono font-semibold">
          <MessageSquare className="w-4 h-4 text-amber-400" />
          <span>Step 3 in the Roadmap</span>
        </div>
        <h2 className="text-xl font-bold text-amber-50">
          "Fine-tune an existing 1B–8B open model on your interview format if you want a system that talks. That is product. It is not more pretraining."
        </h2>
        <p className="text-sm text-amber-200/90 leading-relaxed">
          Pretraining from scratch teaches raw statistical grammar. But to create a <strong>functional conversational product</strong> that speaks fluent English, understands nuances, and strictly adheres to your interview format, the industry solution is <strong>Supervised Fine-Tuning (SFT) using LoRA</strong> on an established open weight model (like Llama-3.2-1B, Qwen-2.5-1.5B, or Mistral-7B).
        </p>

        <div className="flex flex-wrap gap-3 pt-2">
          <button
            onClick={handleDownloadPy}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-zinc-950 text-xs font-semibold transition-colors shadow-sm"
          >
            <Download className="w-4 h-4" />
            <span>Download fine_tune_interview_model.py</span>
          </button>

          <button
            onClick={handleDownloadColab}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-amber-900/80 hover:bg-amber-800 text-amber-100 text-xs font-semibold border border-amber-700 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            <span>Download Colab SFT Notebook (.ipynb)</span>
          </button>

          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-amber-900/80 hover:bg-amber-800 text-amber-100 text-xs font-semibold border border-amber-700 transition-colors"
          >
            {copiedScript ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            <span>{copiedScript ? 'Copied Script!' : 'Copy LoRA Code'}</span>
          </button>
        </div>
      </div>

      {/* The 3 Paradigms Comparison Table */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        
        <div className="p-4 rounded-xl bg-white border border-zinc-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between font-bold text-zinc-900">
            <span>1. Toy Pretraining</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-100 text-zinc-600">Learning</span>
          </div>
          <p className="text-zinc-600 leading-relaxed">
            Pretraining a 100K–10M model on raw text. Teaches token prediction mechanics and loss curves. Not useful as a conversational assistant.
          </p>
          <div className="p-2.5 rounded-lg bg-zinc-100 font-mono text-[11px] text-zinc-600">
            Cost: Free (browser / CPU)<br />Output: Statistical mimicry
          </div>
        </div>

        <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between font-bold text-amber-950">
            <span>2. SFT on 1B–8B Model</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-bold">The Product</span>
          </div>
          <p className="text-amber-900 leading-relaxed">
            Take Llama-3.2-1B or Qwen-2.5-1.5B (already knows the English language) and adapt it to your interview style via LoRA in 15 minutes.
          </p>
          <div className="p-2.5 rounded-lg bg-white border border-amber-200 font-mono text-[11px] text-amber-950">
            Cost: 15 min on free T4 GPU<br />Output: Fluent interview partner
          </div>
        </div>

        <div className="p-4 rounded-xl bg-indigo-50/70 border border-indigo-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between font-bold text-indigo-950">
            <span>3. Attached Memory (RAG)</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-100 text-indigo-800 font-bold">Factual State</span>
          </div>
          <p className="text-indigo-900 leading-relaxed">
            Connect an external episodic store (KV database) so the model remembers candidate notes without hallucinating or forgetting.
          </p>
          <div className="p-2.5 rounded-lg bg-white border border-indigo-200 font-mono text-[11px] text-indigo-950">
            Cost: SQLite / Vector store<br />Output: Exact historical recall
          </div>
        </div>

      </div>

      {/* Model Selector & Python Script Viewer */}
      <div className="bg-zinc-950 rounded-2xl border border-zinc-800 overflow-hidden shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between px-4 py-3 bg-zinc-900 border-b border-zinc-800 text-xs font-mono gap-2">
          <div className="flex items-center gap-3 text-zinc-300">
            <Bot className="w-4 h-4 text-amber-400" />
            <span className="font-semibold">fine_tune_interview_model.py</span>
          </div>

          <div className="flex items-center gap-2">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-zinc-200 text-xs focus:outline-none"
            >
              <option value="unsloth/Llama-3.2-1B-Instruct">Llama-3.2-1B (Fast, 4GB VRAM)</option>
              <option value="unsloth/Qwen2.5-1.5B-Instruct">Qwen-2.5-1.5B (Excellent dialogue)</option>
              <option value="unsloth/Meta-Llama-3.1-8B-Instruct">Llama-3.1-8B (Production Quality)</option>
            </select>
          </div>
        </div>

        <div className="p-4 max-h-[460px] overflow-y-auto text-zinc-300 font-mono text-xs leading-relaxed">
          <pre>{sftScript}</pre>
        </div>
      </div>

    </div>
  );
};
