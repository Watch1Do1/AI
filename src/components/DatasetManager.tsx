import React, { useState } from 'react';
import { BookOpen, Upload, FileText, Sparkles, Check, Hash, RefreshCw, BarChart2 } from 'lucide-react';
import { CorpusPreset, TokenizerData } from '../types';
import { CORPUS_PRESETS } from '../engine/datasets';

interface DatasetManagerProps {
  currentCorpus: string;
  onSelectCorpus: (text: string, title: string) => void;
  selectedTitle: string;
  tokenizerData: TokenizerData;
  applyConfirmation?: { activeTitle: string; vocabSize: number; paramCount: number } | null;
  paramCount?: number;
}

export const DatasetManager: React.FC<DatasetManagerProps> = ({
  currentCorpus,
  onSelectCorpus,
  selectedTitle,
  tokenizerData,
  applyConfirmation,
  paramCount
}) => {
  const [customText, setCustomText] = useState('');
  const [customTitle, setCustomTitle] = useState('My Custom Dataset');
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [synthTopic, setSynthTopic] = useState('Software Engineering Interview Transcript');
  const [synthMessage, setSynthMessage] = useState('');

  const theoreticalLoss = Math.log(Math.max(2, tokenizerData.vocabSize));

  const handleApplyCustomText = () => {
    if (!customText.trim()) return;
    onSelectCorpus(customText, customTitle);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      if (content) {
        setCustomText(content);
        setCustomTitle(file.name.replace(/\.[^/.]+$/, ''));
        onSelectCorpus(content, file.name);
      }
    };
    reader.readAsText(file);
  };

  const handleSynthesize = async () => {
    setIsSynthesizing(true);
    setSynthMessage('');
    try {
      const res = await fetch('/api/synthesize-corpus', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: synthTopic, length: 'medium' })
      });
      const data = await res.json();
      if (data.corpus) {
        setCustomText(data.corpus);
        setCustomTitle(`AI Synthesized: ${synthTopic}`);
        onSelectCorpus(data.corpus, `AI Synthesized: ${synthTopic}`);
        setSynthMessage(`Synthesized ${data.corpus.length} characters via ${data.source === 'gemini' ? 'Gemini 3.8 Flash' : 'Domain Template'}!`);
      }
    } catch (err) {
      console.error(err);
      setSynthMessage('Failed to synthesize text. You can still paste or select built-in datasets.');
    } finally {
      setIsSynthesizing(false);
    }
  };

  // Sample tokenization preview
  const sampleSnippet = currentCorpus.slice(0, 80);

  return (
    <div className="space-y-6">
      
      {/* Visible Confirmation after Apply to Model */}
      {applyConfirmation && (
        <div id="dataset-applied-confirmation-banner" className="p-4 bg-emerald-50 border border-emerald-300 rounded-2xl text-emerald-950 space-y-2 shadow-xs">
          <div className="flex items-center gap-2 font-bold text-emerald-900 text-xs">
            <Check className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>Applied to Model Successfully</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 font-mono text-xs">
            <div className="bg-white/90 p-2.5 rounded-lg border border-emerald-200">
              <div className="text-[10px] uppercase tracking-wider text-emerald-800 font-sans font-medium">Active Title</div>
              <div className="font-bold text-zinc-900 truncate" title={applyConfirmation.activeTitle}>{applyConfirmation.activeTitle}</div>
            </div>
            <div className="bg-white/90 p-2.5 rounded-lg border border-emerald-200">
              <div className="text-[10px] uppercase tracking-wider text-emerald-800 font-sans font-medium">New Vocab Size V</div>
              <div className="font-bold text-zinc-900">{applyConfirmation.vocabSize}</div>
            </div>
            <div className="bg-white/90 p-2.5 rounded-lg border border-emerald-200">
              <div className="text-[10px] uppercase tracking-wider text-emerald-800 font-sans font-medium">New Param Count</div>
              <div className="font-bold text-zinc-900">{applyConfirmation.paramCount.toLocaleString()}</div>
            </div>
          </div>
        </div>
      )}

      {/* Header Info */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-emerald-600" />
            <h2 className="text-base font-semibold text-zinc-900">Pretraining Corpus & Tokenizer</h2>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-zinc-100 text-zinc-700">
            Active: <strong className="text-zinc-900">{selectedTitle}</strong>
          </span>
        </div>
        <p className="text-xs text-zinc-500 leading-relaxed">
          The smallest real pretraining experiment requires a text file of repetitive structures or domain grammar. The character tokenizer maps every unique character to a numeric token ID in [0, V-1].
        </p>
      </div>

      {/* Vocabulary Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">Corpus Size</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">{currentCorpus.length.toLocaleString()}</div>
          <div className="text-[11px] text-zinc-600 mt-1">Total characters</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">Vocabulary Size (V)</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">{tokenizerData.vocabSize}</div>
          <div className="text-[11px] text-zinc-600 mt-1">Unique character tokens</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">Theoretical Initial Loss</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">{theoreticalLoss.toFixed(3)}</div>
          <div className="text-[11px] text-zinc-600 mt-1">-ln(1/V) random baseline</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-zinc-200 shadow-xs">
          <div className="text-xs text-zinc-500 mb-1">Tokenizer Type</div>
          <div className="text-2xl font-bold font-mono text-zinc-900">Char-Level</div>
          <div className="text-[11px] text-zinc-600 mt-1">Deterministic bijection</div>
        </div>
      </div>

      {/* Preset Library Selection */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-4">
        <h3 className="text-sm font-semibold text-zinc-900">Choose a Pre-Curated Training Corpus</h3>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {CORPUS_PRESETS.map((preset) => {
            const isSelected = selectedTitle === preset.title;
            return (
              <div
                key={preset.id}
                onClick={() => onSelectCorpus(preset.text, preset.title)}
                className={`p-4 rounded-xl border cursor-pointer transition-all ${
                  isSelected
                    ? 'border-zinc-900 bg-zinc-50/80 shadow-xs ring-1 ring-zinc-900'
                    : 'border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="font-semibold text-sm text-zinc-900 flex items-center gap-1.5">
                    <span>{preset.title}</span>
                    {isSelected && <Check className="w-4 h-4 text-emerald-600" />}
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-200 text-zinc-700 uppercase">
                    {preset.category}
                  </span>
                </div>
                <p className="text-xs text-zinc-500 line-clamp-2">{preset.description}</p>
                <div className="text-[11px] text-zinc-600 font-mono mt-2">
                  {preset.text.length.toLocaleString()} chars · ~{preset.text.split(/\s+/).length} words
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Custom Upload or AI Synthesizer */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Upload / Paste Custom Text */}
        <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-zinc-900 flex items-center gap-1.5">
              <Upload className="w-4 h-4 text-zinc-600" />
              <span>Paste Custom Text or Upload .txt</span>
            </h3>
          </div>

          <p className="text-xs text-zinc-500">
            Have your own interview transcripts, notes, or code? Paste them below to pretrain on your exact data.
          </p>

          <textarea
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
            placeholder="Paste your text here (a few kilobytes or megabytes)..."
            rows={5}
            className="w-full bg-zinc-50 border border-zinc-300 rounded-xl p-3 text-xs font-mono text-zinc-900 focus:outline-none focus:ring-2 focus:ring-zinc-900"
          />

          <div className="flex items-center justify-between gap-3">
            <label className="cursor-pointer inline-flex items-center gap-1.5 text-xs text-zinc-600 hover:text-zinc-900 font-medium px-3 py-1.5 rounded-lg border border-zinc-200 hover:bg-zinc-50">
              <FileText className="w-3.5 h-3.5" />
              <span>Choose .txt File</span>
              <input type="file" accept=".txt" onChange={handleFileUpload} className="hidden" />
            </label>

            <button
              onClick={handleApplyCustomText}
              disabled={!customText.trim()}
              className="px-4 py-1.5 rounded-lg bg-zinc-900 text-white text-xs font-medium hover:bg-zinc-800 disabled:opacity-40 transition-colors"
            >
              Apply to Model
            </button>
          </div>

          {applyConfirmation && (
            <div id="custom-text-applied-confirmation" className="p-2.5 bg-emerald-50/90 border border-emerald-300 rounded-xl text-emerald-950 text-xs flex flex-wrap items-center gap-2">
              <Check className="w-4 h-4 text-emerald-600 shrink-0" />
              <div className="font-mono text-[11px]">
                <span className="font-sans font-semibold text-emerald-900">Applied to Model: </span>
                Active Title: <strong className="text-zinc-900">{applyConfirmation.activeTitle}</strong> · New Vocab Size V: <strong className="text-zinc-900">{applyConfirmation.vocabSize}</strong> · New Param Count: <strong className="text-zinc-900">{applyConfirmation.paramCount.toLocaleString()}</strong>
              </div>
            </div>
          )}
        </div>

        {/* AI Dataset Synthesizer */}
        <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-3">
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-indigo-600" />
            <h3 className="text-sm font-semibold text-zinc-900">Domain Corpus Synthesizer</h3>
          </div>

          <p className="text-xs text-zinc-500">
            Generate synthetic domain text (e.g. mock technical interviews or specialized logs) with high pattern consistency.
          </p>

          <div className="space-y-2">
            <label className="text-xs font-medium text-zinc-700">Topic / Domain:</label>
            <input
              type="text"
              value={synthTopic}
              onChange={(e) => setSynthTopic(e.target.value)}
              placeholder="e.g. AI System Design Interviews"
              className="w-full bg-zinc-50 border border-zinc-300 rounded-lg px-3 py-2 text-xs text-zinc-900"
            />

            <button
              onClick={handleSynthesize}
              disabled={isSynthesizing || !synthTopic.trim()}
              className="flex items-center justify-center gap-1.5 w-full py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium disabled:opacity-50 transition-colors shadow-xs"
            >
              {isSynthesizing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
              <span>{isSynthesizing ? 'Synthesizing Corpus...' : 'Generate Synthetic Dataset'}</span>
            </button>

            {synthMessage && (
              <div className="text-[11px] text-indigo-700 bg-indigo-50 p-2 rounded-md font-mono">
                {synthMessage}
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Tokenization Visualizer */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-zinc-900 flex items-center gap-2">
            <Hash className="w-4 h-4 text-emerald-600" />
            <span>Character Tokenization Sample</span>
          </h3>
          <span className="text-xs text-zinc-500 font-mono">First {sampleSnippet.length} characters</span>
        </div>

        <div className="p-3 bg-zinc-950 rounded-xl overflow-x-auto">
          <div className="flex flex-wrap gap-1.5 font-mono text-xs">
            {sampleSnippet.split('').map((char, idx) => {
              const charId = tokenizerData.charToId[char] ?? 0;
              return (
                <div key={idx} className="flex flex-col items-center bg-zinc-900 border border-zinc-800 rounded px-1.5 py-1 min-w-[24px]">
                  <span className="text-emerald-400 font-semibold text-[11px]">
                    {char === ' ' ? '␣' : char === '\n' ? '↵' : char}
                  </span>
                  <span className="text-zinc-500 text-[9px]">{charId}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

    </div>
  );
};
