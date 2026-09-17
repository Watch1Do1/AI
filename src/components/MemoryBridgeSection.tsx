import React, { useState } from 'react';
import { Database, Plus, Trash2, ArrowRight, CheckCircle2, XCircle, Search, Lightbulb, Code2, Layers } from 'lucide-react';
import { MemoryRecord } from '../types';

const INITIAL_MEMORIES: MemoryRecord[] = [
  {
    id: 'mem_1',
    category: 'Interview Profile',
    key: 'Candidate Identity',
    value: 'Alex Rivera, 5 years of experience in distributed consensus algorithms (Raft, Paxos) and deep learning systems.',
    timestamp: '2026-09-16 10:15'
  },
  {
    id: 'mem_2',
    category: 'Technical Response',
    key: 'Causal Masking Explanation',
    value: 'Alex stated: "Causal masking is essential in autoregressive generation to zero out attention scores for tokens t+1, preventing the model from cheating by attending to future tokens."',
    timestamp: '2026-09-16 10:22'
  },
  {
    id: 'mem_3',
    category: 'Architecture Choice',
    key: 'Memory Attachment Strategy',
    value: 'Alex recommended: "Decouple foundation pretraining from episodic memory. Keep the transformer frozen as a reasoning engine and attach an external vector/KV store for factual grounding."',
    timestamp: '2026-09-16 10:35'
  }
];

export const MemoryBridgeSection: React.FC = () => {
  const [memories, setMemories] = useState<MemoryRecord[]>(INITIAL_MEMORIES);
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');
  const [newCategory, setNewCategory] = useState('Interview Fact');
  const [userQuery, setUserQuery] = useState('What did Alex say about causal masking?');
  const [simulatedResult, setSimulatedResult] = useState<{
    matched: MemoryRecord | null;
    score: number;
    withoutMemory: string;
    withMemory: string;
    injectedPrompt: string;
  } | null>(null);

  const handleAddMemory = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKey.trim() || !newValue.trim()) return;
    const item: MemoryRecord = {
      id: `mem_${Date.now()}`,
      category: newCategory,
      key: newKey.trim(),
      value: newValue.trim(),
      timestamp: new Date().toISOString().replace('T', ' ').slice(0, 16)
    };
    setMemories([item, ...memories]);
    setNewKey('');
    setNewValue('');
  };

  const handleDeleteMemory = (id: string) => {
    setMemories(memories.filter(m => m.id !== id));
  };

  const handleRunQuery = () => {
    const qLower = userQuery.toLowerCase();
    
    // Semantic/keyword matching simulation
    let bestMatch: MemoryRecord | null = null;
    let bestScore = 0;

    for (const mem of memories) {
      const combined = `${mem.key} ${mem.value} ${mem.category}`.toLowerCase();
      const queryWords = qLower.split(/\s+/).filter(w => w.length > 2);
      let matchCount = 0;
      queryWords.forEach(w => {
        if (combined.includes(w)) matchCount++;
      });
      const score = queryWords.length > 0 ? matchCount / queryWords.length : 0;
      if (score > bestScore) {
        bestScore = score;
        bestMatch = mem;
      }
    }

    // Generate output contrast
    const withoutMemory = `I think someone named Alex might have talked about attention masks or interviews. As a raw pretrained model, I do not have access to any specific private interview transcripts.`;
    
    let withMemory = '';
    let injectedPrompt = '';

    if (bestMatch && bestScore > 0.15) {
      injectedPrompt = `SYSTEM: You are an assistant with access to an external memory store.
RELEVANT MEMORY RETRIEVED:
[Record ID: ${bestMatch.id} | Category: ${bestMatch.category}]
"${bestMatch.value}"

USER QUERY:
${userQuery}`;

      withMemory = `According to the recorded interview transcripts (Memory Record ${bestMatch.id}):\n${bestMatch.value}`;
    } else {
      injectedPrompt = `SYSTEM: You are an assistant with access to an external memory store.
NO RELEVANT MEMORY FOUND.

USER QUERY:
${userQuery}`;
      withMemory = `No matching interview memory found in the attached store for "${userQuery}". The foundation model avoids hallucinating private facts.`;
    }

    setSimulatedResult({
      matched: bestMatch,
      score: Math.min(0.98, bestScore + 0.35),
      withoutMemory,
      withMemory,
      injectedPrompt
    });
  };

  return (
    <div className="space-y-6">
      
      {/* Editorial Card: Direct Answer to User's Intuition */}
      <div className="bg-indigo-950 text-white rounded-2xl p-6 border border-indigo-900 shadow-sm space-y-4">
        <div className="flex items-center gap-2.5 text-indigo-300 text-xs font-mono uppercase tracking-wider font-semibold">
          <Lightbulb className="w-4 h-4 text-amber-400" />
          <span>Core Insight: Decoupling Foundation Pretraining from Memory</span>
        </div>
        
        <h2 className="text-xl font-bold text-indigo-50">
          "The interview script cannot grow into a foundation model — you train a model to learn the method, then attach your memory program."
        </h2>

        <p className="text-sm text-indigo-200 leading-relaxed">
          Your observation is <strong>the exact architectural division used in production AI systems</strong>. A foundation model is an engine of <em>probabilistic syntax, reasoning, and representation</em>. It is not an accurate key-value database. If you attempt to force personal interview transcripts into the weights of a small model:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 text-xs">
          <div className="p-3 rounded-xl bg-indigo-900/60 border border-indigo-800 space-y-1">
            <span className="font-semibold text-rose-300">1. Catastrophic Forgetting</span>
            <p className="text-indigo-200">New updates overwrite existing representations, causing erratic loss spikes.</p>
          </div>
          <div className="p-3 rounded-xl bg-indigo-900/60 border border-indigo-800 space-y-1">
            <span className="font-semibold text-rose-300">2. Stochastic Hallucination</span>
            <p className="text-indigo-200">Weights generate plausible-sounding text, not verified verbatim facts.</p>
          </div>
          <div className="p-3 rounded-xl bg-indigo-900/60 border border-indigo-800 space-y-1">
            <span className="font-semibold text-emerald-300">3. The Attached Memory Fix</span>
            <p className="text-indigo-200">Store facts in an external memory program (Vector/KV store) and feed them dynamically.</p>
          </div>
        </div>
      </div>

      {/* Interactive Memory Attachment Simulator */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-5">
        <div className="border-b border-zinc-100 pb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-base font-semibold text-zinc-900 flex items-center gap-2">
              <Database className="w-4 h-4 text-indigo-600" />
              <span>Interactive Memory Program Workbench</span>
            </h3>
            <p className="text-xs text-zinc-500">
              Simulate attaching an external episodic memory program to a base foundation model
            </p>
          </div>

          <div className="text-xs font-mono text-zinc-500">
            {memories.length} Active Records in Bank
          </div>
        </div>

        {/* Query Runner */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-zinc-700">Ask a Question About the Interview / Domain:</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={userQuery}
              onChange={(e) => setUserQuery(e.target.value)}
              placeholder="e.g. What did Alex say about causal masking?"
              className="flex-1 bg-zinc-50 border border-zinc-300 rounded-lg px-3 py-2 text-sm text-zinc-900 focus:outline-none focus:ring-2 focus:ring-zinc-900"
            />
            <button
              onClick={handleRunQuery}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors shadow-xs"
            >
              <Search className="w-4 h-4" />
              <span>Execute Query</span>
            </button>
          </div>
        </div>

        {/* Side-by-Side Comparison: Base Model Alone vs Base Model + Memory Program */}
        {simulatedResult && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            
            {/* Base Model Alone */}
            <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200 space-y-2">
              <div className="flex items-center gap-2 text-zinc-800 text-xs font-semibold">
                <XCircle className="w-4 h-4 text-rose-500" />
                <span>Base Model Alone (Weights Only)</span>
              </div>
              <p className="text-xs text-zinc-600">
                Has only raw pretrained weights. Cannot recall private interview facts:
              </p>
              <div className="p-3 bg-white rounded-lg border border-zinc-200 text-xs text-zinc-700 font-mono italic">
                "{simulatedResult.withoutMemory}"
              </div>
            </div>

            {/* Base Model + Attached Memory Program */}
            <div className="p-4 rounded-xl bg-indigo-50/70 border border-indigo-200 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-indigo-950 text-xs font-semibold">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>Base Model + Attached Memory Program</span>
                </div>
                {simulatedResult.matched && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-100 text-indigo-800 font-medium">
                    Match Confidence: {(simulatedResult.score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <p className="text-xs text-indigo-900">
                Memory program retrieves exact record and injects it into context window:
              </p>
              <div className="p-3 bg-white rounded-lg border border-indigo-200 text-xs text-indigo-950 font-mono whitespace-pre-wrap">
                {simulatedResult.withMemory}
              </div>
            </div>

          </div>
        )}

        {/* Prompt Context Injection Trace */}
        {simulatedResult?.injectedPrompt && (
          <div className="p-3 rounded-lg bg-zinc-900 text-zinc-300 font-mono text-xs space-y-1">
            <div className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider">
              Runtime Context Construction (What the Transformer Actually Receives):
            </div>
            <pre className="text-zinc-300 whitespace-pre-wrap text-[11px]">
              {simulatedResult.injectedPrompt}
            </pre>
          </div>
        )}

        {/* Memory Bank List */}
        <div className="space-y-3 pt-2">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-zinc-700 uppercase tracking-wider">
              Attached Memory Bank (External Episodic Store)
            </h4>
          </div>

          <div className="grid grid-cols-1 gap-2 max-h-56 overflow-y-auto pr-1">
            {memories.map((mem) => (
              <div
                key={mem.id}
                className="p-3 rounded-xl bg-zinc-50 border border-zinc-200 flex items-start justify-between gap-3 text-xs"
              >
                <div className="space-y-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-zinc-900">{mem.key}</span>
                    <span className="px-1.5 py-0.2 rounded bg-zinc-200 text-zinc-700 text-[10px]">
                      {mem.category}
                    </span>
                  </div>
                  <p className="text-zinc-600 text-xs leading-relaxed">{mem.value}</p>
                  <span className="text-[10px] text-zinc-600 block">{mem.timestamp}</span>
                </div>

                <button
                  onClick={() => handleDeleteMemory(mem.id)}
                  className="text-zinc-600 hover:text-rose-600 p-1 transition-colors"
                  title="Delete memory record"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>

          {/* Add New Memory Form */}
          <form onSubmit={handleAddMemory} className="p-3 bg-zinc-100 rounded-xl border border-zinc-200 space-y-2 text-xs">
            <div className="font-medium text-zinc-800">Add New Fact to Memory Bank:</div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <input
                type="text"
                placeholder="Topic / Key (e.g. Candidate Preferences)"
                value={newKey}
                onChange={(e) => setNewKey(e.target.value)}
                className="bg-white border border-zinc-300 rounded-lg px-2.5 py-1.5 text-zinc-900"
              />
              <input
                type="text"
                placeholder="Content / Verbatim fact..."
                value={newValue}
                onChange={(e) => setNewValue(e.target.value)}
                className="sm:col-span-2 bg-white border border-zinc-300 rounded-lg px-2.5 py-1.5 text-zinc-900"
              />
            </div>
            <button
              type="submit"
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-zinc-900 text-white font-medium hover:bg-zinc-800 transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Add Memory</span>
            </button>
          </form>
        </div>

      </div>

      {/* Architectural Implementation Blueprint: Python Integration */}
      <div className="bg-white rounded-2xl border border-zinc-200 p-5 shadow-xs space-y-3">
        <div className="flex items-center gap-2">
          <Code2 className="w-4 h-4 text-emerald-600" />
          <h3 className="text-sm font-semibold text-zinc-900">
            How to Attach Memory to Your Trained TinyGPT in Python
          </h3>
        </div>
        <p className="text-xs text-zinc-600">
          This sample Python wrapper shows how the trained checkpoint weights from `train_tiny_gpt.py` are loaded, while candidate interview notes live in an episodic memory dictionary:
        </p>

        <div className="bg-zinc-950 p-4 rounded-xl text-zinc-200 font-mono text-xs overflow-x-auto">
          <pre>{`import torch

class TinyGPTWithMemoryEngine:
    def __init__(self, checkpoint_path="tiny_gpt_weights.pt"):
        # 1. Load the frozen pretrained model weights
        checkpoint = torch.load(checkpoint_path)
        self.model = load_model(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 2. Attached external memory program (store explicit facts)
        self.memory_bank = {}

    def store_memory(self, key, text):
        self.memory_bank[key] = text

    def query(self, user_question):
        # 3. Retrieve matching episodic memory
        relevant_context = self.retrieve_memory(user_question)
        
        # 4. Inject retrieved memory into prompt context
        prompt = f"MEMORY:\\n{relevant_context}\\n\\nQUESTION: {user_question}\\nANSWER:"
        
        # 5. Foundation model generates grounded output
        return self.model.generate(prompt)

# Usage:
engine = TinyGPTWithMemoryEngine()
engine.store_memory("candidate_1", "Alex Rivera passed the distributed systems interview.")
response = engine.query("What was Alex's result?")`}</pre>
        </div>
      </div>

    </div>
  );
};
