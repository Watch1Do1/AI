import React, { useState } from 'react';
import { BookOpen, Check, Layers, Cpu, Zap, ShieldCheck, HelpCircle, ArrowRight } from 'lucide-react';

export const WhyThisWorks: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('decoder-only');

  const SECTIONS = [
    {
      id: 'decoder-only',
      title: '1. Why Decoder-Only Won (GPT vs. BERT / T5)',
      badge: 'Architecture',
      summary: 'Why the industry converged on causal autoregressive decoder-only models over encoder-decoder or masked-language architectures.'
    },
    {
      id: 'pre-ln',
      title: '2. Why Pre-LayerNorm (Pre-LN) Replaced Post-LN',
      badge: 'Gradient Flow',
      summary: 'How moving LayerNorm into the residual branches creates an uninhibited identity highway for deep backpropagation.'
    },
    {
      id: 'adamw',
      title: '3. Why Adam & AdamW are Mandatory for Transformers',
      badge: 'Optimization',
      summary: 'Why SGD fails on sparse attention patterns and how first/second moment tracking enables reliable convergence.'
    },
    {
      id: 'block-size',
      title: '4. Why Block Size (Context Length) Matters',
      badge: 'Compute & Memory',
      summary: 'The quadratic O(T²) attention bottleneck and why toy models calibrate context length carefully.'
    },
    {
      id: 'vocab-loss',
      title: '5. Why Vocab Size Dictates the Initial Loss Baseline',
      badge: 'Information Theory',
      summary: 'Mathematical derivation of initial random chance cross-entropy loss: L_0 = -ln(1/V) = ln(V).'
    }
  ];

  return (
    <div className="space-y-6">
      
      {/* Banner */}
      <div className="bg-zinc-900 text-white rounded-2xl p-6 border border-zinc-800 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-emerald-400 text-xs font-mono font-semibold">
          <BookOpen className="w-4 h-4" />
          <span>Foundational Theory & Design Decisions</span>
        </div>
        <h2 className="text-xl font-bold text-zinc-100">
          Why This Works: The Transformer Textbook
        </h2>
        <p className="text-sm text-zinc-300 leading-relaxed max-w-3xl">
          Deep-dive into the architectural choices that made modern Large Language Models possible. Clear mathematical proofs and engineering trade-offs behind Decoder-Only, Pre-LN, AdamW, and context limits.
        </p>
      </div>

      {/* Main Layout: Sidebar Navigation & Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Navigation Sidebar */}
        <div className="lg:col-span-4 space-y-2">
          {SECTIONS.map((sec) => (
            <div
              key={sec.id}
              onClick={() => setActiveSection(sec.id)}
              className={`p-4 rounded-xl border cursor-pointer transition-all ${
                activeSection === sec.id
                  ? 'bg-white border-zinc-900 shadow-sm ring-1 ring-zinc-900'
                  : 'bg-white border-zinc-200 hover:border-zinc-300 text-zinc-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-zinc-900">{sec.title}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 font-mono">
                  {sec.badge}
                </span>
              </div>
              <p className="text-[11px] text-zinc-500 mt-1 line-clamp-2 leading-relaxed">
                {sec.summary}
              </p>
            </div>
          ))}
        </div>

        {/* Detailed Chapter Content */}
        <div className="lg:col-span-8 bg-white rounded-2xl border border-zinc-200 p-6 shadow-xs text-xs space-y-6">
          
          {activeSection === 'decoder-only' && (
            <div className="space-y-4">
              <div>
                <span className="text-xs uppercase tracking-wider text-emerald-600 font-bold font-mono">Chapter 1</span>
                <h3 className="text-lg font-bold text-zinc-900 mt-1">Why Decoder-Only Transformers Won</h3>
              </div>

              <div className="p-4 bg-zinc-50 rounded-xl border border-zinc-200 space-y-2">
                <h4 className="font-bold text-zinc-900 text-sm">The 3 Historical Architectures:</h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 font-mono text-[11px] pt-1">
                  <div className="p-2.5 bg-white rounded border border-zinc-200">
                    <span className="font-bold text-zinc-900 block">Encoder-Only (BERT)</span>
                    <span className="text-zinc-500 text-[10px]">Bidirectional attention. Great for classification; cannot generate open text autoregressively.</span>
                  </div>
                  <div className="p-2.5 bg-white rounded border border-zinc-200">
                    <span className="font-bold text-zinc-900 block">Encoder-Decoder (T5)</span>
                    <span className="text-zinc-500 text-[10px]">Cross-attention between encoder and decoder. High overhead; complex caching.</span>
                  </div>
                  <div className="p-2.5 bg-emerald-50 rounded border border-emerald-300">
                    <span className="font-bold text-emerald-900 block">Decoder-Only (GPT)</span>
                    <span className="text-emerald-800 text-[10px]">Causal masking. Unifies prompt prefix and generated output into a single stream.</span>
                  </div>
                </div>
              </div>

              <div className="space-y-2.5 leading-relaxed text-zinc-700">
                <h4 className="font-bold text-zinc-900 text-sm">Key Advantages:</h4>
                <p>
                  <strong>1. Uniform Key-Value Caching:</strong> In a decoder-only model, every token at step $t$ only attends to positions $\le t$. When predicting token $t+1$, past activations never change. We can cache Key and Value vectors indefinitely, making inference $O(1)$ per token rather than recomputing the full prefix.
                </p>
                <p>
                  <strong>2. Zero Pretraining/Inference Discrepancy:</strong> Masked language models (like BERT) corrupt 15% of tokens with <code className="font-mono text-zinc-800">[MASK]</code> tokens during training, which never appear at runtime. Decoder-only models predict actual next tokens under pure maximum-likelihood estimation.
                </p>
                <p>
                  <strong>3. Few-Shot In-Context Learning:</strong> Because prompts and outputs share the exact same causal context window, the model treats instructions, examples, and generations as a continuous stream of text.
                </p>
              </div>
            </div>
          )}

          {activeSection === 'pre-ln' && (
            <div className="space-y-4">
              <div>
                <span className="text-xs uppercase tracking-wider text-emerald-600 font-bold font-mono">Chapter 2</span>
                <h3 className="text-lg font-bold text-zinc-900 mt-1">Why Pre-LayerNorm (Pre-LN) Replaced Post-LN</h3>
              </div>

              <div className="p-4 bg-zinc-900 text-white rounded-xl font-mono text-xs space-y-2">
                <div className="text-zinc-400 font-sans font-semibold">Mathematical Comparison:</div>
                <div className="p-2 rounded bg-zinc-950 text-rose-300 border border-zinc-800">
                  {"Post-LN (Original 2017 Transformer): x_{l+1} = \\text{LayerNorm}(x_l + \\text{SubLayer}(x_l))"}
                </div>
                <div className="p-2 rounded bg-zinc-950 text-emerald-400 border border-zinc-800">
                  {"Pre-LN (Modern GPT-2/3/Llama): x_{l+1} = x_l + \\text{SubLayer}(\\text{LayerNorm}(x_l))"}
                </div>
              </div>

              <div className="space-y-2.5 leading-relaxed text-zinc-700">
                <h4 className="font-bold text-zinc-900 text-sm">Why This Cured Transformer Instability:</h4>
                <p>
                  In Post-LN, the gradient flowing back from the output must pass directly through LayerNorm at every block:
                </p>
                <div className="p-2 rounded bg-zinc-100 font-mono text-[11px] text-zinc-800">
                  {"\\frac{\\partial x_{l+1}}{\\partial x_l} \\approx \\frac{\\partial \\text{LayerNorm}}{\\partial x_l} \\cdot (I + \\dots)"}
                </div>
                <p>
                  As networks get deeper, the gradients near the input decay or explode, requiring fragile warmup schedules and delicate learning rates.
                </p>
                <p>
                  In <strong>Pre-LN</strong>, the residual branch contains an uninhibited identity connection:
                </p>
                <div className="p-2 rounded bg-zinc-100 font-mono text-[11px] text-zinc-800">
                  {"x_L = x_0 + \\sum_{l=0}^{L-1} \\text{SubLayer}(\\text{LayerNorm}(x_l))"}
                </div>
                <p>
                  Gradients flow directly from layer $L$ to layer $0$ through pure addition, making training stable from step 0 even at large learning rates.
                </p>
              </div>
            </div>
          )}

          {activeSection === 'adamw' && (
            <div className="space-y-4">
              <div>
                <span className="text-xs uppercase tracking-wider text-emerald-600 font-bold font-mono">Chapter 3</span>
                <h3 className="text-lg font-bold text-zinc-900 mt-1">Why Adam & AdamW are Mandatory</h3>
              </div>

              <div className="p-4 bg-zinc-50 rounded-xl border border-zinc-200 space-y-2">
                <div className="font-bold text-zinc-900 text-sm">The 2 Moments of Adam:</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 font-mono text-[11px]">
                  <div className="p-2 bg-white rounded border border-zinc-200">
                    <span className="font-bold text-zinc-900 block">First Moment (m_t)</span>
                    <span className="text-zinc-600">m_t = β1 · m_(t-1) + (1 - β1) · g_t (Exponential moving average)</span>
                  </div>
                  <div className="p-2 bg-white rounded border border-zinc-200">
                    <span className="font-bold text-zinc-900 block">Second Moment (v_t)</span>
                    <span className="text-zinc-600">v_t = β2 · v_(t-1) + (1 - β2) · g_t² (Coordinate variance scale)</span>
                  </div>
                </div>
              </div>

              <div className="space-y-2.5 leading-relaxed text-zinc-700">
                <h4 className="font-bold text-zinc-900 text-sm">Why Standard SGD Fails in Transformers:</h4>
                <p>
                  <strong>Sparse and Non-Uniform Token Frequencies:</strong> Rare words or punctuation tokens appear infrequently in text batches. With standard Stochastic Gradient Descent (SGD), rare token embeddings receive tiny, sluggish updates while common tokens oscillate uncontrollably.
                </p>
                <p>
                  <strong>Per-Coordinate Adaptive Scaling:</strong> Adam divides the update by sqrt(v_t) + eps. Infrequent gradients receive larger effective step sizes, while noisy frequent gradients are damped.
                </p>
                <p>
                  <strong>Decoupled Weight Decay (AdamW):</strong> Standard L2 regularization adds weight to gradient, which gets distorted by sqrt(v_t). Loshchilov & Hutter (2017) proved that decaying weights directly (w = w - η · λ · w) restores true weight regularization.
                </p>
              </div>
            </div>
          )}

          {activeSection === 'block-size' && (
            <div className="space-y-4">
              <div>
                <span className="text-xs uppercase tracking-wider text-emerald-600 font-bold font-mono">Chapter 4</span>
                <h3 className="text-lg font-bold text-zinc-900 mt-1">Why Block Size (Context Length) Matters</h3>
              </div>

              <div className="p-4 bg-zinc-900 text-white rounded-xl font-mono text-xs space-y-2">
                <div className="flex justify-between text-zinc-400">
                  <span>Quadratic Complexity:</span>
                  <span className="text-emerald-400 font-bold">O(T² · d_model)</span>
                </div>
                <div className="p-2 rounded bg-zinc-950 text-zinc-300">
                  Attention Matrix Size = T × T floats per head
                </div>
              </div>

              <div className="space-y-2.5 leading-relaxed text-zinc-700">
                <h4 className="font-bold text-zinc-900 text-sm">The Engineering Trade-Off:</h4>
                <p>
                  Context length $T$ dictates how many tokens the model can look back simultaneously.
                </p>
                <ul className="list-disc pl-4 space-y-1">
                  <li>At $T = 32$: An attention matrix is $32 \times 32 = 1,024$ floats per head (negligible).</li>
                  <li>At $T = 2,048$ (GPT-3): An attention matrix is $4,194,304$ floats per head.</li>
                  <li>At $T = 128,000$ (Modern Llama 3): An attention matrix is $16.3$ billion floats per head, requiring FlashAttention and ring distributed attention.</li>
                </ul>
                <p>
                  For our 1-block educational lab, setting $T=32$ or $T=64$ allows real-time 60fps forward and backward passes directly in browser memory without WebGPU or external dependencies.
                </p>
              </div>
            </div>
          )}

          {activeSection === 'vocab-loss' && (
            <div className="space-y-4">
              <div>
                <span className="text-xs uppercase tracking-wider text-emerald-600 font-bold font-mono">Chapter 5</span>
                <h3 className="text-lg font-bold text-zinc-900 mt-1">Why Vocab Size Dictates Initial Loss</h3>
              </div>

              <div className="p-4 bg-zinc-50 rounded-xl border border-zinc-200 font-mono text-xs space-y-2">
                <div className="text-zinc-500 font-sans font-semibold">Mathematical Proof:</div>
                <div className="p-2.5 rounded bg-white border border-zinc-200 text-zinc-900">
                  {"p_i = 1 / V for all i in {1, ..., V}"}
                </div>
                <div className="p-2.5 rounded bg-zinc-900 text-emerald-400">
                  {"L_0 = -ln(p_target) = -ln(1/V) = ln(V)"}
                </div>
              </div>

              <div className="space-y-2.5 leading-relaxed text-zinc-700">
                <h4 className="font-bold text-zinc-900 text-sm">Concrete Examples Across Tokenizers:</h4>
                <ul className="list-disc pl-4 space-y-1">
                  <li>
                    <strong>Tiny Shakespeare (V = 65 characters):</strong><br />
                    Theoretical baseline loss: $\ln(65) \approx \mathbf{4.174}$.
                  </li>
                  <li>
                    <strong>Technical Interview Corpus (V = 72 characters):</strong><br />
                    Theoretical baseline loss: $\ln(72) \approx \mathbf{4.276}$.
                  </li>
                  <li>
                    <strong>GPT-2 BPE Tokenizer (V = 50,257 subwords):</strong><br />
                    Theoretical baseline loss: $\ln(50257) \approx \mathbf{10.825}$.
                  </li>
                </ul>
                <p>
                  Whenever you initialize random weights, the initial loss always starts exactly at $\ln(V)$. If it starts at 0 or 100, the initialization variance or loss calculation has a fatal bug!
                </p>
              </div>
            </div>
          )}

        </div>

      </div>

    </div>
  );
};
