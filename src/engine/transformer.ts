import { ModelConfig, AttentionHeadData } from '../types';

// Helper: Xavier/Glorot Normal initialization
function randomMatrix(rows: number, cols: number, scale: number = 0.08): Float32Array {
  const arr = new Float32Array(rows * cols);
  for (let i = 0; i < arr.length; i++) {
    // Box-Muller transform for normal distribution
    const u1 = Math.max(1e-7, Math.random());
    const u2 = Math.random();
    const randStd = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    arr[i] = randStd * scale;
  }
  return arr;
}

function zeros(size: number): Float32Array {
  return new Float32Array(size);
}

// Softmax with numerical max subtraction
function softmaxInPlace(arr: Float32Array | number[]): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max) max = arr[i];
  }
  let sum = 0;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = Math.exp(arr[i] - max);
    sum += out[i];
  }
  const invSum = 1.0 / (sum + 1e-12);
  for (let i = 0; i < arr.length; i++) {
    out[i] *= invSum;
  }
  return out;
}

export class MicroGPT {
  config: ModelConfig;

  // Parameters
  wte: Float32Array; // [vocabSize, nEmbd]
  wpe: Float32Array; // [blockSize, nEmbd]

  // Layer 0 Attention
  wq: Float32Array;  // [nEmbd, nEmbd]
  wk: Float32Array;  // [nEmbd, nEmbd]
  wv: Float32Array;  // [nEmbd, nEmbd]
  wo: Float32Array;  // [nEmbd, nEmbd]
  
  // Layer 0 MLP
  w1: Float32Array;  // [nEmbd, 4 * nEmbd]
  b1: Float32Array;  // [4 * nEmbd]
  w2: Float32Array;  // [4 * nEmbd, nEmbd]
  b2: Float32Array;  // [nEmbd]

  // LM Head
  lmHead: Float32Array; // [nEmbd, vocabSize]

  // Adam Optimizer state (first & second moments)
  private m: Map<string, Float32Array> = new Map();
  private v: Map<string, Float32Array> = new Map();
  private adamStep: number = 0;

  // Last cached attention maps for visualization
  lastAttentionMaps: AttentionHeadData[] = [];

  // Latest gradient and activation telemetry
  latestTelemetry: import('../types').GradientTelemetry | null = null;

  constructor(config: ModelConfig) {
    this.config = config;
    const { vocabSize, blockSize, nEmbd } = config;

    this.wte = randomMatrix(vocabSize, nEmbd, 0.08);
    this.wpe = randomMatrix(blockSize, nEmbd, 0.04);

    this.wq = randomMatrix(nEmbd, nEmbd, Math.sqrt(2.0 / nEmbd));
    this.wk = randomMatrix(nEmbd, nEmbd, Math.sqrt(2.0 / nEmbd));
    this.wv = randomMatrix(nEmbd, nEmbd, Math.sqrt(2.0 / nEmbd));
    this.wo = randomMatrix(nEmbd, nEmbd, Math.sqrt(2.0 / nEmbd));

    const mlpHidden = 4 * nEmbd;
    this.w1 = randomMatrix(nEmbd, mlpHidden, Math.sqrt(2.0 / nEmbd));
    this.b1 = zeros(mlpHidden);
    this.w2 = randomMatrix(mlpHidden, nEmbd, Math.sqrt(2.0 / mlpHidden));
    this.b2 = zeros(nEmbd);

    this.lmHead = randomMatrix(nEmbd, vocabSize, 0.08);
    this.initAdam();
  }

  private initAdam() {
    this.adamStep = 0;
    this.m.clear();
    this.v.clear();
    const params: [string, Float32Array][] = [
      ['wte', this.wte],
      ['wpe', this.wpe],
      ['wq', this.wq],
      ['wk', this.wk],
      ['wv', this.wv],
      ['wo', this.wo],
      ['w1', this.w1],
      ['b1', this.b1],
      ['w2', this.w2],
      ['b2', this.b2],
      ['lmHead', this.lmHead]
    ];
    for (const [name, param] of params) {
      this.m.set(name, new Float32Array(param.length));
      this.v.set(name, new Float32Array(param.length));
    }
  }

  // Count total trainable parameters
  getTotalParameters(): number {
    return (
      this.wte.length +
      this.wpe.length +
      this.wq.length +
      this.wk.length +
      this.wv.length +
      this.wo.length +
      this.w1.length +
      this.b1.length +
      this.w2.length +
      this.b2.length +
      this.lmHead.length
    );
  }

  // Forward pass for a single sequence of token IDs
  forward(tokens: number[], storeAttention: boolean = true): {
    logits: Float32Array[]; // [seqLen][vocabSize]
    activations: any;
  } {
    const { blockSize, nEmbd, nHead, vocabSize } = this.config;
    const T = Math.min(tokens.length, blockSize);
    const headDim = Math.floor(nEmbd / nHead);
    const scale = 1.0 / Math.sqrt(headDim);

    // 1. Embeddings: x = wte[token] + wpe[pos]
    const x: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const tok = Math.min(Math.max(0, tokens[t]), vocabSize - 1);
      const xt = new Float32Array(nEmbd);
      const tokOffset = tok * nEmbd;
      const posOffset = t * nEmbd;
      for (let d = 0; d < nEmbd; d++) {
        xt[d] = this.wte[tokOffset + d] + this.wpe[posOffset + d];
      }
      x.push(xt);
    }

    // 2. LayerNorm 1 (simple mean-subtraction + scaling)
    const xNorm: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const vec = new Float32Array(nEmbd);
      let mean = 0;
      for (let d = 0; d < nEmbd; d++) mean += x[t][d];
      mean /= nEmbd;
      let variance = 0;
      for (let d = 0; d < nEmbd; d++) variance += (x[t][d] - mean) ** 2;
      variance /= nEmbd;
      const std = Math.sqrt(variance + 1e-5);
      for (let d = 0; d < nEmbd; d++) {
        vec[d] = (x[t][d] - mean) / std;
      }
      xNorm.push(vec);
    }

    // 3. Multi-Head Self-Attention with Causal Mask
    // Projections Q, K, V
    const Q: Float32Array[] = [];
    const K: Float32Array[] = [];
    const V: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const qVec = new Float32Array(nEmbd);
      const kVec = new Float32Array(nEmbd);
      const vVec = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i++) {
        let sumQ = 0, sumK = 0, sumV = 0;
        const offset = i * nEmbd;
        for (let j = 0; j < nEmbd; j++) {
          sumQ += xNorm[t][j] * this.wq[offset + j];
          sumK += xNorm[t][j] * this.wk[offset + j];
          sumV += xNorm[t][j] * this.wv[offset + j];
        }
        qVec[i] = sumQ;
        kVec[i] = sumK;
        vVec[i] = sumV;
      }
      Q.push(qVec);
      K.push(kVec);
      V.push(vVec);
    }

    // Attention calculation per head
    const attnOut = x.map(() => new Float32Array(nEmbd));
    const capturedAttentionMaps: AttentionHeadData[] = [];

    for (let h = 0; h < nHead; h++) {
      const headOffset = h * headDim;
      const headMatrix: number[][] = [];

      for (let t = 0; t < T; t++) {
        const scores = new Float32Array(t + 1); // Causal: only look back up to t
        for (let prev = 0; prev <= t; prev++) {
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += Q[t][headOffset + d] * K[prev][headOffset + d];
          }
          scores[prev] = dot * scale;
        }

        const weights = softmaxInPlace(scores);
        if (storeAttention) {
          const row: number[] = new Array(T).fill(0);
          for (let prev = 0; prev <= t; prev++) {
            row[prev] = weights[prev];
          }
          headMatrix.push(row);
        }

        // Weighted sum of V
        for (let d = 0; d < headDim; d++) {
          let valSum = 0;
          for (let prev = 0; prev <= t; prev++) {
            valSum += weights[prev] * V[prev][headOffset + d];
          }
          attnOut[t][headOffset + d] += valSum;
        }
      }

      if (storeAttention) {
        capturedAttentionMaps.push({ headIndex: h, matrix: headMatrix });
      }
    }

    if (storeAttention) {
      this.lastAttentionMaps = capturedAttentionMaps;
    }

    // Projection Wo and Residual connection 1: x_mid = x + attnOut @ Wo
    const xMid: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const proj = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i++) {
        let s = 0;
        const offset = i * nEmbd;
        for (let j = 0; j < nEmbd; j++) {
          s += attnOut[t][j] * this.wo[offset + j];
        }
        proj[i] = x[t][i] + s;
      }
      xMid.push(proj);
    }

    // 4. FeedForward Network (MLP) with Residual connection
    const mlpHidden = 4 * nEmbd;
    const xFinal: Float32Array[] = [];

    for (let t = 0; t < T; t++) {
      // Hidden = ReLU(xMid @ w1 + b1)
      const hVec = new Float32Array(mlpHidden);
      for (let i = 0; i < mlpHidden; i++) {
        let s = this.b1[i];
        for (let j = 0; j < nEmbd; j++) {
          s += xMid[t][j] * this.w1[j * mlpHidden + i];
        }
        // GELU approximation or fast ReLU
        hVec[i] = Math.max(0, s);
      }

      // Out = hVec @ w2 + b2 + residual
      const outVec = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i++) {
        let s = this.b2[i];
        for (let j = 0; j < mlpHidden; j++) {
          s += hVec[j] * this.w2[j * nEmbd + i];
        }
        outVec[i] = xMid[t][i] + s;
      }
      xFinal.push(outVec);
    }

    // 5. Language Model Head (Logits)
    const logits: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const logitVec = new Float32Array(vocabSize);
      for (let v = 0; v < vocabSize; v++) {
        let s = 0;
        for (let d = 0; d < nEmbd; d++) {
          s += xFinal[t][d] * this.lmHead[d * vocabSize + v];
        }
        logitVec[v] = s;
      }
      logits.push(logitVec);
    }

    return {
      logits,
      activations: { x, xNorm, xMid, xFinal, Q, K, V }
    };
  }

  // Evaluate loss on a batch without gradient updates (used for validation loss)
  evaluateLoss(batchTokens: { input: number[]; target: number[] }[]): number {
    const { vocabSize } = this.config;
    let totalLoss = 0;
    let totalTokens = 0;

    for (const item of batchTokens) {
      const tokens = item.input;
      const targets = item.target;
      const T = tokens.length;
      if (T === 0) continue;

      const { logits } = this.forward(tokens, false);

      for (let t = 0; t < T; t++) {
        const targetTok = Math.min(Math.max(0, targets[t]), vocabSize - 1);
        const probs = softmaxInPlace(logits[t]);
        const p = Math.max(1e-12, probs[targetTok]);
        totalLoss += -Math.log(p);
        totalTokens++;
      }
    }

    return totalTokens > 0 ? totalLoss / totalTokens : 0;
  }

  // Single step training on a batch of token slices
  // Note: Browser trainStep optimizes embeddings, MLP (w1, b1, w2, b2), and lm_head.
  // Full analytic backprop through causal self-attention is performed in PyTorch (python/train_tiny_gpt.py).
  trainStep(
    batchTokens: { input: number[]; target: number[] }[],
    lr: number = this.config.lr
  ): { loss: number; perplexity: number; telemetry?: import('../types').GradientTelemetry | null } {
    const { vocabSize, nEmbd } = this.config;
    let totalLoss = 0;
    let totalTokens = 0;

    // Gradient accumulators
    const dLmHead = new Float32Array(this.lmHead.length);
    const dWte = new Float32Array(this.wte.length);
    const dWpe = new Float32Array(this.wpe.length);
    const dW1 = new Float32Array(this.w1.length);
    const dB1 = new Float32Array(this.b1.length);
    const dW2 = new Float32Array(this.w2.length);
    const dB2 = new Float32Array(this.b2.length);

    for (const item of batchTokens) {
      const tokens = item.input;
      const targets = item.target;
      const T = tokens.length;
      if (T === 0) continue;

      const { logits, activations } = this.forward(tokens, false);

      for (let t = 0; t < T; t++) {
        const targetTok = Math.min(Math.max(0, targets[t]), vocabSize - 1);
        const probs = softmaxInPlace(logits[t]);

        // Cross entropy loss = -ln(probs[targetTok])
        const p = Math.max(1e-12, probs[targetTok]);
        const stepLoss = -Math.log(p);
        totalLoss += stepLoss;
        totalTokens++;

        // dLoss / dLogits = probs - 1 at target, probs elsewhere
        const dLogits = new Float32Array(vocabSize);
        for (let v = 0; v < vocabSize; v++) {
          dLogits[v] = probs[v];
        }
        dLogits[targetTok] -= 1.0;

        // Backprop into lmHead and xFinal[t]
        const dXFinal = new Float32Array(nEmbd);
        for (let v = 0; v < vocabSize; v++) {
          const grad = dLogits[v];
          if (Math.abs(grad) < 1e-8) continue;
          for (let d = 0; d < nEmbd; d++) {
            const idx = d * vocabSize + v;
            dLmHead[idx] += activations.xFinal[t][d] * grad;
            dXFinal[d] += this.lmHead[idx] * grad;
          }
        }

        // Backprop through MLP
        const mlpHidden = 4 * nEmbd;
        const dXMid = new Float32Array(dXFinal); // residual connection

        // Accumulate bias gradient for b2
        for (let d = 0; d < nEmbd; d++) {
          dB2[d] += dXFinal[d];
        }

        for (let j = 0; j < mlpHidden; j++) {
          // Recompute activation for hidden unit j
          let actH = this.b1[j];
          for (let k = 0; k < nEmbd; k++) {
            actH += activations.xMid[t][k] * this.w1[k * mlpHidden + j];
          }

          if (actH > 0) { // ReLU derivative
            let s = 0;
            for (let d = 0; d < nEmbd; d++) {
              const idx = j * nEmbd + d;
              dW2[idx] += actH * dXFinal[d];
              s += this.w2[idx] * dXFinal[d];
            }
            dB1[j] += s;
            for (let k = 0; k < nEmbd; k++) {
              dW1[k * mlpHidden + j] += activations.xMid[t][k] * s;
              dXMid[k] += this.w1[k * mlpHidden + j] * s;
            }
          }
        }

        // Embedding updates (Token & Positional) through residual
        const tok = tokens[t];
        const tokOffset = tok * nEmbd;
        const posOffset = t * nEmbd;
        for (let d = 0; d < nEmbd; d++) {
          dWte[tokOffset + d] += dXMid[d];
          dWpe[posOffset + d] += dXMid[d];
        }
      }
    }

    if (totalTokens === 0) return { loss: 0, perplexity: 1 };

    const avgLoss = totalLoss / totalTokens;
    const invN = 1.0 / totalTokens;

    // Adam optimizer update step
    this.adamStep++;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const lr_t = lr * Math.sqrt(1.0 - Math.pow(beta2, this.adamStep)) / (1.0 - Math.pow(beta1, this.adamStep));

    const applyAdam = (name: string, param: Float32Array, grad: Float32Array) => {
      const mArr = this.m.get(name)!;
      const vArr = this.v.get(name)!;
      for (let i = 0; i < param.length; i++) {
        // Gradient clipping at [-1.0, 1.0]
        let g = grad[i] * invN;
        if (g > 1.0) g = 1.0;
        else if (g < -1.0) g = -1.0;

        mArr[i] = beta1 * mArr[i] + (1.0 - beta1) * g;
        vArr[i] = beta2 * vArr[i] + (1.0 - beta2) * g * g;
        const delta = lr_t * mArr[i] / (Math.sqrt(vArr[i]) + eps);
        param[i] -= delta;
      }
    };

    applyAdam('lmHead', this.lmHead, dLmHead);
    applyAdam('wte', this.wte, dWte);
    applyAdam('wpe', this.wpe, dWpe);
    applyAdam('w1', this.w1, dW1);
    applyAdam('b1', this.b1, dB1);
    applyAdam('w2', this.w2, dW2);
    applyAdam('b2', this.b2, dB2);

    // Compute L2 Gradient Norms
    const norm2 = (arr: Float32Array, scale: number = invN): number => {
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {
        const val = arr[i] * scale;
        sum += val * val;
      }
      return Math.sqrt(sum);
    };

    const wteNorm = norm2(dWte);
    const wpeNorm = norm2(dWpe);
    const w1Norm = norm2(dW1);
    const w2Norm = norm2(dW2);
    const lmHeadNorm = norm2(dLmHead);
    const totalNorm = Math.sqrt(wteNorm * wteNorm + wpeNorm * wpeNorm + w1Norm * w1Norm + w2Norm * w2Norm + lmHeadNorm * lmHeadNorm);

    // Compute basic activation statistics from the last forward pass
    const computeActStats = (name: string, acts: Float32Array[]) => {
      let sum = 0;
      let sumSq = 0;
      let min = Infinity;
      let max = -Infinity;
      let count = 0;
      for (const vec of acts) {
        for (let i = 0; i < vec.length; i++) {
          const v = vec[i];
          sum += v;
          sumSq += v * v;
          if (v < min) min = v;
          if (v > max) max = v;
          count++;
        }
      }
      const mean = count > 0 ? sum / count : 0;
      const variance = count > 0 ? (sumSq / count) - (mean * mean) : 0;
      
      // 8-bin histogram between min and max
      const bins = [0, 0, 0, 0, 0, 0, 0, 0];
      const range = Math.max(1e-5, max - min);
      for (const vec of acts) {
        for (let i = 0; i < vec.length; i++) {
          const binIdx = Math.min(7, Math.max(0, Math.floor(((vec[i] - min) / range) * 8)));
          bins[binIdx]++;
        }
      }
      return { name, mean, variance, min: isFinite(min) ? min : 0, max: isFinite(max) ? max : 0, bins };
    };

    const lastForward = this.forward(batchTokens[0]?.input || [0], true);
    const activationStats = [
      computeActStats('Token Embeds (x)', lastForward.activations.x),
      computeActStats('Post-LayerNorm (xNorm)', lastForward.activations.xNorm),
      computeActStats('Post-Attn Residual (xMid)', lastForward.activations.xMid),
      computeActStats('MLP Output (xFinal)', lastForward.activations.xFinal)
    ];

    this.latestTelemetry = {
      step: this.adamStep,
      totalNorm,
      wteNorm,
      wpeNorm,
      w1Norm,
      w2Norm,
      lmHeadNorm,
      isVanishing: totalNorm < 1e-4,
      isExploding: totalNorm > 4.5,
      activationStats,
      attentionLogitsHeatmap: this.lastAttentionMaps.length > 0 ? {
        headIndex: 0,
        rawLogits: this.lastAttentionMaps[0].matrix.map(row => row.map(v => v > 0 ? Math.log(v + 1e-6) : -10)),
        softmaxProbs: this.lastAttentionMaps[0].matrix
      } : undefined
    };

    return {
      loss: avgLoss,
      perplexity: Math.min(10000, Math.exp(avgLoss)),
      telemetry: this.latestTelemetry
    };
  }

  // Autoregressive generation given prompt tokens
  generate(
    promptTokens: number[],
    maxNewTokens: number = 50,
    temperature: number = 0.8,
    topK: number = 10
  ): { generatedTokens: number[]; attentionMaps: AttentionHeadData[] } {
    const tokens = [...promptTokens];
    const { blockSize, vocabSize } = this.config;

    for (let step = 0; step < maxNewTokens; step++) {
      // Crop context to block size
      const context = tokens.slice(-blockSize);
      const { logits } = this.forward(context, true);
      const lastLogits = logits[logits.length - 1];

      // Apply temperature
      const temp = Math.max(0.01, temperature);
      const scaledLogits = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        scaledLogits[i] = lastLogits[i] / temp;
      }

      // Top-K filtering
      const indexed = Array.from(scaledLogits).map((logit, idx) => ({ idx, logit }));
      indexed.sort((a, b) => b.logit - a.logit);

      const k = Math.max(1, Math.min(topK, vocabSize));
      const topKIndices = indexed.slice(0, k);

      // Softmax over top-K
      const topKLogits = new Float32Array(k);
      for (let i = 0; i < k; i++) topKLogits[i] = topKIndices[i].logit;
      const topKProbs = softmaxInPlace(topKLogits);

      // Sample from distribution
      const r = Math.random();
      let cum = 0;
      let nextToken = topKIndices[0].idx;
      for (let i = 0; i < k; i++) {
        cum += topKProbs[i];
        if (r <= cum) {
          nextToken = topKIndices[i].idx;
          break;
        }
      }

      tokens.push(nextToken);
    }

    return {
      generatedTokens: tokens,
      attentionMaps: this.lastAttentionMaps
    };
  }

  // Clone snapshot of weights for checkpoint timeline
  cloneWeights() {
    return {
      wte: new Float32Array(this.wte),
      wpe: new Float32Array(this.wpe),
      wq: new Float32Array(this.wq),
      wk: new Float32Array(this.wk),
      wv: new Float32Array(this.wv),
      wo: new Float32Array(this.wo),
      w1: new Float32Array(this.w1),
      b1: new Float32Array(this.b1),
      w2: new Float32Array(this.w2),
      b2: new Float32Array(this.b2),
      lmHead: new Float32Array(this.lmHead)
    };
  }

  // Restore snapshot of weights from checkpoint
  loadWeights(snapshot: {
    wte: Float32Array;
    wpe: Float32Array;
    wq: Float32Array;
    wk: Float32Array;
    wv: Float32Array;
    wo: Float32Array;
    w1: Float32Array;
    b1: Float32Array;
    w2: Float32Array;
    b2: Float32Array;
    lmHead: Float32Array;
  }) {
    this.wte.set(snapshot.wte);
    this.wpe.set(snapshot.wpe);
    this.wq.set(snapshot.wq);
    this.wk.set(snapshot.wk);
    this.wv.set(snapshot.wv);
    this.wo.set(snapshot.wo);
    this.w1.set(snapshot.w1);
    this.b1.set(snapshot.b1);
    this.w2.set(snapshot.w2);
    this.b2.set(snapshot.b2);
    this.lmHead.set(snapshot.lmHead);
  }

  // High-fidelity forward pass inspector for the Step-Through Debugger
  stepThroughDetailed(tokens: number[]) {
    const { blockSize, nEmbd, nHead, vocabSize } = this.config;
    const T = Math.min(tokens.length, blockSize);
    const headDim = Math.floor(nEmbd / nHead);
    const scale = 1.0 / Math.sqrt(headDim);

    // Stage 0: Embeddings
    const xTokens: Float32Array[] = [];
    const xPositions: Float32Array[] = [];
    const xCombined: Float32Array[] = [];

    for (let t = 0; t < T; t++) {
      const tokId = Math.min(Math.max(0, tokens[t]), vocabSize - 1);
      const tokVec = new Float32Array(nEmbd);
      const posVec = new Float32Array(nEmbd);
      const combVec = new Float32Array(nEmbd);

      const tokOffset = tokId * nEmbd;
      const posOffset = t * nEmbd;
      for (let d = 0; d < nEmbd; d++) {
        tokVec[d] = this.wte[tokOffset + d];
        posVec[d] = this.wpe[posOffset + d];
        combVec[d] = tokVec[d] + posVec[d];
      }
      xTokens.push(tokVec);
      xPositions.push(posVec);
      xCombined.push(combVec);
    }

    // Stage 1 & 2: Projections, Raw Attention Logits & Softmax
    const Q: Float32Array[] = [];
    const K: Float32Array[] = [];
    const V: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const qVec = new Float32Array(nEmbd);
      const kVec = new Float32Array(nEmbd);
      const vVec = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i++) {
        let sumQ = 0, sumK = 0, sumV = 0;
        const offset = i * nEmbd;
        for (let j = 0; j < nEmbd; j++) {
          sumQ += xCombined[t][j] * this.wq[offset + j];
          sumK += xCombined[t][j] * this.wk[offset + j];
          sumV += xCombined[t][j] * this.wv[offset + j];
        }
        qVec[i] = sumQ;
        kVec[i] = sumK;
        vVec[i] = sumV;
      }
      Q.push(qVec);
      K.push(kVec);
      V.push(vVec);
    }

    const rawAttentionScores: number[][][] = []; // [head][T][T]
    const softmaxProbabilities: number[][][] = []; // [head][T][T]
    const weightedVPerHead: Float32Array[][] = []; // [head][T]
    const attnOut = xCombined.map(() => new Float32Array(nEmbd));

    for (let h = 0; h < nHead; h++) {
      const headOffset = h * headDim;
      const rawMatrix: number[][] = [];
      const probMatrix: number[][] = [];
      const vSumList: Float32Array[] = [];

      for (let t = 0; t < T; t++) {
        const rawRow = new Array(T).fill(0);
        const probRow = new Array(T).fill(0);
        const causalScores = new Float32Array(t + 1);

        for (let prev = 0; prev <= t; prev++) {
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += Q[t][headOffset + d] * K[prev][headOffset + d];
          }
          const scaled = dot * scale;
          rawRow[prev] = scaled;
          causalScores[prev] = scaled;
        }

        const weights = softmaxInPlace(causalScores);
        for (let prev = 0; prev <= t; prev++) {
          probRow[prev] = weights[prev];
        }
        rawMatrix.push(rawRow);
        probMatrix.push(probRow);

        const vHeadSum = new Float32Array(headDim);
        for (let d = 0; d < headDim; d++) {
          let val = 0;
          for (let prev = 0; prev <= t; prev++) {
            val += weights[prev] * V[prev][headOffset + d];
          }
          vHeadSum[d] = val;
          attnOut[t][headOffset + d] += val;
        }
        vSumList.push(vHeadSum);
      }

      rawAttentionScores.push(rawMatrix);
      softmaxProbabilities.push(probMatrix);
      weightedVPerHead.push(vSumList);
    }

    // Stage 3 & 4: Output Projection and Residual 1
    const xMid: Float32Array[] = [];
    const attnProj: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const proj = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i++) {
        let s = 0;
        const offset = i * nEmbd;
        for (let j = 0; j < nEmbd; j++) {
          s += attnOut[t][j] * this.wo[offset + j];
        }
        proj[i] = s;
      }
      attnProj.push(proj);

      const midVec = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i++) {
        midVec[i] = xCombined[t][i] + proj[i];
      }
      xMid.push(midVec);
    }

    // Stage 5: MLP + Residual 2
    const mlpHiddenDim = 4 * nEmbd;
    const mlpPreGelu: Float32Array[] = [];
    const mlpActivations: Float32Array[] = [];
    const mlpOut: Float32Array[] = [];
    const xFinal: Float32Array[] = [];

    for (let t = 0; t < T; t++) {
      const pre = new Float32Array(mlpHiddenDim);
      const post = new Float32Array(mlpHiddenDim);
      for (let i = 0; i < mlpHiddenDim; i++) {
        let s = this.b1[i];
        for (let j = 0; j < nEmbd; j++) {
          s += xMid[t][j] * this.w1[j * mlpHiddenDim + i];
        }
        pre[i] = s;
        post[i] = Math.max(0, s); // ReLU
      }
      mlpPreGelu.push(pre);
      mlpActivations.push(post);

      const out = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i++) {
        let s = this.b2[i];
        for (let j = 0; j < mlpHiddenDim; j++) {
          s += post[j] * this.w2[j * nEmbd + i];
        }
        out[i] = s;
      }
      mlpOut.push(out);

      const finalVec = new Float32Array(nEmbd);
      for (let i = 0; i < nEmbd; i++) {
        finalVec[i] = xMid[t][i] + out[i];
      }
      xFinal.push(finalVec);
    }

    // Stage 6: LM Head Logits for the final token
    const lastT = T - 1;
    const lastLogits = new Float32Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      let s = 0;
      for (let d = 0; d < nEmbd; d++) {
        s += xFinal[lastT][d] * this.lmHead[d * vocabSize + v];
      }
      lastLogits[v] = s;
    }

    const lastProbs = softmaxInPlace(lastLogits);

    // Calculate Shannon entropy: - sum(p * log2(p))
    let entropy = 0;
    for (let v = 0; v < vocabSize; v++) {
      const p = lastProbs[v];
      if (p > 1e-12) {
        entropy -= p * Math.log2(p);
      }
    }

    // Top 10 candidates
    const indexed = Array.from(lastLogits).map((logit, idx) => ({
      idx,
      logit,
      prob: lastProbs[idx]
    }));
    indexed.sort((a, b) => b.prob - a.prob);
    const topCandidates = indexed.slice(0, 10);

    return {
      tokens,
      xTokens,
      xPositions,
      xCombined,
      Q,
      K,
      V,
      rawAttentionScores,
      softmaxProbabilities,
      weightedVPerHead,
      attnProj,
      xMid,
      mlpPreGelu,
      mlpActivations,
      mlpOut,
      xFinal,
      lastLogits,
      lastProbs,
      entropy,
      topCandidates
    };
  }

  // Reset weights to random initial state
  resetWeights() {
    const { vocabSize, blockSize, nEmbd } = this.config;
    this.wte = randomMatrix(vocabSize, nEmbd, 0.08);
    this.wpe = randomMatrix(blockSize, nEmbd, 0.04);
    this.wq = randomMatrix(nEmbd, nEmbd, Math.sqrt(2.0 / nEmbd));
    this.wk = randomMatrix(nEmbd, nEmbd, Math.sqrt(2.0 / nEmbd));
    this.wv = randomMatrix(nEmbd, nEmbd, Math.sqrt(2.0 / nEmbd));
    this.wo = randomMatrix(nEmbd, nEmbd, Math.sqrt(2.0 / nEmbd));
    const mlpHidden = 4 * nEmbd;
    this.w1 = randomMatrix(nEmbd, mlpHidden, Math.sqrt(2.0 / nEmbd));
    this.b1 = zeros(mlpHidden);
    this.w2 = randomMatrix(mlpHidden, nEmbd, Math.sqrt(2.0 / mlpHidden));
    this.b2 = zeros(nEmbd);
    this.lmHead = randomMatrix(nEmbd, vocabSize, 0.08);
    this.initAdam();
  }
}
