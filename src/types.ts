export interface ModelConfig {
  vocabSize: number;
  blockSize: number; // context window
  nEmbd: number;     // embedding dimension
  nHead: number;     // number of attention heads
  nLayer: number;    // number of transformer blocks
  lr: number;        // learning rate
  batchSize: number; // training batch size
}

export interface TrainingMetrics {
  step: number;
  loss: number;
  valLoss?: number;
  perplexity: number;
  tokensProcessed: number;
  tokensPerSec: number;
  timestamp: number;
}

export interface TokenizerData {
  charToId: Record<string, number>;
  idToChar: Record<number, string>;
  vocabSize: number;
  chars: string[];
}

export interface CorpusPreset {
  id: string;
  title: string;
  category: 'interview' | 'literature' | 'code' | 'philosophy' | 'custom';
  description: string;
  text: string;
}

export interface AttentionHeadData {
  headIndex: number;
  matrix: number[][]; // [query_pos][key_pos]
}

export interface GenerationResult {
  prompt: string;
  generatedText: string;
  tokens: number[];
  attentionMaps?: AttentionHeadData[];
  logitsEntropy?: number[];
}

export interface MemoryRecord {
  id: string;
  category: string;
  key: string;
  value: string;
  timestamp: string;
}

export interface MemorySimulationResult {
  query: string;
  retrievedMemories: MemoryRecord[];
  injectedPrompt: string;
  baseModelOutput: string;
  augmentedOutput: string;
  explanation: string;
}

export interface CheckpointSnapshot {
  step: number;
  timestamp: number;
  loss: number;
  valLoss: number;
  perplexity: number;
  samplePrompt: string;
  sampleText: string;
  attentionMaps?: AttentionHeadData[];
  weightsData?: {
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
  };
}

export interface StepDebugState {
  tokens: string[];
  tokenIds: number[];
  selectedTokenIdx: number;
  stage: number; // 0 to 6
  stages: {
    title: string;
    subtitle: string;
    shape: string;
    formula: string;
    explanation: string;
    matrixName: string;
    data: number[][] | number[];
    labels?: { rows?: string[]; cols?: string[] };
    highlight?: { r: number; c: number };
  }[];
}

export interface GradientTelemetry {
  step: number;
  totalNorm: number;
  wteNorm: number;
  wpeNorm: number;
  w1Norm: number;
  w2Norm: number;
  lmHeadNorm: number;
  isVanishing: boolean;
  isExploding: boolean;
  activationStats: {
    name: string;
    mean: number;
    variance: number;
    min: number;
    max: number;
    bins: number[]; // 8-bin histogram
  }[];
  attentionLogitsHeatmap?: {
    headIndex: number;
    rawLogits: number[][];
    softmaxProbs: number[][];
  };
}

export interface ConfigPresetItem {
  id: string;
  name: string;
  badge: string;
  description: string;
  blockSize: number;
  nEmbd: number;
  nHead: number;
  lr: number;
  batchSize: number;
}
