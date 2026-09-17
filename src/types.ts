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
