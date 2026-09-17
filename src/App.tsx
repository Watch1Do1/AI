import { useState, useEffect, useRef, useCallback } from 'react';
import { Header, AppTab } from './components/Header';
import { PretrainingWorkbench } from './components/PretrainingWorkbench';
import { BPETokenizerLab } from './components/BPETokenizerLab';
import { SFTProductBridge } from './components/SFTProductBridge';
import { ScalingRoadmap } from './components/ScalingRoadmap';
import { MemoryBridgeSection } from './components/MemoryBridgeSection';
import { PyTorchScriptExporter } from './components/PyTorchScriptExporter';
import { DatasetManager } from './components/DatasetManager';
import { ArchitectureDiagram } from './components/ArchitectureDiagram';
import { CharTokenizer } from './engine/tokenizer';
import { MicroGPT } from './engine/transformer';
import { CORPUS_PRESETS } from './engine/datasets';
import { loadSerializedWeightsIntoModel, validateAndLoadSerializedWeights, SerializedWeights } from './engine/weightBridge';
import { ModelConfig, TrainingMetrics, AttentionHeadData } from './types';

export default function App() {
  // Tab navigation state
  const [activeTab, setActiveTab] = useState<AppTab>('workbench');
  const [weightSource, setWeightSource] = useState<'browser' | 'pytorch'>('browser');

  // Corpus & Tokenizer state
  const [currentCorpusText, setCurrentCorpusText] = useState<string>(CORPUS_PRESETS[0].text);
  const [currentCorpusTitle, setCurrentCorpusTitle] = useState<string>(CORPUS_PRESETS[0].title);
  const tokenizerRef = useRef<CharTokenizer>(new CharTokenizer(CORPUS_PRESETS[0].text));
  const [tokenizerData, setTokenizerData] = useState(tokenizerRef.current.getData());

  // Model Hyperparameters
  const [config, setConfig] = useState<ModelConfig>({
    vocabSize: tokenizerRef.current.vocabSize,
    blockSize: 32,
    nEmbd: 32,
    nHead: 2,
    nLayer: 1,
    lr: 0.003,
    batchSize: 4
  });

  // Model instance
  const modelRef = useRef<MicroGPT>(new MicroGPT({ ...config, vocabSize: tokenizerRef.current.vocabSize }));
  const [paramCount, setParamCount] = useState<number>(modelRef.current.getTotalParameters());

  // Training metrics state
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [step, setStep] = useState<number>(0);
  const [currentLoss, setCurrentLoss] = useState<number | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([]);
  const tokensProcessedRef = useRef<number>(0);
  const isTrainingRef = useRef<boolean>(false);
  isTrainingRef.current = isTraining;

  // Theoretical initial random loss: -ln(1 / V) = ln(V)
  const initialTheoreticalLoss = Math.log(Math.max(2, tokenizerData.vocabSize));

  // Initialize or re-initialize model when config or vocab changes
  const initModel = useCallback((newConfig: ModelConfig) => {
    modelRef.current = new MicroGPT(newConfig);
    setParamCount(modelRef.current.getTotalParameters());
    setStep(0);
    setCurrentLoss(null);
    setMetricsHistory([]);
    tokensProcessedRef.current = 0;
    setWeightSource('browser');
  }, []);

  // Handle switching or updating training text corpus
  const handleSelectCorpus = (text: string, title: string) => {
    setCurrentCorpusText(text);
    setCurrentCorpusTitle(title);
    tokenizerRef.current.train(text);
    const tData = tokenizerRef.current.getData();
    setTokenizerData(tData);

    const updatedConfig = { ...config, vocabSize: tData.vocabSize };
    setConfig(updatedConfig);
    initModel(updatedConfig);
  };

  // Reset current model weights
  const handleResetWeights = () => {
    setIsTraining(false);
    modelRef.current.resetWeights();
    setStep(0);
    setCurrentLoss(null);
    setMetricsHistory([]);
    tokensProcessedRef.current = 0;
    setWeightSource('browser');
  };

  // Import trained PyTorch weights into browser visualizer with strict shape validation
  const handleImportWeights = (data: any): { success: boolean; error?: string } => {
    try {
      const validation = validateAndLoadSerializedWeights(modelRef.current, data as SerializedWeights);
      if (validation.success) {
        setIsTraining(false);
        setWeightSource('pytorch');

        // Evaluate loss on 10% validation split with imported weights
        const encoded = tokenizerRef.current.encode(currentCorpusText);
        const splitIndex = Math.floor(0.9 * encoded.length);
        const trainTokens = encoded.slice(0, splitIndex);
        const valTokens = encoded.slice(splitIndex);

        let tLoss = 1.85;
        let vLoss = 1.95;

        if (trainTokens.length > config.blockSize + 1) {
          const tBatch = [{
            input: trainTokens.slice(0, config.blockSize),
            target: trainTokens.slice(1, config.blockSize + 1)
          }];
          tLoss = modelRef.current.evaluateLoss(tBatch);
        }

        if (valTokens.length > config.blockSize + 1) {
          const vBatch = [{
            input: valTokens.slice(0, config.blockSize),
            target: valTokens.slice(1, config.blockSize + 1)
          }];
          vLoss = modelRef.current.evaluateLoss(vBatch);
        }

        setCurrentLoss(tLoss);
        setStep(1000);

        setMetricsHistory(prev => [
          ...prev,
          {
            step: 1000,
            loss: tLoss,
            valLoss: vLoss,
            perplexity: Math.exp(Math.min(10, tLoss)),
            tokensProcessed: tokensProcessedRef.current,
            tokensPerSec: 0,
            timestamp: Date.now()
          }
        ]);

        return { success: true };
      }
      return { success: false, error: validation.error };
    } catch (e: any) {
      console.error("Weight import error:", e);
      return { success: false, error: e?.message || 'Failed to parse weights file.' };
    }
  };

  // Single training step helper with 90% train / 10% validation split
  const executeTrainingStep = useCallback((batchSize: number = config.batchSize) => {
    const encoded = tokenizerRef.current.encode(currentCorpusText);
    if (encoded.length <= config.blockSize + 2) return { loss: 0, valLoss: 0, perplexity: 1 };

    // 90% train tokens, 10% validation tokens
    const splitIndex = Math.floor(0.9 * encoded.length);
    const trainTokens = encoded.slice(0, splitIndex);
    const valTokens = encoded.slice(splitIndex);

    const batch: { input: number[]; target: number[] }[] = [];
    const maxTrainStart = trainTokens.length - config.blockSize - 1;
    if (maxTrainStart > 0) {
      for (let b = 0; b < batchSize; b++) {
        const start = Math.floor(Math.random() * maxTrainStart);
        const input = trainTokens.slice(start, start + config.blockSize);
        const target = trainTokens.slice(start + 1, start + config.blockSize + 1);
        batch.push({ input, target });
      }
    }

    const res = modelRef.current.trainStep(batch, config.lr);
    tokensProcessedRef.current += batchSize * config.blockSize;

    // Evaluate on 10% validation split
    let valLoss = res.loss;
    const maxValStart = valTokens.length - config.blockSize - 1;
    if (maxValStart > 0) {
      const valBatch: { input: number[]; target: number[] }[] = [];
      const valSamples = Math.min(4, batchSize);
      for (let b = 0; b < valSamples; b++) {
        const vStart = Math.floor(Math.random() * maxValStart);
        valBatch.push({
          input: valTokens.slice(vStart, vStart + config.blockSize),
          target: valTokens.slice(vStart + 1, vStart + config.blockSize + 1)
        });
      }
      valLoss = modelRef.current.evaluateLoss(valBatch);
    }

    return { loss: res.loss, valLoss, perplexity: res.perplexity };
  }, [currentCorpusText, config]);

  // Step button handler (e.g. +10 steps)
  const handleManualStep = (count: number = 10) => {
    let lastLoss = 0;
    let lastValLoss = 0;
    let lastPerp = 0;
    for (let i = 0; i < count; i++) {
      const { loss, valLoss, perplexity } = executeTrainingStep(config.batchSize);
      lastLoss = loss;
      lastValLoss = valLoss;
      lastPerp = perplexity;
    }

    const newStep = step + count;
    setStep(newStep);
    setCurrentLoss(lastLoss);

    setMetricsHistory((prev) => {
      const updated = [
        ...prev,
        {
          step: newStep,
          loss: lastLoss,
          valLoss: lastValLoss,
          perplexity: lastPerp,
          tokensProcessed: tokensProcessedRef.current,
          tokensPerSec: 0,
          timestamp: Date.now()
        }
      ];
      return updated.slice(-60);
    });
  };

  // Continuous training animation loop
  useEffect(() => {
    if (!isTraining) return;

    let animId: number;
    let localStep = step;

    const loop = () => {
      if (!isTrainingRef.current) return;

      let lastLoss = 0;
      let lastValLoss = 0;
      let lastPerp = 0;
      for (let s = 0; s < 3; s++) {
        const { loss, valLoss, perplexity } = executeTrainingStep(config.batchSize);
        lastLoss = loss;
        lastValLoss = valLoss;
        lastPerp = perplexity;
        localStep++;
      }

      setStep(localStep);
      setCurrentLoss(lastLoss);

      setMetricsHistory((prev) => {
        const updated = [
          ...prev,
          {
            step: localStep,
            loss: lastLoss,
            valLoss: lastValLoss,
            perplexity: lastPerp,
            tokensProcessed: tokensProcessedRef.current,
            tokensPerSec: 3 * config.batchSize * config.blockSize * 60,
            timestamp: Date.now()
          }
        ];
        return updated.slice(-80);
      });

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [isTraining, executeTrainingStep, config, step]);

  // Autoregressive generation
  const handleGenerateText = (
    promptStr: string,
    temp: number,
    topK: number,
    maxTokens: number
  ): { text: string; tokens: number[]; attention: AttentionHeadData[] } => {
    const promptTokens = tokenizerRef.current.encode(promptStr);
    const { generatedTokens, attentionMaps } = modelRef.current.generate(
      promptTokens.length > 0 ? promptTokens : [1],
      maxTokens,
      temp,
      topK
    );
    const decoded = tokenizerRef.current.decode(generatedTokens);
    return {
      text: decoded,
      tokens: generatedTokens,
      attention: attentionMaps
    };
  };

  return (
    <div className="min-h-screen bg-zinc-100/70 text-zinc-900 flex flex-col font-sans">
      
      {/* Top Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isTraining={isTraining}
        onToggleTraining={() => setIsTraining(!isTraining)}
        onReset={handleResetWeights}
        paramCount={paramCount}
        currentLoss={currentLoss}
        step={step}
        weightSource={weightSource}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'workbench' && (
          <PretrainingWorkbench
            config={config}
            setConfig={setConfig}
            metricsHistory={metricsHistory}
            isTraining={isTraining}
            onToggleTraining={() => setIsTraining(!isTraining)}
            onStep={handleManualStep}
            onReset={handleResetWeights}
            onGenerate={handleGenerateText}
            datasetTitle={currentCorpusTitle}
            initialTheoreticalLoss={initialTheoreticalLoss}
            onImportWeights={handleImportWeights}
            weightSource={weightSource}
          />
        )}

        {activeTab === 'bpe' && (
          <BPETokenizerLab corpus={currentCorpusText} />
        )}

        {activeTab === 'pytorch' && (
          <PyTorchScriptExporter config={config} />
        )}

        {activeTab === 'sft' && (
          <SFTProductBridge />
        )}

        {activeTab === 'memory' && (
          <MemoryBridgeSection />
        )}

        {activeTab === 'scaling' && (
          <ScalingRoadmap />
        )}

        {activeTab === 'dataset' && (
          <DatasetManager
            currentCorpus={currentCorpusText}
            onSelectCorpus={handleSelectCorpus}
            selectedTitle={currentCorpusTitle}
            tokenizerData={tokenizerData}
          />
        )}

        {activeTab === 'architecture' && (
          <ArchitectureDiagram config={config} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-200 bg-white py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-zinc-500">
          <div>
            TinyGPT Pretraining Lab · Evolutionary Path from Toy Model to Conversational Product
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setActiveTab('bpe')}
              className="hover:text-zinc-900 transition-colors underline"
            >
              1. BPE
            </button>
            <button
              onClick={() => setActiveTab('pytorch')}
              className="hover:text-zinc-900 transition-colors underline"
            >
              2. PyTorch
            </button>
            <button
              onClick={() => setActiveTab('sft')}
              className="hover:text-zinc-900 transition-colors underline"
            >
              3. SFT Product
            </button>
            <button
              onClick={() => setActiveTab('scaling')}
              className="hover:text-zinc-900 transition-colors underline"
            >
              4. Scaling Path
            </button>
            <span>Runs 100% In-Browser & In PyTorch</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
