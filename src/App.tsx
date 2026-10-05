import { useState, useEffect, useRef, useCallback } from 'react';
import { Header, AppTab } from './components/Header';
import { PretrainingWorkbench } from './components/PretrainingWorkbench';
import { StepThroughDebugger } from './components/StepThroughDebugger';
import { GPT2Comparison } from './components/GPT2Comparison';
import { GradientActivationVisualizer } from './components/GradientActivationVisualizer';
import { CheckpointTimeline } from './components/CheckpointTimeline';
import { ExplainabilityView } from './components/ExplainabilityView';
import { BrowserVsPyTorchView } from './components/BrowserVsPyTorchView';
import { WhyThisWorks } from './components/WhyThisWorks';
import { TrainWizardModal } from './components/TrainWizardModal';
import { PlaygroundTourModal } from './components/PlaygroundTourModal';
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
import { rebuildMicroGPTFromSerializedWeights } from './engine/weightBridge';
import { ModelConfig, TrainingMetrics, AttentionHeadData, CheckpointSnapshot } from './types';

export default function App() {
  // Tab navigation state
  const [activeTab, setActiveTab] = useState<AppTab>('workbench');
  const [weightSource, setWeightSource] = useState<'browser' | 'pytorch'>('browser');

  // Modal dialog states
  const [isTrainWizardOpen, setIsTrainWizardOpen] = useState<boolean>(false);
  const [isTourOpen, setIsTourOpen] = useState<boolean>(false);

  // Corpus & Tokenizer state
  const [currentCorpusText, setCurrentCorpusText] = useState<string>(CORPUS_PRESETS[0].text);
  const [currentCorpusTitle, setCurrentCorpusTitle] = useState<string>(CORPUS_PRESETS[0].title);
  const tokenizerRef = useRef<CharTokenizer>(new CharTokenizer(CORPUS_PRESETS[0].text));
  const [tokenizerData, setTokenizerData] = useState(tokenizerRef.current.getData());

  // Visible confirmation state after Apply to Model
  const [applyConfirmation, setApplyConfirmation] = useState<{
    activeTitle: string;
    vocabSize: number;
    paramCount: number;
  } | null>(null);

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

  // Checkpoint snapshots for timeline scrubbing
  const [snapshots, setSnapshots] = useState<CheckpointSnapshot[]>([]);

  // Generate text helper using active model
  const handleGenerateText = useCallback((
    prompt: string,
    temp: number = 0.7,
    topK: number = 8,
    maxTokens: number = 70
  ): { text: string; tokens: number[]; attention: AttentionHeadData[] } => {
    const promptTokens = tokenizerRef.current.encode(prompt.slice(0, config.blockSize));
    const safePromptTokens = promptTokens.length > 0 ? promptTokens : [0];
    const gen = modelRef.current.generate(safePromptTokens, maxTokens, temp, topK);
    const fullText = tokenizerRef.current.decode(gen.generatedTokens);
    return {
      text: fullText,
      tokens: gen.generatedTokens,
      attention: gen.attentionMaps || []
    };
  }, [config.blockSize]);

  // Take snapshot of weights and output for timeline
  const takeSnapshot = useCallback(() => {
    const samplePrompt = currentCorpusText.slice(0, 15) || 'INTERVIEWER:';
    const gen = handleGenerateText(samplePrompt, 0.7, 8, 45);
    const snap: CheckpointSnapshot = {
      step,
      timestamp: Date.now(),
      loss: currentLoss ?? initialTheoreticalLoss,
      valLoss: metricsHistory[metricsHistory.length - 1]?.valLoss ?? (currentLoss ?? initialTheoreticalLoss),
      perplexity: Math.exp(Math.min(10, currentLoss ?? initialTheoreticalLoss)),
      samplePrompt,
      sampleText: gen.text,
      attentionMaps: gen.attention,
      weightsData: modelRef.current.cloneWeights()
    };
    setSnapshots(prev => [...prev.filter(s => s.step !== step), snap].sort((a, b) => a.step - b.step));
  }, [step, currentLoss, initialTheoreticalLoss, metricsHistory, currentCorpusText, handleGenerateText]);

  // Restore weights from a snapshot
  const restoreSnapshot = useCallback((snap: CheckpointSnapshot) => {
    if (snap.weightsData) {
      modelRef.current.loadWeights(snap.weightsData);
      setStep(snap.step);
      setCurrentLoss(snap.loss);
      setWeightSource('browser');
    }
  }, []);

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
    modelRef.current = new MicroGPT(updatedConfig);
    const pCount = modelRef.current.getTotalParameters();
    setParamCount(pCount);
    setStep(0);
    setCurrentLoss(null);
    setMetricsHistory([]);
    tokensProcessedRef.current = 0;
    setWeightSource('browser');

    setApplyConfirmation({
      activeTitle: title,
      vocabSize: tData.vocabSize,
      paramCount: pCount
    });
  };

  // Reset current model weights
  const handleResetWeights = () => {
    modelRef.current.resetWeights();
    setStep(0);
    setCurrentLoss(null);
    setMetricsHistory([]);
    tokensProcessedRef.current = 0;
    setWeightSource('browser');
    takeSnapshot();
  };

  // Import trained weights from JSON exported by python/train_tiny_gpt.py
  const handleImportWeights = (data: any, customTitle?: string) => {
    try {
      const rebuildResult = rebuildMicroGPTFromSerializedWeights(data, config);
      if (!rebuildResult.success || !rebuildResult.model || !rebuildResult.config) {
        return {
          success: false,
          error: rebuildResult.error || 'Failed to rebuild model from checkpoint configuration.'
        };
      }

      setIsTraining(false);
      const newModel = rebuildResult.model;
      const newConfig = rebuildResult.config;
      modelRef.current = newModel;
      setConfig(newConfig);

      if (rebuildResult.vocab && rebuildResult.vocab.length > 0) {
        tokenizerRef.current.setVocab(rebuildResult.vocab);
      }
      const tData = tokenizerRef.current.getData();
      setTokenizerData(tData);

      const newParamCount = newModel.getTotalParameters();
      setParamCount(newParamCount);
      setWeightSource('pytorch');

      const title =
        customTitle ||
        data.title ||
        (newConfig.vocabSize === 65
          ? 'Tiny Shakespeare (PyTorch Checkpoint)'
          : `PyTorch Checkpoint (V=${newConfig.vocabSize})`);
      setCurrentCorpusTitle(title);

      if (newConfig.vocabSize === 65) {
        const shkPreset = CORPUS_PRESETS.find(p => p.id === 'tiny_shakespeare');
        if (shkPreset) {
          setCurrentCorpusText(shkPreset.text);
        }
      }

      const encoded = tokenizerRef.current.encode(currentCorpusText);
      const splitIndex = Math.floor(0.9 * encoded.length);
      const trainTokens = encoded.slice(0, splitIndex);
      const valTokens = encoded.slice(splitIndex);

      let tLoss = 1.85;
      let vLoss = 1.95;

      if (trainTokens.length > newConfig.blockSize + 1) {
        const tBatch = [{
          input: trainTokens.slice(0, newConfig.blockSize),
          target: trainTokens.slice(1, newConfig.blockSize + 1)
        }];
        tLoss = newModel.evaluateLoss(tBatch);
      }

      if (valTokens.length > newConfig.blockSize + 1) {
        const vBatch = [{
          input: valTokens.slice(0, newConfig.blockSize),
          target: valTokens.slice(1, newConfig.blockSize + 1)
        }];
        vLoss = newModel.evaluateLoss(vBatch);
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

      const conf = {
        activeTitle: title,
        vocabSize: newConfig.vocabSize,
        paramCount: newParamCount
      };
      setApplyConfirmation(conf);

      return {
        success: true,
        activeTitle: title,
        vocabSize: newConfig.vocabSize,
        paramCount: newParamCount
      };
    } catch (e: any) {
      console.error("Weight import error:", e);
      return { success: false, error: e?.message || 'Failed to parse weights file.' };
    }
  };

  // Single training step helper with 90% train / 10% validation split
  const executeTrainingStep = useCallback((batchSize: number = config.batchSize) => {
    const encoded = tokenizerRef.current.encode(currentCorpusText);
    if (encoded.length <= config.blockSize + 2) return { loss: 0, valLoss: 0, perplexity: 1 };

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

    // Auto-capture timeline snapshot at key milestones
    if (newStep === 10 || newStep === 50 || newStep === 100 || newStep === 250 || newStep === 500) {
      takeSnapshot();
    }
  };

  // Continuous training animation loop
  useEffect(() => {
    if (!isTraining) return;

    let animId: number;
    let localStep = step;

    const loop = () => {
      if (!isTrainingRef.current) return;

      const subSteps = 5;
      let lastLoss = 0;
      let lastValLoss = 0;
      let lastPerp = 0;
      for (let i = 0; i < subSteps; i++) {
        const { loss, valLoss, perplexity } = executeTrainingStep(config.batchSize);
        lastLoss = loss;
        lastValLoss = valLoss;
        lastPerp = perplexity;
      }

      localStep += subSteps;
      setStep(localStep);
      setCurrentLoss(lastLoss);

      setMetricsHistory((prev) => {
        const next = [
          ...prev,
          {
            step: localStep,
            loss: lastLoss,
            valLoss: lastValLoss,
            perplexity: lastPerp,
            tokensProcessed: tokensProcessedRef.current,
            tokensPerSec: 0,
            timestamp: Date.now()
          }
        ];
        return next.slice(-60);
      });

      // Capture milestone snapshots during training
      if (localStep % 100 === 0 && localStep <= 1000) {
        takeSnapshot();
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [isTraining, executeTrainingStep, config.batchSize, step, takeSnapshot]);

  // Initial baseline snapshot on startup
  useEffect(() => {
    if (snapshots.length === 0) {
      takeSnapshot();
    }
  }, [snapshots.length, takeSnapshot]);

  return (
    <div className="min-h-screen bg-zinc-100 flex flex-col font-sans text-zinc-900 selection:bg-zinc-800 selection:text-white">
      
      {/* Global Header */}
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
        onOpenWizard={() => setIsTrainWizardOpen(true)}
        onOpenTour={() => setIsTourOpen(true)}
      />

      {/* Main Tab Routing */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        
        {/* Visualizer Workbench */}
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
            applyConfirmation={applyConfirmation}
            onOpenWizard={() => setIsTrainWizardOpen(true)}
            onSelectTab={(tab) => setActiveTab(tab)}
          />
        )}

        {/* 1. Step Through a Transformer Debugger */}
        {activeTab === 'step-debugger' && (
          <StepThroughDebugger
            model={modelRef.current}
            tokenizer={tokenizerRef.current}
            config={config}
          />
        )}

        {/* 2. Compare Against GPT-2 */}
        {activeTab === 'compare-gpt2' && (
          <GPT2Comparison
            model={modelRef.current}
            tokenizer={tokenizerRef.current}
          />
        )}

        {/* 3. Gradient & Activation Visualizer */}
        {activeTab === 'gradient-viz' && (
          <GradientActivationVisualizer
            model={modelRef.current}
            telemetry={modelRef.current.latestTelemetry}
            step={step}
          />
        )}

        {/* 4. Checkpoint Timeline */}
        {activeTab === 'checkpoints' && (
          <CheckpointTimeline
            model={modelRef.current}
            tokenizer={tokenizerRef.current}
            snapshots={snapshots}
            onTakeSnapshot={takeSnapshot}
            onRestoreSnapshot={restoreSnapshot}
            currentStep={step}
          />
        )}

        {/* 5. Explainability Mode */}
        {activeTab === 'explainability' && (
          <ExplainabilityView
            model={modelRef.current}
            tokenizer={tokenizerRef.current}
          />
        )}

        {/* 6. Browser vs PyTorch Side-by-Side */}
        {activeTab === 'browser-vs-pytorch' && (
          <BrowserVsPyTorchView
            model={modelRef.current}
            tokenizer={tokenizerRef.current}
            onOpenImportModal={() => setActiveTab('workbench')}
            weightSource={weightSource}
          />
        )}

        {/* 7. Why This Works Unified Guide */}
        {activeTab === 'why-this-works' && (
          <WhyThisWorks />
        )}

        {/* BPE Tokenizer Lab */}
        {activeTab === 'bpe' && (
          <BPETokenizerLab corpus={currentCorpusText} />
        )}

        {/* PyTorch Script Exporter */}
        {activeTab === 'pytorch' && (
          <PyTorchScriptExporter config={config} />
        )}

        {/* SFT LoRA Product Bridge */}
        {activeTab === 'sft' && (
          <SFTProductBridge />
        )}

        {/* Memory Bridge */}
        {activeTab === 'memory' && (
          <MemoryBridgeSection />
        )}

        {/* Scaling Roadmap */}
        {activeTab === 'scaling' && (
          <ScalingRoadmap />
        )}

        {/* Corpus Manager */}
        {activeTab === 'dataset' && (
          <DatasetManager
            currentCorpus={currentCorpusText}
            onSelectCorpus={handleSelectCorpus}
            selectedTitle={currentCorpusTitle}
            tokenizerData={tokenizerData}
            applyConfirmation={applyConfirmation}
            paramCount={paramCount}
          />
        )}

        {/* Architecture Diagram */}
        {activeTab === 'architecture' && (
          <ArchitectureDiagram config={config} />
        )}

      </main>

      {/* Global Modals */}
      <TrainWizardModal
        isOpen={isTrainWizardOpen}
        onClose={() => setIsTrainWizardOpen(false)}
        config={config}
        onUpdateConfig={setConfig}
        onSelectCorpus={handleSelectCorpus}
        currentCorpusTitle={currentCorpusTitle}
        onRunSteps={handleManualStep}
        onGenerate={(p, t, k, m) => handleGenerateText(p, t, k, m)}
        currentLoss={currentLoss}
        step={step}
      />

      <PlaygroundTourModal
        isOpen={isTourOpen}
        onClose={() => setIsTourOpen(false)}
        onSelectTab={(tab) => setActiveTab(tab)}
        onOpenWizard={() => setIsTrainWizardOpen(true)}
      />

      {/* Footer */}
      <footer className="border-t border-zinc-200 bg-white py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-zinc-500">
          <div>
            TinyGPT Pretraining Lab · Minimal Causal Decoder-Only Transformer
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setActiveTab('step-debugger')}
              className="hover:text-zinc-900 transition-colors underline"
            >
              Step Debugger
            </button>
            <button
              onClick={() => setActiveTab('compare-gpt2')}
              className="hover:text-zinc-900 transition-colors underline"
            >
              vs. GPT-2
            </button>
            <button
              onClick={() => setActiveTab('gradient-viz')}
              className="hover:text-zinc-900 transition-colors underline"
            >
              Gradients
            </button>
            <button
              onClick={() => setActiveTab('why-this-works')}
              className="hover:text-zinc-900 transition-colors underline"
            >
              Why This Works
            </button>
            <span>Runs 100% In-Browser & In PyTorch</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
