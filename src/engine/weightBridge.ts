import { MicroGPT } from './transformer';
import { ModelConfig } from '../types';

export interface SerializedWeights {
  config: {
    vocabSize?: number;
    vocab_size?: number;
    blockSize?: number;
    block_size?: number;
    nEmbd?: number;
    n_embd?: number;
    nHead?: number;
    n_head?: number;
    nLayer?: number;
    n_layer?: number;
  };
  vocab?: string[];
  title?: string;
  weights: {
    wte: number[];
    wpe: number[];
    wq: number[];
    wk: number[];
    wv: number[];
    wo: number[];
    w1: number[];
    b1: number[];
    w2: number[];
    b2: number[];
    lmHead: number[];
  };
}

export function generateWeightExportPythonCode(): string {
  return `import json
import torch

def export_model_for_browser_visualizer(model, vocab, config, output_path="tiny_gpt_weights.json"):
    """
    Exports a trained PyTorch TinyGPT checkpoint into a format that can be
    dragged-and-dropped directly into the browser visualizer.
    """
    state = model.state_dict()
    
    export_dict = {
        "config": {
            "vocabSize": len(vocab),
            "blockSize": config["block_size"],
            "nEmbd": config["n_embd"],
            "nHead": config["n_head"],
            "nLayer": config.get("n_layer", 1)
        },
        "vocab": vocab,
        "weights": {
            "wte": state["wte.weight"].detach().cpu().numpy().flatten().tolist(),
            "wpe": state["wpe.weight"].detach().cpu().numpy().flatten().tolist(),
            "wq": state["block.attn.q_proj.weight"].detach().cpu().numpy().flatten().tolist(),
            "wk": state["block.attn.k_proj.weight"].detach().cpu().numpy().flatten().tolist(),
            "wv": state["block.attn.v_proj.weight"].detach().cpu().numpy().flatten().tolist(),
            "wo": state["block.attn.out_proj.weight"].detach().cpu().numpy().flatten().tolist(),
            "w1": state["block.mlp.c_fc.weight"].detach().cpu().t().numpy().flatten().tolist(),
            "b1": state["block.mlp.c_fc.bias"].detach().cpu().numpy().flatten().tolist(),
            "w2": state["block.mlp.c_proj.weight"].detach().cpu().t().numpy().flatten().tolist(),
            "b2": state["block.mlp.c_proj.bias"].detach().cpu().numpy().flatten().tolist(),
            "lmHead": state["lm_head.weight"].detach().cpu().t().numpy().flatten().tolist()
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(export_dict, f)
    print(f"--> Saved browser-compatible checkpoint to: {output_path}")

# Run after training:
# checkpoint = torch.load("tiny_gpt_weights.pt")
# export_model_for_browser_visualizer(model, checkpoint["vocab"], checkpoint["config"])
`;
}

export interface RebuildModelResult {
  success: boolean;
  error?: string;
  model?: MicroGPT;
  config?: ModelConfig;
  vocab?: string[];
  vocabSize?: number;
  paramCount?: number;
  totalElements?: number;
}

export function rebuildMicroGPTFromSerializedWeights(
  data: SerializedWeights | any,
  fallbackConfig?: Partial<ModelConfig>
): RebuildModelResult {
  if (!data || typeof data !== 'object') {
    return { success: false, error: 'Invalid file format: JSON root must be an object.' };
  }

  if (!data.weights || typeof data.weights !== 'object') {
    return { success: false, error: 'Invalid file format: missing "weights" dictionary.' };
  }

  const weights = data.weights;
  const cfg = data.config || {};

  const nEmbd = Number(cfg.nEmbd ?? cfg.n_embd) || fallbackConfig?.nEmbd || 32;
  const blockSize = Number(cfg.blockSize ?? cfg.block_size) || fallbackConfig?.blockSize || 32;
  const nHead = Number(cfg.nHead ?? cfg.n_head) || fallbackConfig?.nHead || 2;

  // Extract vocabulary if provided in the export
  const vocab: string[] | undefined = Array.isArray(data.vocab) && data.vocab.length > 0 ? data.vocab : undefined;

  // Determine vocabSize from vocab array, config, or wte length / nEmbd
  let vocabSize = vocab ? vocab.length : Number(cfg.vocabSize ?? cfg.vocab_size);
  if (!vocabSize && weights.wte && Array.isArray(weights.wte) && nEmbd > 0) {
    vocabSize = Math.round(weights.wte.length / nEmbd);
  }

  if (!vocabSize || vocabSize <= 0) {
    return { success: false, error: 'Could not determine vocabulary size from weights file.' };
  }

  // Accept wte if length is vocabSize * nEmbd
  if (!weights.wte || !Array.isArray(weights.wte)) {
    return { success: false, error: 'Missing token embedding weights "wte".' };
  }

  const expectedWteLen = vocabSize * nEmbd;
  if (weights.wte.length !== expectedWteLen) {
    if (weights.wte.length % nEmbd === 0) {
      vocabSize = weights.wte.length / nEmbd;
    } else {
      return {
        success: false,
        error: `wte length (${weights.wte.length}) does not match vocabSize × nEmbd (${vocabSize} × ${nEmbd} = ${expectedWteLen}).`
      };
    }
  }

  const mlpHidden = 4 * nEmbd;
  const tensorDefs: Array<{
    name: keyof SerializedWeights['weights'];
    expectedLength: number;
    shapeDesc: string;
  }> = [
    { name: 'wte', expectedLength: vocabSize * nEmbd, shapeDesc: `(${vocabSize}, ${nEmbd})` },
    { name: 'wpe', expectedLength: blockSize * nEmbd, shapeDesc: `(${blockSize}, ${nEmbd})` },
    { name: 'wq', expectedLength: nEmbd * nEmbd, shapeDesc: `(${nEmbd}, ${nEmbd})` },
    { name: 'wk', expectedLength: nEmbd * nEmbd, shapeDesc: `(${nEmbd}, ${nEmbd})` },
    { name: 'wv', expectedLength: nEmbd * nEmbd, shapeDesc: `(${nEmbd}, ${nEmbd})` },
    { name: 'wo', expectedLength: nEmbd * nEmbd, shapeDesc: `(${nEmbd}, ${nEmbd})` },
    { name: 'w1', expectedLength: nEmbd * mlpHidden, shapeDesc: `(${nEmbd}, ${mlpHidden})` },
    { name: 'b1', expectedLength: mlpHidden, shapeDesc: `(${mlpHidden},)` },
    { name: 'w2', expectedLength: mlpHidden * nEmbd, shapeDesc: `(${mlpHidden}, ${nEmbd})` },
    { name: 'b2', expectedLength: nEmbd, shapeDesc: `(${nEmbd},)` },
    { name: 'lmHead', expectedLength: nEmbd * vocabSize, shapeDesc: `(${nEmbd}, ${vocabSize})` },
  ];

  let totalElements = 0;
  for (const def of tensorDefs) {
    const arr = weights[def.name];
    if (!arr || !Array.isArray(arr)) {
      return {
        success: false,
        error: `Import rejected: Missing or invalid tensor "${def.name}". Required shape: ${def.shapeDesc}.`
      };
    }
    if (arr.length !== def.expectedLength) {
      return {
        success: false,
        error: `Import rejected: Tensor shape mismatch for "${def.name}". Expected length ${def.expectedLength} ${def.shapeDesc}, but received ${arr.length}.`
      };
    }
    totalElements += arr.length;
  }

  // Build new MicroGPT instance with exact config from the file
  const newConfig: ModelConfig = {
    vocabSize,
    blockSize,
    nEmbd,
    nHead,
    nLayer: 1,
    lr: fallbackConfig?.lr || 0.001,
    batchSize: fallbackConfig?.batchSize || 8
  };

  const newModel = new MicroGPT(newConfig);

  // Atomically load validated arrays into model
  newModel.wte.set(weights.wte);
  newModel.wpe.set(weights.wpe);
  newModel.wq.set(weights.wq);
  newModel.wk.set(weights.wk);
  newModel.wv.set(weights.wv);
  newModel.wo.set(weights.wo);
  newModel.w1.set(weights.w1);
  newModel.b1.set(weights.b1);
  newModel.w2.set(weights.w2);
  newModel.b2.set(weights.b2);
  newModel.lmHead.set(weights.lmHead);

  return {
    success: true,
    model: newModel,
    config: newConfig,
    vocab,
    vocabSize,
    paramCount: newModel.getTotalParameters(),
    totalElements
  };
}

export interface WeightValidationResult {
  success: boolean;
  error?: string;
  tensorsValidated?: number;
  totalElements?: number;
}

export function validateAndLoadSerializedWeights(
  model: MicroGPT,
  data: SerializedWeights
): WeightValidationResult {
  const result = rebuildMicroGPTFromSerializedWeights(data, model.config);
  if (!result.success || !result.model) {
    return { success: false, error: result.error };
  }

  // If dimensions match current model instance, copy into it as well
  if (
    model.config.vocabSize === result.config?.vocabSize &&
    model.config.blockSize === result.config?.blockSize &&
    model.config.nEmbd === result.config?.nEmbd
  ) {
    model.wte.set(result.model.wte);
    model.wpe.set(result.model.wpe);
    model.wq.set(result.model.wq);
    model.wk.set(result.model.wk);
    model.wv.set(result.model.wv);
    model.wo.set(result.model.wo);
    model.w1.set(result.model.w1);
    model.b1.set(result.model.b1);
    model.w2.set(result.model.w2);
    model.b2.set(result.model.b2);
    model.lmHead.set(result.model.lmHead);
  }

  return {
    success: true,
    tensorsValidated: 11,
    totalElements: result.totalElements
  };
}

export function loadSerializedWeightsIntoModel(
  model: MicroGPT,
  data: SerializedWeights
): { success: boolean; error?: string } {
  return validateAndLoadSerializedWeights(model, data);
}
