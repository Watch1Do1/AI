import { MicroGPT } from './transformer';

export interface SerializedWeights {
  config: {
    vocabSize: number;
    blockSize: number;
    nEmbd: number;
    nHead: number;
    nLayer: number;
  };
  vocab: string[];
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
            "nLayer": config["n_layer"]
        },
        "vocab": vocab,
        "weights": {
            "wte": state["token_embedding_table.weight"].detach().cpu().numpy().flatten().tolist(),
            "wpe": state["position_embedding_table.weight"].detach().cpu().numpy().flatten().tolist(),
            "wq": state["blocks.0.sa.heads.0.query.weight"].detach().cpu().numpy().flatten().tolist() if "blocks.0.sa.heads.0.query.weight" in state else [],
            "wk": state["blocks.0.sa.heads.0.key.weight"].detach().cpu().numpy().flatten().tolist() if "blocks.0.sa.heads.0.key.weight" in state else [],
            "wv": state["blocks.0.sa.heads.0.value.weight"].detach().cpu().numpy().flatten().tolist() if "blocks.0.sa.heads.0.value.weight" in state else [],
            "wo": state["blocks.0.sa.proj.weight"].detach().cpu().numpy().flatten().tolist() if "blocks.0.sa.proj.weight" in state else [],
            "w1": state["blocks.0.ffwd.net.0.weight"].detach().cpu().numpy().flatten().tolist() if "blocks.0.ffwd.net.0.weight" in state else [],
            "b1": state["blocks.0.ffwd.net.0.bias"].detach().cpu().numpy().flatten().tolist() if "blocks.0.ffwd.net.0.bias" in state else [],
            "w2": state["blocks.0.ffwd.net.2.weight"].detach().cpu().numpy().flatten().tolist() if "blocks.0.ffwd.net.2.weight" in state else [],
            "b2": state["blocks.0.ffwd.net.2.bias"].detach().cpu().numpy().flatten().tolist() if "blocks.0.ffwd.net.2.bias" in state else [],
            "lmHead": state["lm_head.weight"].detach().cpu().numpy().flatten().tolist()
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
  if (!data || typeof data !== 'object') {
    return { success: false, error: 'Invalid file format: JSON root must be an object.' };
  }

  if (!data.weights || typeof data.weights !== 'object') {
    return { success: false, error: 'Invalid file format: missing "weights" dictionary.' };
  }

  const { vocabSize, blockSize, nEmbd } = model.config;
  const mlpHidden = 4 * nEmbd;

  // Exact tensor specifications and expected lengths
  const tensorDefs: Array<{
    name: keyof SerializedWeights['weights'];
    expectedLength: number;
    shapeDesc: string;
    target: Float32Array;
  }> = [
    { name: 'wte', expectedLength: vocabSize * nEmbd, shapeDesc: `(${vocabSize}, ${nEmbd})`, target: model.wte },
    { name: 'wpe', expectedLength: blockSize * nEmbd, shapeDesc: `(${blockSize}, ${nEmbd})`, target: model.wpe },
    { name: 'wq', expectedLength: nEmbd * nEmbd, shapeDesc: `(${nEmbd}, ${nEmbd})`, target: model.wq },
    { name: 'wk', expectedLength: nEmbd * nEmbd, shapeDesc: `(${nEmbd}, ${nEmbd})`, target: model.wk },
    { name: 'wv', expectedLength: nEmbd * nEmbd, shapeDesc: `(${nEmbd}, ${nEmbd})`, target: model.wv },
    { name: 'wo', expectedLength: nEmbd * nEmbd, shapeDesc: `(${nEmbd}, ${nEmbd})`, target: model.wo },
    { name: 'w1', expectedLength: nEmbd * mlpHidden, shapeDesc: `(${nEmbd}, ${mlpHidden})`, target: model.w1 },
    { name: 'b1', expectedLength: mlpHidden, shapeDesc: `(${mlpHidden},)`, target: model.b1 },
    { name: 'w2', expectedLength: mlpHidden * nEmbd, shapeDesc: `(${mlpHidden}, ${nEmbd})`, target: model.w2 },
    { name: 'b2', expectedLength: nEmbd, shapeDesc: `(${nEmbd},)`, target: model.b2 },
    { name: 'lmHead', expectedLength: nEmbd * vocabSize, shapeDesc: `(${nEmbd}, ${vocabSize})`, target: model.lmHead },
  ];

  let totalElements = 0;

  // STRICT VALIDATION: Check every tensor before modifying model state
  for (const def of tensorDefs) {
    const arr = data.weights[def.name];
    if (!arr || !Array.isArray(arr)) {
      return {
        success: false,
        error: `Import rejected: Missing or invalid tensor "${def.name}". Required shape: ${def.shapeDesc} (length ${def.expectedLength}).`
      };
    }

    if (arr.length !== def.expectedLength) {
      return {
        success: false,
        error: `Import rejected: Tensor shape mismatch for "${def.name}". Expected length ${def.expectedLength} ${def.shapeDesc}, but received ${arr.length}. Ensure your PyTorch model config matches (vocabSize: ${vocabSize}, blockSize: ${blockSize}, nEmbd: ${nEmbd}).`
      };
    }

    totalElements += arr.length;
  }

  // All shapes strictly match -> atomically copy into model Float32Arrays
  for (const def of tensorDefs) {
    const arr = data.weights[def.name];
    def.target.set(arr);
  }

  return {
    success: true,
    tensorsValidated: tensorDefs.length,
    totalElements
  };
}

export function loadSerializedWeightsIntoModel(
  model: MicroGPT,
  data: SerializedWeights
): { success: boolean; error?: string } {
  return validateAndLoadSerializedWeights(model, data);
}
