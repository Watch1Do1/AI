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

export function loadSerializedWeightsIntoModel(model: MicroGPT, data: SerializedWeights): boolean {
  try {
    const w = data.weights;
    if (w.wte && w.wte.length === model.wte.length) model.wte.set(w.wte);
    if (w.wpe && w.wpe.length === model.wpe.length) model.wpe.set(w.wpe);
    if (w.wq && w.wq.length === model.wq.length) model.wq.set(w.wq);
    if (w.wk && w.wk.length === model.wk.length) model.wk.set(w.wk);
    if (w.wv && w.wv.length === model.wv.length) model.wv.set(w.wv);
    if (w.wo && w.wo.length === model.wo.length) model.wo.set(w.wo);
    if (w.w1 && w.w1.length === model.w1.length) model.w1.set(w.w1);
    if (w.b1 && w.b1.length === model.b1.length) model.b1.set(w.b1);
    if (w.w2 && w.w2.length === model.w2.length) model.w2.set(w.w2);
    if (w.b2 && w.b2.length === model.b2.length) model.b2.set(w.b2);
    if (w.lmHead && w.lmHead.length === model.lmHead.length) model.lmHead.set(w.lmHead);
    return true;
  } catch (err) {
    console.error("Failed to load weights into model:", err);
    return false;
  }
}
