export function generateSFTFineTuningScript(modelName: string = 'unsloth/Llama-3.2-1B-Instruct'): string {
  return `"""
fine_tune_interview_model.py
============================
Step 3 in the Roadmap: Fine-Tuning a 1B-8B Open Model (Product, Not Pretraining)

Instead of spending weeks pretraining from scratch, take a high-performing open foundation
model (like Llama-3.2-1B, Qwen-2.5-1.5B, or Mistral-7B) and apply LoRA (Low-Rank Adaptation)
Supervised Fine-Tuning (SFT) on your specific interview transcript format.

Requirements:
  pip install unsloth "xformers<0.0.28" "trl<0.9.0" peft accelerate bitsandbytes
  # Or standard HuggingFace: pip install torch transformers datasets trl peft
"""

import json
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Choose base open foundation model (1B to 8B parameters)
# Options:
#   "unsloth/Llama-3.2-1B-Instruct" (Super fast, fits in 4GB VRAM)
#   "unsloth/Qwen2.5-1.5B-Instruct" (Great multi-turn dialogue)
#   "unsloth/Meta-Llama-3.1-8B-Instruct" (Full production quality)
MODEL_NAME = "${modelName}"

# 2. Prepare Interview Dataset in Standard Conversational Format (ChatML / Messages)
interview_samples = [
    {
        "messages": [
            {"role": "system", "content": "You are an expert technical interviewer evaluating distributed systems and AI candidates."},
            {"role": "user", "content": "Can you explain why we need causal masking in autoregressive transformers?"},
            {"role": "assistant", "content": "In an autoregressive transformer, each token should only attend to tokens that appear before it. The causal mask sets upper-triangular attention logits to negative infinity before the softmax, ensuring position t has zero attention weight on future positions t+1."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are an expert technical interviewer evaluating distributed systems and AI candidates."},
            {"role": "user", "content": "Why should we decouple memory programs from foundation model weights?"},
            {"role": "assistant", "content": "Foundation model pretraining optimizes general linguistic syntax and world knowledge. Forcing private interview transcripts or candidate facts into weights causes catastrophic forgetting and hallucination. Decoupling memory into an external episodic store ensures exact factual retrieval."}
        ]
    }
]

# Save sample dataset
with open("interview_data.jsonl", "w") as f:
    for s in interview_samples:
        f.write(json.dumps(s) + "\\n")

print(f"--> Loaded interview dataset with {len(interview_samples)} dialogues.")

# 3. Load Base Model with Fast 4-bit Quantization (fits free Colab T4 GPU)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 4. Attach LoRA Adapters (train only ~1% of parameters)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,            # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 5. Fine-Tune with SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = Dataset.from_list(interview_samples),
    dataset_text_field = "messages",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("--> Starting SFT Fine-Tuning...")
trainer.train()

# 6. Save LoRA Adapter Checkpoint (Only ~50 MB!)
model.save_pretrained("interview_lora_adapter")
tokenizer.save_pretrained("interview_lora_adapter")
print("--> Fine-tuned LoRA weights successfully saved to: interview_lora_adapter")

# 7. Test Generation with Fine-Tuned Persona
FastLanguageModel.for_inference(model)
messages = [
    {"role": "system", "content": "You are an expert technical interviewer evaluating distributed systems and AI candidates."},
    {"role": "user", "content": "How do you test candidate memory recall?"}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=150, temperature=0.7)
print("\\nModel Output:")
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
`;
}

export function generateSFTColabJSON(): string {
  const script = generateSFTFineTuningScript('unsloth/Llama-3.2-1B-Instruct');
  const notebook = {
    nbformat: 4,
    nbformat_minor: 0,
    metadata: {
      colab: { name: "Fine_Tune_Interview_Model_Llama3.ipynb", provenance: [] },
      kernelspec: { name: "python3", display_name: "Python 3" },
      language_info: { name: "python" }
    },
    cells: [
      {
        cell_type: "markdown",
        metadata: {},
        source: [
          "# Step 3: Fine-Tune an Open 1B-8B Model on Your Interview Format\n",
          "**\"That is product. It is not more pretraining.\"**\n",
          "Take an existing 1B-8B base model (Llama-3.2-1B / Qwen-2.5) and apply LoRA supervised fine-tuning in ~15 minutes on a free Google Colab T4 GPU."
        ]
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [
          "%%capture\n",
          "!pip install unsloth\n",
          "!pip install --no-deps \"xformers<0.0.28\" \"trl<0.9.0\" peft accelerate bitsandbytes"
        ]
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: [script]
      }
    ]
  };
  return JSON.stringify(notebook, null, 2);
}
