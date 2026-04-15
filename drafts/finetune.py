"""
Fine-tuning script for pay equity compliance SLMs
Uses Unsloth + QLoRA on RTX 5060 8GB

Supported models (change MODEL_NAME):
  - "unsloth/Meta-Llama-3.1-8B-Instruct"
  - "unsloth/mistral-7b-instruct-v0.3"
  - "unsloth/Phi-3.5-mini-instruct"

Usage:
    python finetune.py

Output:
    ./output/<model_short_name>/  — LoRA adapter + merged model
"""
import os
os.environ["UNSLOTH_USE_FUSED_CE_LOSS"] = "0"
import json
import torch
from datasets import Dataset

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "unsloth/Meta-Llama-3.1-8B-Instruct"  # change per run
OUTPUT_DIR   = "./output/llama3.1-8b"                 # change per run
DATA_PATH    = "training_data.jsonl"

MAX_SEQ_LEN  = 1024   
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.00

BATCH_SIZE        = 1
GRAD_ACCUM        = 8      
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 3
WARMUP_STEPS      = 10
SAVE_STEPS        = 50
LOGGING_STEPS     = 10

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,           # auto: bfloat16 on RTX 5060
        load_in_4bit=True,    # QLoRA
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer

# ── LOAD DATASET ──────────────────────────────────────────────────────────────

def load_dataset(tokenizer):
    records = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} training examples.")

    def format_example(rec):
        instruction = rec["instruction"]
        inp         = rec.get("input", "")
        output      = rec["output"]

        if inp:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )
        return {"text": prompt}

    formatted = [format_example(r) for r in records]
    dataset = Dataset.from_list(formatted)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return dataset

# ── TRAIN ─────────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="input_ids",
        max_seq_length=MAX_SEQ_LEN,
        args=args,
        packing=False,    
    )

    print(f"\nStarting fine-tuning: {MODEL_NAME}")
    print(f"Examples: {len(dataset)} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}x{GRAD_ACCUM}")
    print(f"Output: {OUTPUT_DIR}\n")

    trainer.train(resume_from_checkpoint=True)

# ── SAVE ──────────────────────────────────────────────────────────────────────

def save(model, tokenizer, trainer):
    # Save LoRA adapter only (small, ~50MB)
    lora_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"\nLoRA adapter saved to: {lora_path}")

    # Save training loss log
    log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    log = trainer.state.log_history
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to: {log_path}")

    # Print final loss
    train_losses = [e["loss"] for e in log if "loss" in e]
    if train_losses:
        print(f"\nFinal training loss: {train_losses[-1]:.4f}")
        print(f"Initial training loss: {train_losses[0]:.4f}")

# ── VRAM CHECK ────────────────────────────────────────────────────────────────

def check_vram():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training will be very slow on CPU.")
        return
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    name    = torch.cuda.get_device_name(0)
    print(f"GPU: {name} | VRAM: {vram_gb:.1f} GB")
    if vram_gb < 7:
        print("WARNING: Less than 7GB VRAM — reduce BATCH_SIZE to 1 if OOM.")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_vram()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = load_model()

    print("\nLoading dataset...")
    dataset = load_dataset(tokenizer)

    trainer = train(model, tokenizer, dataset)
    save(model, tokenizer, trainer)

    print("\nDone. To run inference with the adapter:")
    print(f"  from unsloth import FastLanguageModel")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{OUTPUT_DIR}/lora_adapter')")
