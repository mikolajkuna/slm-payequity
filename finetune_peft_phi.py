"""
Fine-tuning script for Phi-3.5-mini-instruct (microsoft/Phi-3.5-mini-instruct)
QLoRA for RTX 5060 8GB on Windows

Usage:
    python finetune_peft_phi.py

Output:
    ./output_peft/phi-3.5-mini/lora_adapter/
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR  = "./output_peft/phi-3.5-mini"
DATA_PATH   = "training_data.jsonl"

MAX_SEQ_LEN     = 2048

LORA_RANK       = 16
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.1

BATCH_SIZE      = 1
GRAD_ACCUM      = 8
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 3
WARMUP_STEPS    = 10
SAVE_STEPS      = 500
LOGGING_STEPS   = 5

# ── VRAM CHECK ────────────────────────────────────────────────────────────────

def check_vram():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available.")
        return
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} | VRAM: {vram_gb:.1f} GB")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        inp = rec.get("input", "")
        output = rec["output"]
        if inp:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}{tokenizer.eos_token}"
            )
        else:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}{tokenizer.eos_token}"
            )
        return {"text": text}

    formatted = [format_example(r) for r in records]
    return Dataset.from_list(formatted)

# ── TRAIN ─────────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        bf16=True,
        fp16=False,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=1,
        optim="paged_adamw_8bit",
        adam_beta2=0.999,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        seed=42,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    print(f"\nStarting fine-tuning: {MODEL_NAME}")
    print(f"Examples: {len(dataset)} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE}x{GRAD_ACCUM}")
    print(f"LoRA: r={LORA_RANK}, α={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Output: {OUTPUT_DIR}\n")

    trainer.train()
    return trainer

# ── SAVE ──────────────────────────────────────────────────────────────────────

def save(model, tokenizer, trainer):
    lora_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"\nLoRA adapter saved to: {lora_path}")

    log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    log = trainer.state.log_history
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    train_losses = [e["loss"] for e in log if "loss" in e]
    if train_losses:
        print(f"Initial loss: {train_losses[0]:.4f}")
        print(f"Final loss:   {train_losses[-1]:.4f}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_vram()
    model, tokenizer = load_model()
    dataset = load_dataset(tokenizer)
    trainer = train(model, tokenizer, dataset)
    save(model, tokenizer, trainer)
    print("\nDone.")
