# SLM-PayEquity – Small Language Models for Pay Equity Compliance

This repository implements fine-tuning and evaluation pipelines for **Small Language Models (SLMs)**
applied to pay equity compliance tasks under **EU Directive 2023/970**.

Three open-weight models are evaluated:
- [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Phi-3.5 Mini Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)

Fine-tuning uses **QLoRA** (r=16, α=16, dropout=0.1, 3 epochs) on a synthetic instruction dataset.
Each model is benchmarked against a **zero-shot baseline** and a **RAG baseline** (k=3 few-shot retrieval without fine-tuning) across three compliance task categories (TC1–TC3).

This repository accompanies the paper:
> Kuna, M., Kowalczyk, M.: *Small Language Models for Pay Equity Compliance: A Fine-Tuning and Evaluation Study*.
> FedCSIS 2026, Riga, Latvia.

---

## Tasks

Three compliance task categories derived from EU Directive 2023/970:

| Task | Input | Output | Description |
|---|---|---|---|
| **TC1** | Natural language analyst query | JSON config | Extract analysis parameters (target variable, comparison groups, controls, filters) |
| **TC2** | JSON pipeline output | Narrative report | Generate professional pay equity compliance narrative |
| **TC3** | Narrative report | JSON checklist | Evaluate compliance with Article 9 mandatory disclosure items |

---

## Results

| Model | Zero-shot | RAG k=3 | Fine-tuned |
|---|---|---|---|
| Llama 3.1 8B | 0.294 | 0.268 | **0.839** |
| Mistral 7B | 0.219 | 0.535 | **0.833** |
| Phi-3.5 Mini | 0.193 | 0.484 | **0.756** |

Scores are weighted averages across TC1, TC2, TC3. See `reports/results/` for per-task breakdowns.

---

## Data

All training and test examples are **synthetic** — no real employee records are used.

| Split | Examples | Location |
|---|---|---|
| Train | 180 (60 per TC) | `data/raw/train.jsonl` |
| Test | 60 (20 per TC) | `data/raw/test.jsonl` |
| Ablation subsets | 20 / 40 / 60 | `data/raw/train_n*.jsonl` |

Per-task training files: `data/raw/training_tc{1,2,3}_{1,2,3}.jsonl`

The dataset encodes realistic Polish HR pay equity scenarios consistent with
Directive 2023/970 requirements, generated synthetically using a CTGAN-based
compensation dataset (n=2,000) as the statistical backend.

---

## Fine-Tuning Configuration

| Parameter | Llama / Mistral | Phi-3.5 Mini |
|---|---|---|
| Method | QLoRA (NF4, double quant) | QLoRA (NF4) |
| LoRA rank (r) | 16 | 16 |
| LoRA alpha (α) | 16 | 16 |
| LoRA dropout | 0.1 | 0.1 |
| Target modules | all linear layers | all linear layers |
| Epochs | 3 | 3 |
| Learning rate | 2×10⁻⁴ | 2×10⁻⁴ |
| Optimizer | paged AdamW 8-bit | paged AdamW 8-bit |
| Effective batch size | 8 (1×8 grad accum) | 8 (1×8 grad accum) |
| Script | `src/finetune.py` | `src/finetune_phi.py` |

Hardware: Dell Precision 5820, Intel Xeon W-2155, 64 GB RAM, RTX 5060 8 GB VRAM.

> **Note on Phi-3.5 Mini**: The `microsoft/Phi-3.5-mini-instruct` modeling file contains
> `DynamicCache` API calls incompatible with transformers ≥ 4.44. A patched version is
> provided in `src/modeling_phi3_fixed.py`. Copy it to the HuggingFace modules cache
> before running evaluation:
> ```
> %USERPROFILE%\.cache\huggingface\modules\transformers_modules\microsoft\
>   Phi_hyphen_3_dot_5_hyphen_mini_hyphen_instruct\<hash>\modeling_phi3.py
> ```

---

## RAG Baseline

Retrieval-augmented generation without fine-tuning. For each test query, k=3 most similar
training examples are retrieved via cosine similarity using
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings
and a FAISS flat inner product index, then injected as few-shot context.

Script: `src/rag_baseline.py`

---

## How to Run

```bash
# Setup
python -m venv venv
venv\Scripts\activate       # Windows
pip install torch transformers peft trl bitsandbytes datasets sentence-transformers faiss-cpu

# Fine-tune Llama or Mistral (change MODEL_NAME + OUTPUT_DIR at top of script)
python src/finetune.py

# Fine-tune Phi-3.5 Mini
python src/finetune_phi.py

# Evaluate fine-tuned model (set BASE_MODEL + ADAPTER_PATH at top of script)
python src/evaluate.py

# Run RAG baseline (set BASE_MODEL + OUTPUT_DIR at top of script)
python src/rag_baseline.py
```

CUDA ≥ 12.1 and bitsandbytes required for QLoRA.

---

## Repository Structure

```
slm-payequity/
├── data/
│   └── raw/
│       ├── train.jsonl              # 180 training examples
│       ├── test.jsonl               # 60 test examples
│       ├── train_n20/40/60.jsonl    # ablation subsets
│       ├── training_tc1_*.jsonl     # per-task training files
│       ├── training_tc2_*.jsonl
│       ├── training_tc3_*.jsonl
│       └── salary_data_2009.csv     # synthetic compensation dataset
│
├── src/
│   ├── finetune.py                  # QLoRA fine-tuning (Llama, Mistral)
│   ├── finetune_phi.py              # QLoRA fine-tuning (Phi-3.5 Mini)
│   ├── evaluate.py                  # Evaluation script (zero-shot + fine-tuned)
│   ├── rag_baseline.py              # RAG baseline pipeline
│   └── modeling_phi3_fixed.py       # Patched Phi-3.5 modeling file
│
├── reports/
│   └── results/
│       └── eval_results/            # Per-model JSON result files
│           ├── llama3.1-8b/
│           ├── mistral-7b/
│           ├── phi-3.5-mini/
│           └── rag/
│
├── drafts/                          # Development scripts (Ollama-based prototypes)
└── README.md
```

---

## Citation

```bibtex
@inproceedings{kuna2026slm,
  title     = {Small Language Models for Pay Equity Compliance: A Fine-Tuning and Evaluation Study},
  author    = {Kuna, Miko{\l}aj and Kowalczyk, Marcin},
  booktitle = {Proceedings of FedCSIS 2026},
  year      = {2026}
}
```
