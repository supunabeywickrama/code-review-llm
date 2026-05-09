# Code Review LLM — Implementation Plan

> Fine-tuning LLaMA 3.1 8B with QLoRA + DPO for automated code reviews.
> All code is already written. This document covers **what to do, in what order, and where manual action is required**.

---

## Quick Reference: Execution Order

```
Phase 0  →  Account setup + API keys (manual, ~1 hr)
Phase 1  →  python src/data/dataset_prep.py
Phase 2  →  python src/training/sft_train.py
Phase 3  →  python src/training/dpo_train.py
Phase 4  →  python src/evaluation/evaluate.py
Phase 5  →  Deploy FastAPI + Gradio to HF Spaces
```

---

## Phase 0 — Environment Setup (Manual)

### 0.1 Create Accounts (all free)

| Service | Purpose | URL |
|---|---|---|
| Hugging Face | Download base model, push trained adapters | https://huggingface.co |
| Weights & Biases | Experiment tracking, loss curves, eval tables | https://wandb.ai |
| GitHub | Pull real PR review data (optional, fallback exists) | https://github.com |

**HF Token:** Settings → Access Tokens → New token → set scope to **Write**
**W&B Key:** User Settings → API Keys → copy key

### 0.2 Request LLaMA 3.1 Access (Manual — do this first, takes ~5 min to approve)

Visit the model page and click **Request Access**:
`https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct`

Without this, the model download will be blocked with a 401 error.

### 0.3 Fill in `.env`

```bash
cp .env.example .env
```

Edit `.env` with your real keys:

```env
WANDB_API_KEY=...        # from wandb.ai/settings
HF_TOKEN=...             # write-scope token from huggingface.co
GITHUB_TOKEN=...         # optional, for live GitHub PR data
OPENAI_API_KEY=...       # optional, only if using GPT-4o rubric scoring
```

### 0.4 Update `configs/train_config.yaml` (Critical)

Open `configs/train_config.yaml` and change line 59:

```yaml
# BEFORE (will silently fail)
hub_model_id: "your-username/code-review-llm"

# AFTER
hub_model_id: "YOUR_ACTUAL_HF_USERNAME/code-review-llm"
```

### 0.5 Install Dependencies

```bash
pip install -r requirements.txt
```

For Google Colab (required for GPU training):

```python
!pip install unsloth[colab-new] trl peft wandb omegaconf loguru -q
```

---

## Phase 1 — Dataset Preparation

**Script:** `src/data/dataset_prep.py`
**Runtime:** ~15 min | Runs locally or on Colab | No GPU needed

### What it does

1. Downloads `microsoft/CodeReviewData` from Hugging Face
2. Falls back to synthetic examples if the dataset is unavailable
3. Formats rows into SFT chat templates: `instruction → ideal_review`
4. Creates DPO preference pairs: `prompt → (chosen good review, rejected vague review)`
5. Saves train/val splits to `data/`

### Command

```bash
python src/data/dataset_prep.py --max_samples 5000
```

**Manual decision:** Start with `--max_samples 1000` for a quick smoke test. Use 5000 for the final run.

### Expected outputs

```
data/
  sft_dataset/     (train + validation splits)
  dpo_dataset/     (train + validation splits)
```

### Verify success

The script prints dataset statistics at the end. You should see roughly:
- SFT train: ~4500 samples, val: ~500
- DPO train: ~4500 pairs, val: ~500

---

## Phase 2 — SFT Training with QLoRA

**Script:** `src/training/sft_train.py`
**Runtime:** 2–4 hours on Colab T4 | **Requires GPU — cannot run locally without one**

### What it does

1. Loads `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (pre-quantized, avoids OOM)
2. Attaches LoRA adapters (rank 16, alpha 32) to attention + MLP layers
3. Trains 3 epochs using TRL's `SFTTrainer` — only trains on completion tokens, not the prompt
4. Logs loss/eval metrics to W&B every 10 steps
5. Saves adapter weights to `outputs/sft_model/` and pushes to HF Hub

### Colab Setup

```python
# In a Colab cell
!git clone https://github.com/YOUR_USERNAME/code-review-llm
%cd code-review-llm
!pip install unsloth[colab-new] trl peft wandb omegaconf loguru -q
!cp .env.example .env  # then fill in keys via the file editor
```

Runtime → Change runtime type → **T4 GPU**

### Command

```bash
python src/training/sft_train.py --config configs/train_config.yaml
```

### Key config values (`configs/train_config.yaml`)

| Key | Default | Notes |
|---|---|---|
| `sft.num_train_epochs` | 3 | Reduce to 1 for a quick test |
| `sft.per_device_train_batch_size` | 2 | **Reduce to 1 if you get CUDA OOM** |
| `sft.learning_rate` | 2e-4 | Standard for LoRA fine-tuning |
| `lora.r` | 16 | Higher = more capacity, more VRAM |

### What to watch in W&B

- `train/loss` should drop from ~2.5 → ~0.8 over 3 epochs
- `eval/loss` should track training loss without diverging
- If eval loss increases while train loss decreases → overfitting, reduce epochs

### Expected output

```
outputs/sft_model/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
```

---

## Phase 3 — DPO Alignment

**Script:** `src/training/dpo_train.py`
**Runtime:** 1–2 hours on Colab T4 | Run in the same session immediately after SFT

### What it does

1. Loads the SFT checkpoint from `outputs/sft_model/`
2. Reattaches fresh LoRA adapters for DPO training
3. Trains 1 epoch on preference pairs using DPO loss (no reward model needed)
4. Pushes aligned model to HF Hub as `code-review-llm-dpo`

### Command

```bash
python src/training/dpo_train.py --config configs/train_config.yaml
```

### Key config values

| Key | Default | Notes |
|---|---|---|
| `dpo.beta` | 0.1 | Lower = stronger preference signal. Keep at 0.1 |
| `dpo.num_train_epochs` | 1 | DPO needs fewer epochs than SFT |
| `dpo.per_device_train_batch_size` | 1 | Already set to 1 (DPO is more memory-intensive) |

### What to watch in W&B

- `rewards/chosen` should be consistently higher than `rewards/rejected`
- The gap between them should widen over training — this is the alignment working
- `dpo/loss` should decrease steadily

### Expected output

```
outputs/dpo_model/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
```

---

## Phase 4 — Evaluation

**Script:** `src/evaluation/evaluate.py`
**Runtime:** ~30 min on Colab T4 | Runs all three models sequentially

### What it does

Loads base model, SFT checkpoint, and DPO checkpoint. For each, generates reviews on 200 test samples and computes:

| Metric | What it measures |
|---|---|
| BLEU-4 | N-gram overlap with reference reviews |
| ROUGE-L | Longest common subsequence with references |
| BERTScore F1 | Semantic similarity using BERT embeddings |
| Specificity | Heuristic: presence of actionable keywords (rename, extract, null check, etc.) |

### Command

```bash
python src/evaluation/evaluate.py --config configs/train_config.yaml --num_samples 200
```

**Manual decision:** Use `--num_samples 50` for a fast sanity check, 200 for the final run.

### Expected outputs

```
evaluation_results.csv      (one row per model × metric)
evaluation_results.json     (full results with sample predictions)
```

Results are also uploaded to W&B as a comparison table.

### Manual step: Update README

After evaluation, update the results table in `README.md` with your **actual** numbers. The current values (BLEU 14.2, ROUGE 33.1, etc.) are placeholders from the project design.

---

## Phase 5 — Deployment

### 5.1 FastAPI Server (local or cloud)

**Script:** `src/serving/api.py`

Endpoints:
- `POST /review` — accepts `{code, language}`, returns review + latency
- `GET /health` — liveness check
- `GET /model-info` — metadata

```bash
MODEL_PATH=./outputs/dpo_model uvicorn src.serving.api:app --port 8000
```

Test with curl:

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"code": "def add(a,b): return a+b", "language": "python"}'
```

### 5.2 Gradio Demo on HF Spaces (Manual, ~10 min)

**Script:** `src/serving/gradio_app.py`

1. Go to `https://huggingface.co/new-space`
2. Name it `code-review-llm-demo`
3. SDK: **Gradio** | Hardware: **CPU** (free) or **T4** ($0.60/hr)
4. Upload `src/serving/gradio_app.py` renamed as `app.py`
5. Add Space secrets: `HF_TOKEN`, `MODEL_PATH` (your HF Hub model ID)

**Note on CPU inference:** Expect 10–20s per review on free CPU tier. Set `load_in_4bit=True` in the script — it is already configured for this.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401 Unauthorized` when downloading model | LLaMA access not approved or HF_TOKEN not set | Check HF token in `.env`, verify access at huggingface.co/meta-llama |
| `CUDA out of memory` during SFT | Batch size too large | Set `per_device_train_batch_size: 1` in config |
| DPO `rewards/chosen` not higher than `rewards/rejected` | Preference pairs too similar | Check `data/dpo_dataset` — chosen and rejected reviews should differ meaningfully |
| W&B metrics not appearing | `WANDB_API_KEY` not loaded | Run `wandb login` manually or `import os; os.environ['WANDB_API_KEY'] = '...'` in notebook |
| Gradio Space shows error | `MODEL_PATH` secret missing or wrong | Set secrets in Space settings, use your HF Hub model ID not a local path |

---

## Project File Map

```
code-review-llm/
├── .env                          ← fill this in (not committed)
├── .env.example                  ← template
├── configs/
│   └── train_config.yaml         ← all hyperparameters (edit hub_model_id)
├── src/
│   ├── data/
│   │   └── dataset_prep.py       ← Phase 1
│   ├── training/
│   │   ├── sft_train.py          ← Phase 2
│   │   └── dpo_train.py          ← Phase 3
│   ├── evaluation/
│   │   └── evaluate.py           ← Phase 4
│   └── serving/
│       ├── api.py                ← Phase 5 (FastAPI)
│       └── gradio_app.py         ← Phase 5 (Gradio)
├── scripts/
│   └── run_pipeline.sh           ← runs phases 1–4 end-to-end
├── data/                         ← created by dataset_prep.py
└── outputs/                      ← created by training scripts
    ├── sft_model/
    └── dpo_model/
```

---

## Estimated Timeline

| Phase | Time | Blocking? |
|---|---|---|
| Phase 0: Setup | 1 hour | Yes — must complete before anything else |
| Phase 1: Dataset | 15 min | Yes — training depends on this |
| Phase 2: SFT | 2–4 hours | Yes — DPO depends on SFT checkpoint |
| Phase 3: DPO | 1–2 hours | Yes — evaluation needs all three models |
| Phase 4: Evaluation | 30 min | No — can deploy while this runs |
| Phase 5: Deployment | 30 min | No |
| **Total** | **~6–8 hours** | (mostly waiting for GPU training) |
