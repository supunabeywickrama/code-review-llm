"""
evaluate.py
-----------
Evaluation pipeline comparing three model versions:
  1. Base model (no fine-tuning)
  2. SFT model  (supervised fine-tuning)
  3. DPO model  (SFT + preference alignment)

Metrics computed:
  - BLEU-4
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BERTScore (F1)
  - Avg response length
  - Specificity score (heuristic: presence of actionable keywords)

Results are logged to W&B and saved as JSON + CSV.

Run:
    python src/evaluation/evaluate.py --config configs/train_config.yaml --num_samples 200
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb
import torch
import nltk
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from datasets import load_from_disk
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from evaluate import load as load_metric
from bert_score import score as bert_score

load_dotenv()
nltk.download("punkt", quiet=True)

RESULTS_DIR = Path("./outputs/evaluation")

# Keywords that indicate a specific, actionable review (heuristic)
ACTIONABLE_KEYWORDS = [
    "use", "replace", "add", "remove", "consider", "avoid", "rename",
    "refactor", "extract", "move", "check", "handle", "ensure", "fix",
    "security", "performance", "memory", "complexity", "type hint",
    "edge case", "unit test", "error handling",
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = load_metric("bleu")
    tokenized_preds = [nltk.word_tokenize(p.lower()) for p in predictions]
    tokenized_refs  = [[nltk.word_tokenize(r.lower())] for r in references]
    result = bleu.compute(predictions=tokenized_preds, references=tokenized_refs)
    return round(result["bleu"] * 100, 2)


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    rouge = load_metric("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    return {k: round(v * 100, 2) for k, v in result.items()}


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return round(F1.mean().item() * 100, 2)


def specificity_score(predictions: list[str]) -> float:
    """Heuristic: fraction of predictions containing at least 2 actionable keywords."""
    actionable = 0
    for pred in predictions:
        pred_lower = pred.lower()
        hits = sum(1 for kw in ACTIONABLE_KEYWORDS if kw in pred_lower)
        if hits >= 2:
            actionable += 1
    return round(100 * actionable / len(predictions), 2)


def avg_length(texts: list[str]) -> float:
    return round(sum(len(t.split()) for t in texts) / len(texts), 1)


# ── Model inference ────────────────────────────────────────────────────────────

def load_pipeline(model_path: str, max_new_tokens: int = 256):
    """Load a causal LM pipeline (works for both base and fine-tuned models)."""
    logger.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,         # greedy for reproducibility
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return pipe


def generate_reviews(pipe, prompts: list[str]) -> list[str]:
    """Run batch inference and extract only the assistant's response."""
    outputs = []
    for prompt in prompts:
        result = pipe(prompt)[0]["generated_text"]
        # Extract text after the last assistant header
        if "<|start_header_id|>assistant<|end_header_id|>" in result:
            response = result.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            response = response.replace("<|eot_id|>", "").strip()
        else:
            response = result[len(prompt):].strip()
        outputs.append(response)
    return outputs


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    prompts: list[str],
    references: list[str],
    model_label: str,
) -> dict:
    """Evaluate a single model on all metrics."""
    logger.info(f"\n{'='*60}\nEvaluating: {model_label}\n{'='*60}")
    pipe = load_pipeline(model_path)
    predictions = generate_reviews(pipe, prompts)

    metrics = {
        "model": model_label,
        "bleu_4":         compute_bleu(predictions, references),
        "rouge_l":        compute_rouge(predictions, references)["rougeL"],
        "rouge_1":        compute_rouge(predictions, references)["rouge1"],
        "bertscore_f1":   compute_bertscore(predictions, references),
        "specificity":    specificity_score(predictions),
        "avg_length_wds": avg_length(predictions),
        "num_samples":    len(predictions),
    }

    logger.info(json.dumps({k: v for k, v in metrics.items() if k != "model"}, indent=2))
    return metrics, predictions


def run_evaluation(config_path: str, num_samples: int = 100) -> None:
    cfg = OmegaConf.load(config_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wandb.init(project="code-review-llm", name="evaluation-comparison")

    # ── Load eval subset ──────────────────────────────────────────────────────
    dataset = load_from_disk(cfg.data.sft_dataset_path)["validation"]
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    prompts    = dataset["prompt"]
    references = dataset["completion"]
    logger.info(f"Evaluating on {len(prompts)} samples")

    # ── Evaluate each model ───────────────────────────────────────────────────
    model_configs = [
        (cfg.model.base_model,  "base_model"),
        (cfg.sft.output_dir,    "sft_model"),
        (cfg.dpo.output_dir,    "dpo_model"),
    ]

    all_metrics  = []
    all_preds    = {}

    for model_path, label in model_configs:
        if not Path(model_path).exists() and not model_path.startswith("unsloth/"):
            logger.warning(f"Skipping {label} — path not found: {model_path}")
            continue
        metrics, preds = evaluate_model(model_path, prompts, references, label)
        all_metrics.append(metrics)
        all_preds[label] = preds
        wandb.log({f"{label}/{k}": v for k, v in metrics.items() if k not in ("model",)})

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)

    with open(RESULTS_DIR / "metrics_comparison.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save sample predictions side-by-side for qualitative review
    sample_rows = []
    for i in range(min(10, len(prompts))):
        row = {"prompt": prompts[i], "reference": references[i]}
        for label, preds in all_preds.items():
            row[f"pred_{label}"] = preds[i]
        sample_rows.append(row)

    pd.DataFrame(sample_rows).to_csv(RESULTS_DIR / "sample_predictions.csv", index=False)

    # ── Print summary table ───────────────────────────────────────────────────
    logger.info("\n" + results_df.to_string(index=False))

    # W&B comparison table
    wandb.log({"evaluation/metrics_table": wandb.Table(dataframe=results_df)})
    wandb.finish()
    logger.info(f"\nResults saved to {RESULTS_DIR} ✓")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, default="configs/train_config.yaml")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    run_evaluation(args.config, args.num_samples)
