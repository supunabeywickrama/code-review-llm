"""
dataset_prep.py
---------------
Prepares two datasets:
  1. SFT dataset  — (instruction, code_snippet) → ideal_review
  2. DPO dataset  — (prompt, chosen_review, rejected_review)

Sources used (all free):
  - Hugging Face: microsoft/CodeReviewData
  - Hugging Face: codeparrot/github-code (for synthetic pair generation)
  - GitHub API via PyGitHub (optional, requires GITHUB_TOKEN)
"""

import os
import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
SFT_DIR  = DATA_DIR / "sft_dataset"
DPO_DIR  = DATA_DIR / "dpo_dataset"

# ── Prompt templates ───────────────────────────────────────────────────────────
SFT_SYSTEM = (
    "You are an expert software engineer performing a thorough code review. "
    "Provide specific, actionable feedback on correctness, style, performance, "
    "and security. Be concise but precise."
)

SFT_TEMPLATE = """### Code to Review:
```{language}
{code}
```

### Review:"""

DPO_TEMPLATE = """### Code to Review:
```{language}
{code}
```

### Review:"""


def format_sft_example(code: str, review: str, language: str = "python") -> dict:
    """Format a single example into the chat template expected by Llama-3."""
    return {
        "prompt": SFT_TEMPLATE.format(language=language, code=code),
        "completion": review,
        "text": (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SFT_SYSTEM}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{SFT_TEMPLATE.format(language=language, code=code)}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{review}<|eot_id|>"
        ),
    }


def format_dpo_example(
    code: str, chosen: str, rejected: str, language: str = "python"
) -> dict:
    """Format a preference pair for DPO training."""
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SFT_SYSTEM}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{DPO_TEMPLATE.format(language=language, code=code)}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return {
        "prompt": prompt,
        "chosen": chosen + "<|eot_id|>",
        "rejected": rejected + "<|eot_id|>",
    }


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_code_review_data(max_samples: int = 5000) -> pd.DataFrame:
    """
    Load from microsoft/CodeReviewData on Hugging Face.
    Falls back to a small synthetic set if unavailable.
    """
    logger.info("Loading microsoft/CodeReviewData...")
    try:
        ds = load_dataset("microsoft/CodeReviewData", split="train")
        df = ds.to_pandas()

        # Normalise column names — dataset may vary
        rename_map = {}
        for col in df.columns:
            if "code" in col.lower() or "diff" in col.lower():
                rename_map[col] = "code"
            if "comment" in col.lower() or "review" in col.lower():
                rename_map[col] = "review"
        df = df.rename(columns=rename_map)

        df = df[["code", "review"]].dropna()
        df["language"] = "python"
        logger.info(f"Loaded {len(df)} samples from CodeReviewData")
        return df.head(max_samples)

    except Exception as e:
        logger.warning(f"Could not load CodeReviewData: {e}. Using synthetic fallback.")
        return _synthetic_fallback(max_samples)


def _synthetic_fallback(n: int = 200) -> pd.DataFrame:
    """Minimal synthetic examples so the pipeline always runs end-to-end."""
    templates = [
        {
            "code": "def add(a, b):\n    return a + b",
            "review": (
                "**Correctness:** Looks correct for numeric types.\n"
                "**Improvement:** Add type hints — `def add(a: int, b: int) -> int`. "
                "Consider handling non-numeric input with a try/except or isinstance check."
            ),
        },
        {
            "code": "for i in range(len(lst)):\n    print(lst[i])",
            "review": (
                "**Style:** Prefer `for item in lst: print(item)` — avoid index iteration "
                "unless the index is needed. This is more Pythonic and less error-prone."
            ),
        },
        {
            "code": "password = 'admin123'\ndb.connect(host, password)",
            "review": (
                "**Security (Critical):** Hardcoded credentials are a severe security risk. "
                "Use environment variables (`os.getenv('DB_PASSWORD')`) or a secrets manager. "
                "Never commit credentials to version control."
            ),
        },
    ]
    rows = []
    for i in range(n):
        t = templates[i % len(templates)]
        rows.append({"code": t["code"], "review": t["review"], "language": "python"})
    return pd.DataFrame(rows)


def build_dpo_pairs(df: pd.DataFrame) -> list[dict]:
    """
    Build (chosen, rejected) pairs from the SFT dataframe.

    Strategy:
      - chosen  = the original high-quality review
      - rejected = a deliberately degraded version (vague, non-actionable)

    In a production project you'd collect real human preference labels.
    This heuristic is sufficient for demonstrating the pipeline.
    """
    pairs = []
    rejection_templates = [
        "Looks fine.",
        "This code could be better.",
        "I think there might be some issues here.",
        "Maybe refactor this?",
        "Okay code, but needs work.",
    ]
    for _, row in df.iterrows():
        chosen   = row["review"]
        rejected = random.choice(rejection_templates)
        pairs.append(
            format_dpo_example(
                code=row["code"],
                chosen=chosen,
                rejected=rejected,
                language=row.get("language", "python"),
            )
        )
    return pairs


# ── Main pipeline ──────────────────────────────────────────────────────────────

def prepare_datasets(
    max_samples: int = 5000,
    train_split: float = 0.9,
    seed: int = 42,
) -> None:
    """Full dataset preparation pipeline."""
    random.seed(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    df = load_code_review_data(max_samples=max_samples)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    split_idx = int(len(df) * train_split)
    train_df  = df[:split_idx]
    val_df    = df[split_idx:]

    # 2. Build SFT dataset
    logger.info("Building SFT dataset...")
    sft_train = [format_sft_example(r["code"], r["review"], r.get("language", "python"))
                 for _, r in train_df.iterrows()]
    sft_val   = [format_sft_example(r["code"], r["review"], r.get("language", "python"))
                 for _, r in val_df.iterrows()]

    sft_ds = DatasetDict({
        "train":      Dataset.from_list(sft_train),
        "validation": Dataset.from_list(sft_val),
    })
    sft_ds.save_to_disk(SFT_DIR)
    logger.info(f"SFT dataset saved → {SFT_DIR}  (train={len(sft_train)}, val={len(sft_val)})")

    # 3. Build DPO dataset
    logger.info("Building DPO preference dataset...")
    dpo_train = build_dpo_pairs(train_df)
    dpo_val   = build_dpo_pairs(val_df)

    dpo_ds = DatasetDict({
        "train":      Dataset.from_list(dpo_train),
        "validation": Dataset.from_list(dpo_val),
    })
    dpo_ds.save_to_disk(DPO_DIR)
    logger.info(f"DPO dataset saved  → {DPO_DIR}  (train={len(dpo_train)}, val={len(dpo_val)})")

    # 4. Save a quick stats report
    stats = {
        "total_samples":      len(df),
        "sft_train_samples":  len(sft_train),
        "sft_val_samples":    len(sft_val),
        "dpo_train_samples":  len(dpo_train),
        "dpo_val_samples":    len(dpo_val),
        "avg_code_length":    int(df["code"].str.len().mean()),
        "avg_review_length":  int(df["review"].str.len().mean()),
    }
    with open(DATA_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_datasets(
        max_samples=args.max_samples,
        train_split=args.train_split,
        seed=args.seed,
    )
