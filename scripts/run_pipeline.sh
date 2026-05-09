#!/bin/bash
# run_pipeline.sh
# ───────────────
# Runs the full training + evaluation pipeline end-to-end.
# Usage: bash scripts/run_pipeline.sh

set -e   # Exit on any error

CONFIG="configs/train_config.yaml"

echo "=========================================="
echo "  Code Review LLM - Full Pipeline"
echo "=========================================="

echo ""
echo "[1/4] Preparing datasets..."
python src/data/dataset_prep.py --max_samples 5000

echo ""
echo "[2/4] SFT fine-tuning with QLoRA..."
python src/training/sft_train.py --config $CONFIG

echo ""
echo "[3/4] DPO alignment..."
python src/training/dpo_train.py --config $CONFIG

echo ""
echo "[4/4] Evaluating all model versions..."
python src/evaluation/evaluate.py --config $CONFIG --num_samples 200

echo ""
echo "=========================================="
echo "  Pipeline complete! Results in ./outputs"
echo "=========================================="
