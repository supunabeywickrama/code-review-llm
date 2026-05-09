"""
sft_train.py
------------
Supervised Fine-Tuning (SFT) of LLaMA 3.1 8B using QLoRA via Unsloth.

Key techniques:
  - 4-bit NF4 quantisation (bitsandbytes)
  - LoRA adapters on all attention + MLP projection layers
  - Gradient checkpointing for memory efficiency
  - W&B experiment tracking

Run:
    python src/training/sft_train.py --config configs/train_config.yaml
"""

import os
import sys
from pathlib import Path

import wandb
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

load_dotenv()

# ── Unsloth must be imported before transformers patches ──────────────────────
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    logger.warning("Unsloth not installed. Falling back to standard transformers + PEFT.")
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    import torch


def load_model_and_tokenizer(cfg):
    """Load base model with 4-bit quantisation and attach LoRA adapters."""
    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model.base_model,
            max_seq_length=cfg.model.max_seq_length,
            dtype=None,           # auto-detect
            load_in_4bit=cfg.model.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=list(cfg.lora.target_modules),
            bias=cfg.lora.bias,
            use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
            random_state=42,
            use_rslora=False,
        )
        logger.info("Loaded model with Unsloth FastLanguageModel ✓")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=list(cfg.lora.target_modules),
            bias=cfg.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        logger.info("Loaded model with standard PEFT/bitsandbytes ✓")

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model, tokenizer


def get_training_args(cfg) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from config."""
    return TrainingArguments(
        output_dir=cfg.sft.output_dir,
        num_train_epochs=cfg.sft.num_train_epochs,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        learning_rate=cfg.sft.learning_rate,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,
        warmup_ratio=cfg.sft.warmup_ratio,
        fp16=cfg.sft.fp16,
        bf16=cfg.sft.bf16,
        logging_steps=cfg.sft.logging_steps,
        save_steps=cfg.sft.save_steps,
        eval_steps=cfg.sft.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=cfg.sft.save_total_limit,
        load_best_model_at_end=True,
        report_to=cfg.sft.report_to,
        run_name=cfg.sft.run_name,
        optim="adamw_8bit" if UNSLOTH_AVAILABLE else "adamw_torch",
        seed=42,
    )


def train(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)

    # ── W&B init ──────────────────────────────────────────────────────────────
    wandb.init(
        project="code-review-llm",
        name=cfg.sft.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading SFT dataset from {cfg.data.sft_dataset_path}")
    dataset = load_from_disk(cfg.data.sft_dataset_path)
    logger.info(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Only compute loss on the completion tokens (not the prompt)
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = get_training_args(cfg)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        data_collator=collator,
        args=training_args,
        packing=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting SFT training...")
    trainer_stats = trainer.train()
    logger.info(f"Training complete. Stats: {trainer_stats}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = Path(cfg.sft.output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")

    # ── Push to HF Hub (optional) ─────────────────────────────────────────────
    if cfg.hub.push_to_hub:
        logger.info(f"Pushing to Hugging Face Hub: {cfg.hub.hub_model_id}")
        model.push_to_hub(cfg.hub.hub_model_id, private=cfg.hub.private)
        tokenizer.push_to_hub(cfg.hub.hub_model_id, private=cfg.hub.private)

    wandb.finish()
    logger.info("SFT training pipeline complete ✓")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)
