"""
dpo_train.py
------------
Direct Preference Optimization (DPO) alignment on top of the SFT checkpoint.

DPO trains the model to prefer "chosen" completions over "rejected" ones
without needing a separate reward model — using the implicit reward defined
by the log-ratio of the policy vs reference model.

Paper: "Direct Preference Optimization: Your Language Model is Secretly a
        Reward Model" (Rafailov et al., 2023) — https://arxiv.org/abs/2305.18290

Run:
    python src/training/dpo_train.py --config configs/train_config.yaml
"""

import os
from pathlib import Path

import wandb
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig

load_dotenv()

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel


def load_sft_model(cfg):
    """Load the SFT checkpoint (with LoRA adapters merged for DPO reference)."""
    sft_path = cfg.sft.output_dir
    logger.info(f"Loading SFT model from {sft_path}")

    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_path,
            max_seq_length=cfg.model.max_seq_length,
            dtype=None,
            load_in_4bit=cfg.model.load_in_4bit,
        )
        # Re-attach fresh LoRA adapters for continued training
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=list(cfg.lora.target_modules),
            bias=cfg.lora.bias,
            use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
            random_state=42,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(sft_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, sft_path)

    tokenizer.padding_side = "left"   # DPO requires left-padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_dpo_config(cfg) -> DPOConfig:
    """Build TRL DPOConfig from training config."""
    return DPOConfig(
        output_dir=cfg.dpo.output_dir,
        num_train_epochs=cfg.dpo.num_train_epochs,
        per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
        learning_rate=cfg.dpo.learning_rate,
        beta=cfg.dpo.beta,               # KL penalty coefficient
        max_prompt_length=cfg.dpo.max_prompt_length,
        max_length=cfg.dpo.max_length,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        report_to=cfg.dpo.report_to,
        run_name=cfg.dpo.run_name,
        optim="adamw_8bit" if UNSLOTH_AVAILABLE else "adamw_torch",
        remove_unused_columns=False,
        seed=42,
    )


def train(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project="code-review-llm",
        name=cfg.dpo.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading DPO dataset from {cfg.data.dpo_dataset_path}")
    dataset = load_from_disk(cfg.data.dpo_dataset_path)
    logger.info(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_sft_model(cfg)

    dpo_config = get_dpo_config(cfg)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # None → use the model itself as frozen ref (memory-efficient)
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting DPO alignment training...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = Path(cfg.dpo.output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"DPO-aligned model saved to {output_dir}")

    # ── Log reward margin metric to W&B ──────────────────────────────────────
    eval_results = trainer.evaluate()
    logger.info(f"DPO eval results: {eval_results}")
    wandb.log({"dpo/reward_margin": eval_results.get("eval_rewards/margins", 0)})

    # ── Push to Hub ───────────────────────────────────────────────────────────
    if cfg.hub.push_to_hub:
        hub_id = cfg.hub.hub_model_id + "-dpo"
        logger.info(f"Pushing DPO model to Hub: {hub_id}")
        model.push_to_hub(hub_id, private=cfg.hub.private)
        tokenizer.push_to_hub(hub_id, private=cfg.hub.private)

    wandb.finish()
    logger.info("DPO training pipeline complete ✓")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)
