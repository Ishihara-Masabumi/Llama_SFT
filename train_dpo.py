"""
DPO training script for SFT-trained Llama-3.2-3B-Instruct
using Aratako/iterative-dpo-data-for-SimPO-iter2 dataset.

Uses the merged SFT model (LoRA merged into base) so that
DPO trainer can cleanly create its own LoRA + reference model.
Precomputes ref log probs to avoid keeping ref model on GPU during training.
"""

import logging
import torch
from peft import LoraConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


def format_row(row):
    """Format dataset row for DPO trainer with chat template."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main():
    model_path = "/home/ubuntu/Llama_SFT/sft_merged_llama3"
    output_dir = "/home/ubuntu/Llama_SFT/results_dpo_simpo"

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset")
    dataset = load_dataset(
        "Aratako/iterative-dpo-data-for-SimPO-iter2",
        split="train",
    )
    dataset = dataset.map(format_row, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    peft_config = LoraConfig(
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="adamw_torch",
        max_length=2048,
        beta=0.1,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=2,
    )

    logger.info("Setting up DPO trainer")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
