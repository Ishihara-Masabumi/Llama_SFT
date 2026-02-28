"""
Llama-3.2-3B-Instruct LoRA Fine-tuning on JCommonsenseQA
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "/home/ubuntu/Llama_SFT/output"
DATASET_NAME = "leemeng/jcommonsenseqa"

# LoRA parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training parameters
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.1


# =============================================================================
# Data formatting
# =============================================================================
CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def format_example(example):
    """Format a JCommonsenseQA example into an instruction-following format."""
    question = example["question"]
    choices = [example[f"choice{i}"] for i in range(5)]
    label = example["label"]
    answer_label = CHOICE_LABELS[label]
    answer_text = choices[label]

    choices_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {choices[i]}" for i in range(5)
    )

    # Build chat-style messages for Llama-3.2-Instruct format
    user_message = (
        f"以下の質問に対して、最も適切な選択肢を1つ選んでください。\n\n"
        f"質問: {question}\n\n"
        f"選択肢:\n{choices_text}"
    )
    assistant_message = f"{answer_label}. {answer_text}"

    return {"user_message": user_message, "assistant_message": assistant_message}


def format_chat(example, tokenizer):
    """Apply chat template to the formatted example."""
    formatted = format_example(example)
    messages = [
        {"role": "user", "content": formatted["user_message"]},
        {"role": "assistant", "content": formatted["assistant_message"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Llama-3.2-3B-Instruct LoRA Fine-tuning on JCommonsenseQA")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and format dataset
    print("[2/5] Loading and formatting dataset...")
    dataset = load_dataset(DATASET_NAME)
    train_dataset = dataset["train"].map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    val_dataset = dataset["validation"].map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset["validation"].column_names,
    )
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Validation: {len(val_dataset)} examples")
    print(f"  Sample:\n{train_dataset[0]['text'][:500]}...")

    # Load model with 4-bit quantization
    print("\n[3/5] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("[4/5] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training
    print("\n[5/5] Starting training...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}/final")
    print("=" * 60)


if __name__ == "__main__":
    main()
