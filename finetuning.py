"""
Llama-3.2-3B-Instruct LoRA/QLoRA Fine-tuning on JCommonsenseQA (Japanese MCQ)
- Output format: "[ANSWER] X" only
- Supervise ONLY the assistant completion part (prompt tokens are masked out)
"""

import os
import re
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "/home/ubuntu/Llama_SFT/output_mcq_answer_only"
DATASET_NAME = "leemeng/jcommonsenseqa"

# LoRA parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training parameters
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.1
SEED = 42

# If you want 4-bit QLoRA
USE_4BIT = True

# Eval sampling (small & fast)
EVAL_SAMPLES = 300
GEN_MAX_NEW_TOKENS = 8


# =============================================================================
# Data formatting
# =============================================================================
CHOICE_LABELS = ["A", "B", "C", "D", "E"]
ANSWER_RE = re.compile(r"\[ANSWER\]\s*([ABCDE])", re.IGNORECASE)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_example_as_messages(example):
    """
    Convert one JCommonsenseQA row into:
      user: question + choices
      assistant: "[ANSWER] X"
    """
    question = example["question"]
    choices = [example[f"choice{i}"] for i in range(5)]
    label = int(example["label"])  # 0..4
    answer_label = CHOICE_LABELS[label]

    choices_text = "\n".join(f"{CHOICE_LABELS[i]}. {choices[i]}" for i in range(5))

    user_message = (
        "以下の質問に対して、最も適切な選択肢を1つ選んでください。\n"
        "最後に必ず「[ANSWER] X」の形式で答えてください。\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[CHOICES]\n{choices_text}\n"
    )

    assistant_message = f"[ANSWER] {answer_label}"

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]
    return messages, answer_label


def format_chat_to_text(example, tokenizer):
    """
    Produce a single training text using the model's chat template.
    Also keep gold answer for eval.
    """
    messages, answer_label = format_example_as_messages(example)
    # add_generation_prompt=False because we include assistant message in text for SFT
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "answer": answer_label}


def extract_answer_from_generation(text: str):
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    # fallback: single letter
    m2 = re.search(r"\b([ABCDE])\b", text.strip().upper())
    if m2:
        return m2.group(1)
    return None


def quick_mcq_eval(model, tokenizer, dataset, n_samples=200, device="cuda"):
    """
    Very small evaluation:
    - Build prompt-only text by using add_generation_prompt=True
    - Generate short completion
    - Extract [ANSWER] X and compute accuracy
    """
    model.eval()
    n = min(len(dataset), n_samples)
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    idxs = idxs[:n]

    correct = 0
    total = 0

    for i in idxs:
        ex = dataset[i]
        gold = ex["answer"]

        # Reconstruct prompt-only from stored "text" is tricky.
        # Instead, rebuild messages from original fields if they exist.
        # But our mapped dataset removed original fields. So we do a simple trick:
        # The chat template includes the assistant answer; we cut it off at the assistant header.
        full_text = ex["text"]

        # Find last assistant header and keep everything up to it, then ask model to complete.
        # This depends on the chat template tokens for Llama-3.*; we use a robust split key.
        # Typical segment contains: "<|start_header_id|>assistant<|end_header_id|>"
        key = "<|start_header_id|>assistant<|end_header_id|>"
        pos = full_text.rfind(key)
        if pos == -1:
            # fallback: just use full_text (not ideal, but avoids crash)
            prompt_text = full_text
        else:
            prompt_text = full_text[: pos + len(key)] + "\n\n"

        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer_from_generation(gen)

        if pred == gold:
            correct += 1
        total += 1

    return {"mcq_acc": correct / max(total, 1), "n_eval": total}


# =============================================================================
# Main
# =============================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("Llama-3.2-3B-Instruct QLoRA LoRA SFT on JCommonsenseQA (Answer-only)")
    print("=" * 80)

    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and format dataset
    print("[2/6] Loading and formatting dataset...")
    dataset = load_dataset(DATASET_NAME)

    train_dataset = dataset["train"].map(
        lambda x: format_chat_to_text(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    val_dataset = dataset["validation"].map(
        lambda x: format_chat_to_text(x, tokenizer),
        remove_columns=dataset["validation"].column_names,
    )

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Validation: {len(val_dataset)} examples")
    print(f"  Sample text:\n{train_dataset[0]['text'][:600]}...")

    # Load model (QLoRA 4-bit optional)
    print("\n[3/6] Loading model...")
    if USE_4BIT:
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
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.config.use_cache = False

    # Configure LoRA
    print("[4/6] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collator: supervise ONLY completion after assistant header
    # For Llama-3.* chat template, the assistant section begins with:
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Training config
    print("\n[5/6] Starting training...")
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
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,  # <-- key: masks prompt tokens to -100
    )

    trainer.train()

    # Save final adapter/tokenizer
    print("\n[6/6] Saving model...")
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved to: {final_dir}")

    # Quick MCQ evaluation by generation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = quick_mcq_eval(trainer.model, tokenizer, val_dataset, n_samples=EVAL_SAMPLES, device=device)
    print("Quick MCQ eval:", metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()

