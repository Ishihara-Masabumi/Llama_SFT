"""
Evaluate fine-tuned Llama-3.2-3B-Instruct (completion-only LoRA) on JCommonsenseQA
Uses the same prompt format as finetuning.py: [ANSWER] X
"""

import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/home/ubuntu/Llama_SFT/output_mcq_answer_only/final"
DATASET_NAME = "leemeng/jcommonsenseqa"

CHOICE_LABELS = ["A", "B", "C", "D", "E"]
ANSWER_RE = re.compile(r"\[ANSWER\]\s*([ABCDE])", re.IGNORECASE)


def build_prompt(example):
    """Build the same prompt format used during finetuning.py."""
    question = example["question"]
    choices = [example[f"choice{i}"] for i in range(5)]
    choices_text = "\n".join(
        f"{CHOICE_LABELS[i]}. {choices[i]}" for i in range(5)
    )
    user_message = (
        "以下の質問に対して、最も適切な選択肢を1つ選んでください。\n"
        "最後に必ず「[ANSWER] X」の形式で答えてください。\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[CHOICES]\n{choices_text}\n"
    )
    return user_message


def extract_answer(text):
    """Extract answer label from model output."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    # Fallback: single letter
    m2 = re.search(r"\b([ABCDE])\b", text.strip().upper())
    if m2:
        return m2.group(1)
    return None


def main():
    print("=" * 60)
    print("Evaluating completion-only LoRA model on JCommonsenseQA")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model + LoRA adapter
    print("[2/4] Loading model with LoRA adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Load dataset
    print("[3/4] Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    val_dataset = dataset["validation"]
    print(f"  Validation: {len(val_dataset)} examples")

    # Evaluate
    print("[4/4] Running evaluation...")
    correct = 0
    total = 0
    errors = []

    for i, example in enumerate(val_dataset):
        user_message = build_prompt(example)
        messages = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )

        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        pred = extract_answer(generated)
        gold = CHOICE_LABELS[example["label"]]

        if pred == gold:
            correct += 1
        elif pred is None:
            errors.append((i, generated))

        total += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(val_dataset)}] Accuracy so far: {correct/total:.4f}")

    accuracy = correct / total
    print(f"\n{'=' * 60}")
    print(f"Results (completion-only LoRA):")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    if errors:
        print(f"  Parse failures: {len(errors)}")
        print(f"  Sample failures:")
        for idx, gen in errors[:5]:
            print(f"    [{idx}] '{gen}'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
