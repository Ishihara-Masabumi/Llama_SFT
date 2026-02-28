"""
Evaluate SFT model (before DPO) vs DPO model (after DPO).
- Qualitative: side-by-side generation comparison on Japanese prompts
- Quantitative: DPO reward accuracy on held-out eval set
"""

import json
import sys
import time
import torch
from peft import PeftModel
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_PROMPTS = [
    "日本の四季の中で、あなたが最も好きな季節とその理由を教えてください。",
    "プログラミング初心者に、Pythonを学ぶメリットを3つ挙げて説明してください。",
    "健康的な食生活を送るために、毎日心がけるべきことを具体的にアドバイスしてください。",
    "AIが社会にもたらすメリットとデメリットについて、バランスよく論じてください。",
    "次の文章を英語に翻訳してください：「桜の季節になると、多くの人が公園でお花見を楽しみます。」",
    "小学生にもわかるように、地球温暖化の原因と影響を説明してください。",
    "効率的な時間管理の方法を5つ、具体的な例を交えて提案してください。",
    "「努力は必ず報われる」という考え方に対して、賛成・反対の両方の立場から意見を述べてください。",
]

SFT_MODEL_PATH = "/home/ubuntu/Llama_SFT/sft_merged_llama3"
DPO_ADAPTER_PATH = "/home/ubuntu/Llama_SFT/results_dpo_simpo"


def load_model(model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


def compute_dpo_eval_metrics(model, ref_model, tokenizer, num_samples=200):
    """Compute reward accuracy on held-out DPO eval data."""
    dataset = load_dataset(
        "Aratako/iterative-dpo-data-for-SimPO-iter2", split="train"
    )
    split = dataset.train_test_split(test_size=0.05, seed=42)
    eval_data = split["test"].select(range(min(num_samples, len(split["test"]))))

    correct = 0
    total = 0

    for row in eval_data:
        messages_chosen = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["chosen"]},
        ]
        messages_rejected = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["rejected"]},
        ]

        text_chosen = tokenizer.apply_chat_template(
            messages_chosen, tokenize=False
        )
        text_rejected = tokenizer.apply_chat_template(
            messages_rejected, tokenize=False
        )

        for text, label in [(text_chosen, "chosen"), (text_rejected, "rejected")]:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=2048
            ).to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                ref_logits = ref_model(**inputs).logits

            # Compute log probs
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs["input_ids"][:, 1:]
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            ref_shift_logits = ref_logits[:, :-1, :]
            ref_log_probs = torch.nn.functional.log_softmax(ref_shift_logits, dim=-1)
            ref_token_log_probs = ref_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            if label == "chosen":
                chosen_reward = (token_log_probs.sum() - ref_token_log_probs.sum()).item()
            else:
                rejected_reward = (token_log_probs.sum() - ref_token_log_probs.sum()).item()

        if chosen_reward > rejected_reward:
            correct += 1
        total += 1

        if total % 50 == 0:
            print(f"  Eval progress: {total}/{len(eval_data)}, "
                  f"accuracy so far: {correct/total:.4f}")

    return correct / total if total > 0 else 0.0


def main():
    print("=" * 80)
    print("DPO学習 前後の評価")
    print("=" * 80)

    # --- Part 1: Qualitative evaluation (generation comparison) ---
    print("\n[1/2] 生成比較 (Qualitative Evaluation)")
    print("-" * 80)

    results = []

    # SFT model
    print("\nSFTモデルを読み込み中...")
    model_sft, tokenizer = load_model(SFT_MODEL_PATH)

    print("SFTモデルで生成中...")
    sft_responses = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"  Prompt {i+1}/{len(EVAL_PROMPTS)}...")
        resp = generate_response(model_sft, tokenizer, prompt)
        sft_responses.append(resp)

    # Keep SFT model as ref, load DPO on top
    ref_model = model_sft  # will be used for metrics later

    print("\nDPOモデルを読み込み中...")
    model_dpo, _ = load_model(SFT_MODEL_PATH, DPO_ADAPTER_PATH)

    print("DPOモデルで生成中...")
    dpo_responses = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"  Prompt {i+1}/{len(EVAL_PROMPTS)}...")
        resp = generate_response(model_dpo, tokenizer, prompt)
        dpo_responses.append(resp)

    # Print results
    print("\n" + "=" * 80)
    print("生成結果の比較")
    print("=" * 80)

    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"\n{'─' * 80}")
        print(f"📝 プロンプト {i+1}: {prompt}")
        print(f"{'─' * 80}")
        print(f"\n【SFTモデル (DPO前)】")
        print(sft_responses[i][:500])
        if len(sft_responses[i]) > 500:
            print(f"... (全{len(sft_responses[i])}文字)")
        print(f"\n【DPOモデル (DPO後)】")
        print(dpo_responses[i][:500])
        if len(dpo_responses[i]) > 500:
            print(f"... (全{len(dpo_responses[i])}文字)")

        results.append({
            "prompt": prompt,
            "sft_response": sft_responses[i],
            "dpo_response": dpo_responses[i],
            "sft_length": len(sft_responses[i]),
            "dpo_length": len(dpo_responses[i]),
        })

    # --- Part 2: Quantitative evaluation (reward accuracy) ---
    print("\n" + "=" * 80)
    print("[2/2] 定量評価 (Reward Accuracy on Eval Set)")
    print("-" * 80)

    # Need both models on GPU - check memory
    # ref_model is SFT (already on GPU), model_dpo is DPO (already on GPU)
    # This won't work with both on GPU for 3B model - evaluate sequentially

    # Move DPO model off GPU, compute SFT reward accuracy
    del model_dpo
    torch.cuda.empty_cache()

    print("\nSFTモデルのreward accuracy計算中...")
    # For SFT model, reward relative to itself is always 0, so accuracy = random
    # Instead, compute perplexity on chosen vs rejected
    dataset = load_dataset(
        "Aratako/iterative-dpo-data-for-SimPO-iter2", split="train"
    )
    split = dataset.train_test_split(test_size=0.05, seed=42)
    eval_data = split["test"].select(range(min(200, len(split["test"]))))

    sft_chosen_better = 0
    sft_total = 0
    for row in eval_data:
        messages_chosen = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["chosen"]},
        ]
        messages_rejected = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["rejected"]},
        ]
        text_chosen = tokenizer.apply_chat_template(messages_chosen, tokenize=False)
        text_rejected = tokenizer.apply_chat_template(messages_rejected, tokenize=False)

        chosen_ll = 0
        rejected_ll = 0
        for text, label in [(text_chosen, "chosen"), (text_rejected, "rejected")]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(ref_model.device)
            with torch.no_grad():
                logits = ref_model(**inputs).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs["input_ids"][:, 1:]
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            avg_ll = token_log_probs.mean().item()
            if label == "chosen":
                chosen_ll = avg_ll
            else:
                rejected_ll = avg_ll

        if chosen_ll > rejected_ll:
            sft_chosen_better += 1
        sft_total += 1
        if sft_total % 50 == 0:
            print(f"  Progress: {sft_total}/{len(eval_data)}, "
                  f"accuracy: {sft_chosen_better/sft_total:.4f}")

    sft_accuracy = sft_chosen_better / sft_total

    # Now DPO model
    del ref_model
    torch.cuda.empty_cache()

    print("\nDPOモデルのreward accuracy計算中...")
    model_dpo, _ = load_model(SFT_MODEL_PATH, DPO_ADAPTER_PATH)

    dpo_chosen_better = 0
    dpo_total = 0
    for row in eval_data:
        messages_chosen = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["chosen"]},
        ]
        messages_rejected = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["rejected"]},
        ]
        text_chosen = tokenizer.apply_chat_template(messages_chosen, tokenize=False)
        text_rejected = tokenizer.apply_chat_template(messages_rejected, tokenize=False)

        chosen_ll = 0
        rejected_ll = 0
        for text, label in [(text_chosen, "chosen"), (text_rejected, "rejected")]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model_dpo.device)
            with torch.no_grad():
                logits = model_dpo(**inputs).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs["input_ids"][:, 1:]
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            avg_ll = token_log_probs.mean().item()
            if label == "chosen":
                chosen_ll = avg_ll
            else:
                rejected_ll = avg_ll

        if chosen_ll > rejected_ll:
            dpo_chosen_better += 1
        dpo_total += 1
        if dpo_total % 50 == 0:
            print(f"  Progress: {dpo_total}/{len(eval_data)}, "
                  f"accuracy: {dpo_chosen_better/dpo_total:.4f}")

    dpo_accuracy = dpo_chosen_better / dpo_total

    # Summary
    print("\n" + "=" * 80)
    print("評価結果サマリー")
    print("=" * 80)

    print(f"\n{'指標':<30} {'SFTモデル(DPO前)':<20} {'DPOモデル(DPO後)':<20}")
    print(f"{'─' * 70}")
    print(f"{'Chosen選好精度 (eval 200件)':<30} {sft_accuracy:<20.4f} {dpo_accuracy:<20.4f}")

    avg_sft_len = sum(r["sft_length"] for r in results) / len(results)
    avg_dpo_len = sum(r["dpo_length"] for r in results) / len(results)
    print(f"{'平均応答長 (文字数)':<30} {avg_sft_len:<20.1f} {avg_dpo_len:<20.1f}")

    # Save full results
    output_path = "/home/ubuntu/Llama_SFT/eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "sft_chosen_accuracy": sft_accuracy,
            "dpo_chosen_accuracy": dpo_accuracy,
            "avg_sft_response_length": avg_sft_len,
            "avg_dpo_response_length": avg_dpo_len,
            "generation_comparisons": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n詳細結果を保存: {output_path}")

    del model_dpo
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
