import re
import torch

CHOICE_LABELS = ["A","B","C","D","E"]
ANSWER_RE = re.compile(r"\[ANSWER\]\s*([ABCDE])", re.IGNORECASE)


def extract_answer(text):
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


def evaluate_jcommonsenseqa(model, tokenizer, dataset, device="cuda"):

    model.eval()
    correct = 0
    total = 0

    for ex in dataset:

        # ===== prompt作成 =====
        question = ex["question"]
        choices = [ex[f"choice{i}"] for i in range(5)]

        choices_text = "\n".join(
            f"{CHOICE_LABELS[i]}. {choices[i]}" for i in range(5)
        )

        prompt = (
            "以下の質問に対して最も適切な選択肢を1つ選び、"
            "最後に必ず [ANSWER] X の形式で答えてください。\n\n"
            f"[QUESTION]\n{question}\n\n"
            f"[CHOICES]\n{choices_text}\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # ===== generation =====
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )

        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        pred = extract_answer(gen)
        gold = CHOICE_LABELS[ex["label"]]

        if pred == gold:
            correct += 1

        total += 1

    acc = correct / total
    print("JCommonsenseQA Accuracy:", acc)
    return acc

