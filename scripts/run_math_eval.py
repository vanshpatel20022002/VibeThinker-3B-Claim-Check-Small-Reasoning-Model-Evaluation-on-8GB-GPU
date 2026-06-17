import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "WeiboAI/VibeThinker-3B"
EVAL_PATH = Path("evals/math_basic.jsonl")
RESULTS_PATH = Path("results/math_basic_results.csv")


def extract_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()

    # fallback: use the last number in the response
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1].strip() if numbers else ""


def load_examples(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def main():
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    rows = []

    for ex in load_examples(EVAL_PATH):
        messages = [{"role": "user", "content": build_prompt(ex["question"])}]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_sec = round(time.time() - start, 2)

        new_tokens = output[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        prediction = extract_boxed_answer(response)
        correct = prediction == ex["answer"]

        print(f"{ex['id']}: predicted={prediction}, expected={ex['answer']}, correct={correct}")

        rows.append({
            "id": ex["id"],
            "question": ex["question"],
            "expected": ex["answer"],
            "prediction": prediction,
            "correct": correct,
            "latency_sec": latency_sec,
            "response": response,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)

    accuracy = df["correct"].mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
