import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_ID = "WeiboAI/VibeThinker-3B"


def extract_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()

    # fallback: use the last number in the response
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1].strip() if numbers else ""


def normalize_answer(answer: str) -> str:
    """Normalize simple numeric answers for fair comparison.

    This removes formatting differences like dollar signs, commas,
    whitespace, and LaTeX percent signs. It does not attempt symbolic math.
    """
    text = str(answer).strip().lower()
    text = text.replace("\\%", "")
    text = text.replace("%", "")
    text = text.replace("$", "")
    text = text.replace("\\$", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)

    try:
        value = float(text)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return text


def load_examples(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small-model math evaluation.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID to evaluate.",
    )
    parser.add_argument(
        "--eval-file",
        default="evals/math_basic.jsonl",
        help="Path to the JSONL evaluation file.",
    )
    parser.add_argument(
        "--output-file",
        default="results/math_basic_results.csv",
        help="Path where CSV results will be saved.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens generated per question.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_path = Path(args.eval_file)
    results_path = Path(args.output_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_id}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    rows = []

    for ex in load_examples(eval_path):
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
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_sec = round(time.time() - start, 2)

        new_tokens = output[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        prediction = extract_boxed_answer(response)
        expected_normalized = normalize_answer(ex["answer"])
        prediction_normalized = normalize_answer(prediction)
        correct = prediction_normalized == expected_normalized

        print(
            f"{ex['id']}: predicted={prediction} "
            f"expected={ex['answer']} correct={correct}"
        )

        rows.append({
            "model_id": args.model_id,
            "id": ex["id"],
            "question": ex["question"],
            "expected": ex["answer"],
            "prediction": prediction,
            "expected_normalized": expected_normalized,
            "prediction_normalized": prediction_normalized,
            "correct": correct,
            "latency_sec": latency_sec,
            "response": response,
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)

    accuracy = df["correct"].mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
