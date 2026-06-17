import argparse
import json
import re
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_ID = "WeiboAI/VibeThinker-3B"


def load_examples(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_prompt(task: str, function_name: str) -> str:
    return f"""
You are solving a coding task.

Task:
{task}

Requirements:
- Write Python 3 code.
- Define a function named {function_name}.
- Return only the code.
- Do not include explanations.
""".strip()


def strip_thinking_traces(text: str) -> str:
    """Remove reasoning traces before extracting executable code."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_code(text: str) -> str:
    cleaned_text = strip_thinking_traces(text)

    code_blocks = re.findall(
        r"```(?:python)?\s*(.*?)```",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if code_blocks:
        return code_blocks[-1].strip()

    return cleaned_text.strip()


def run_code_tests(code: str, tests: list[dict], timeout_sec: int = 5):
    test_lines = ["import json"]
    test_lines.append(code)
    test_lines.append("\nresults = []")

    for test in tests:
        expected_json = json.dumps(test["expected"])
        test_lines.append(textwrap.dedent(f"""
        try:
            actual = {test['call']}
            expected = json.loads({expected_json!r})
            results.append({{"call": {test['call']!r}, "actual": actual, "expected": expected, "passed": actual == expected}})
        except Exception as exc:
            results.append({{"call": {test['call']!r}, "actual": repr(exc), "expected": json.loads({expected_json!r}), "passed": False}})
        """))

    test_lines.append("print(json.dumps(results, default=str))")
    full_script = "\n".join(test_lines)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_script)
        temp_path = f.name

    try:
        completed = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        if completed.returncode != 0:
            return False, [], completed.stderr.strip()

        results = json.loads(completed.stdout.strip())
        passed = all(item["passed"] for item in results)
        return passed, results, ""
    except subprocess.TimeoutExpired:
        return False, [], "Timeout while running generated code."
    except Exception as exc:
        return False, [], repr(exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small coding evaluation.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID to evaluate.")
    parser.add_argument("--eval-file", default="evals/coding_basic.jsonl", help="Path to the JSONL coding eval file.")
    parser.add_argument("--output-file", default="results/coding_basic_results.csv", help="Path where CSV results will be saved.")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Maximum number of tokens generated per task.")
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
        prompt = build_prompt(ex["prompt"], ex["function_name"])
        messages = [{"role": "user", "content": prompt}]

        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
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
        code = extract_code(response)
        passed, test_results, error = run_code_tests(code, ex["tests"])

        print(f"{ex['id']}: passed={passed}")

        rows.append({
            "model_id": args.model_id,
            "id": ex["id"],
            "function_name": ex["function_name"],
            "passed": passed,
            "latency_sec": latency_sec,
            "error": error,
            "test_results": json.dumps(test_results),
            "generated_code": code,
            "raw_response": response,
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)

    pass_rate = df["passed"].mean()
    print(f"\nPass rate: {pass_rate:.2%}")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
