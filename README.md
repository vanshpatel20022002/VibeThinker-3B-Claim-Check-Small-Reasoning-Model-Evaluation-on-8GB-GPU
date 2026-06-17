# VibeThinker-3B Claim Check

This project is an independent, small-scale evaluation of **VibeThinker-3B**, a 3B reasoning model focused on verifiable reasoning tasks.

The goal is not to fully reproduce the paper. Instead, this repo checks a few claims in a practical local setup using an **RTX 4070 Laptop GPU with 8GB VRAM**.

## What This Project Tests

- Can VibeThinker-3B run locally on an 8GB GPU?
- Can it solve simple verifiable math problems?
- Can we extract final boxed answers automatically?
- Can we track accuracy and latency in a reproducible way?

## Current Status

- GPU setup verified
- VibeThinker-3B smoke test completed
- Basic math evaluation harness added
- Results saved to `results/math_basic_results.csv`

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install PyTorch with CUDA support:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install project dependencies:

```powershell
python -m pip install transformers accelerate pandas tqdm
```

## Run GPU Check

```powershell
python scripts\check_gpu.py
```

## Run Smoke Test

```powershell
python scripts\smoke_test_model.py
```

## Run Basic Math Evaluation

```powershell
python scripts\run_math_eval.py
```

## Project Structure

```text
evals/
  math_basic.jsonl

results/
  math_basic_results.csv

scripts/
  check_gpu.py
  smoke_test_model.py
  run_math_eval.py

src/
  .gitkeep
```

## Notes

This is a lightweight claim-check project. The results here should not be treated as a full reproduction of the official VibeThinker-3B benchmark numbers.
