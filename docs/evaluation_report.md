# Evaluation Report

This report summarizes the current small-scale evaluation of VibeThinker-3B on local consumer hardware.

## Goal

The goal of this project is not to reproduce the full VibeThinker-3B paper. Instead, it checks a small subset of the model's reported strengths using a practical local setup.

The focus is on verifiable tasks where outputs can be checked automatically:

- math reasoning with known final answers
- coding tasks with executable unit tests

## Hardware

The evaluation was run locally on:

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- VRAM: 8GB
- PyTorch: CUDA-enabled build
- OS: Windows

## Models Evaluated

| Model | Role |
|---|---|
| WeiboAI/VibeThinker-3B | Target model |
| Qwen/Qwen2.5-Coder-3B-Instruct | Baseline model |

## Evaluation Sets

| Eval set | Description |
|---|---|
| `math_basic.jsonl` | Simple arithmetic sanity-check questions |
| `math_reasoning.jsonl` | Slightly harder word problems involving percentages, proportions, averages, and geometry |
| `coding_basic.jsonl` | Small Python function-generation tasks evaluated with unit tests |

## Current Results

| Eval set | Model | Questions / Tasks | Correct / Passed | Score | Latency range |
|---|---|---:|---:|---:|---:|
| Basic math sanity check | WeiboAI/VibeThinker-3B | 5 | 5 | 100% | 10.72s - 19.71s |
| Math reasoning set | WeiboAI/VibeThinker-3B | 8 | 8 | 100% | 9.30s - 148.78s |
| Math reasoning set | Qwen/Qwen2.5-Coder-3B-Instruct | 8 | 7 | 87.5% | 6.78s - 14.89s |
| Basic coding tasks | WeiboAI/VibeThinker-3B | 5 | 5 | 100% | 12.94s - 22.66s |
| Basic coding tasks | Qwen/Qwen2.5-Coder-3B-Instruct | 5 | 5 | 100% | 1.44s - 2.39s |

## Observations

1. VibeThinker-3B ran successfully on an 8GB GPU for inference.
2. On the small math reasoning set, VibeThinker-3B solved all examples.
3. The Qwen baseline missed one proportional reasoning problem, predicting `30` instead of `10`.
4. On basic coding tasks, both models passed all unit tests.
5. Qwen was much faster on the small coding prompts.
6. VibeThinker-3B often emits `<think>...</think>` reasoning traces before the final answer or code, so the evaluator strips those traces before scoring executable code.
7. Some numeric answers require light normalization before scoring, such as `15` vs `15\%`.

## Limitations

This is a small local evaluation and should not be interpreted as a reproduction of the official VibeThinker-3B benchmark results.

Important limitations:

- The datasets are small and hand-written.
- The benchmark does not cover AIME, LiveCodeBench, HMMT, or official paper datasets.
- Sampling is deterministic with `do_sample=False`.
- Coding tasks are intentionally simple and only use toy unit tests.
- Latency numbers depend on local hardware, model loading behavior, output length, and Windows runtime overhead.
- Generated code execution is only used on controlled toy prompts.

## Next Steps

Potential improvements:

- Add more math questions with varied difficulty.
- Add a HumanEval-style coding subset.
- Track generated token counts and tokens per second.
- Add per-task failure analysis.
- Add support for evaluating quantized model variants.
- Add charts for accuracy and latency comparison.
