# tts-finance

A small, clean research framework for experimenting with **test-time / inference-time scaling** applied to financial reasoning.

## Why test-time scaling

Test-time scaling — spending more inference compute per question (more samples, longer search, more verification) instead of a bigger model — has produced large accuracy gains on math and code benchmarks. Financial reasoning shares their structure: a question grounded in numeric context (a table, prose) where the reasoning can be expressed as an executable program and checked automatically. This repo is set up to measure whether that same lever works for finance.

**Current research question:**

> Does generating multiple executable reasoning paths improve financial reasoning accuracy as inference compute increases?

The first experiment runs a local LLM on [FinQA](https://arxiv.org/abs/2109.00122), asking it to produce a Python program per question, executing that program, and voting across N independent samples (N = 1, 2, 4, 8, 16) to see how accuracy moves as a function of compute.

## Repository structure

```text
tts-finance/
├── src/
│   ├── models/
│   │   └── local_llm.py         # LLM interface + local Transformers backend
│   ├── tasks/
│   │   ├── base.py              # Task interface, FinancialQuestion
│   │   └── finqa.py             # FinQA loader + prompt construction
│   ├── strategies/
│   │   ├── base.py              # InferenceStrategy interface, response parsing
│   │   ├── greedy.py            # N=1 baseline
│   │   └── sampling.py          # N samples, majority vote
│   ├── verifiers/
│   │   ├── base.py              # Verifier interface
│   │   └── python_executor.py   # sandboxed Python execution + comparison
│   ├── evaluation/
│   │   ├── metrics.py           # pure metric functions
│   │   └── evaluator.py         # strategy-agnostic aggregation
│   └── runner.py                # config -> model -> task -> strategy -> results
│
├── configs/
│   └── baseline.yaml            # experiment configuration
├── experiments/
│   └── run_baseline.py          # CLI entrypoint
├── results/                     # one JSON file per experiment run (gitignored)
├── analysis/
│   └── initial_analysis.ipynb   # loads results/, plots accuracy vs. compute
└── tests/                       # pytest suite -- no GPU required
```

Every piece is swappable independently: a new strategy (best-of-N, search, adaptive compute) only needs to implement `InferenceStrategy.run`; a new verifier only needs `Verifier.verify`; a new dataset only needs `Task.load` + `Task.build_prompt`. The evaluator consumes `InferenceResult` objects and never knows which strategy produced them.

## Installation

Requires Python 3.11+ (this repo was set up against 3.12 via Homebrew — `brew install python@3.12` — since macOS ships 3.9 by default).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## The FinQA dataset

`FinQATask` loads [`wujian123/finqa`](https://huggingface.co/datasets/wujian123/finqa) from the Hugging Face Hub, a Parquet mirror of the original FinQA release with the paper's schema (`pre_text`, `post_text`, `table`, `qa.program`, `qa.exe_ans`). The original `dreamerdeo/finqa` and `ibm-research/finqa` repos rely on a Hugging Face "dataset script" loader that current versions of the `datasets` library refuse to execute — this mirror avoids that without changing the data.

The dataset downloads and caches automatically (to `~/.cache/huggingface`) the first time it's used — no manual download step needed.

## Configuring the local model

Model, device, and dtype are set in the YAML config, never hard-coded:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  device: "auto"   # -> mps on Apple Silicon, cuda if available, else cpu
  dtype: "auto"    # -> bfloat16 on mps/cuda, float32 on cpu
  temperature: 0.7
  max_new_tokens: 512
```

`Qwen2.5-3B-Instruct` (~6GB in bf16) was chosen as the default for a 16GB-unified-memory Apple Silicon machine. If you hit an out-of-memory error, drop to a smaller model:

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
```

Swapping in a different backend later (vLLM, a hosted API) means implementing `LLM.generate` in `models/` — strategy and runner code don't change.

## Running the first experiment

```bash
python experiments/run_baseline.py --config configs/baseline.yaml
```

This loads FinQA, loads the local model, runs the configured strategy over every question, executes and verifies each generated program, computes accuracy/token/latency metrics, and writes one timestamped JSON file to `results/` (existing result files are never overwritten).

To sweep N, change one line in the config and rerun — nothing else needs to change:

```yaml
strategy:
  name: sampling
  num_samples: 8   # was 4
```

Sweep suggestion for the accuracy-vs-compute curve: N = 1 (use `strategy.name: greedy`), then `sampling` with `num_samples` = 2, 4, 8, 16, all against the same `dataset.max_examples` subset.

## Running the tests

```bash
pytest tests/ -v
```

No GPU is required. `test_finqa_task.py` downloads a few FinQA examples from the Hub, so it needs network access; everything else (parsing, the verifier's sandboxed execution, sampling strategy logic, evaluation metrics) runs fully offline with a scripted fake `LLM`.

## Analyzing results

```bash
jupyter lab analysis/initial_analysis.ipynb
```

Run experiments at several values of N first (`results/` needs more than one JSON file), then the notebook plots accuracy vs. N, accuracy vs. total tokens, a comparison table across experiments, and surfaces specific trajectories: N=1 failures that N=8 fixes, examples where every sample failed, samples that disagreed, and cases where execution caught a stated-but-wrong answer.

## Security note on the Python verifier

`PythonExecutorVerifier` executes model-generated code in a separate subprocess (`python -I -S`, empty environment, restricted builtins, no `import`, no `open`/`eval`/`exec`, a wall-clock timeout, and — on POSIX — CPU/memory `rlimit`s). This is defense-in-depth for a research prototype, **not a hardened sandbox**. Don't point it at adversarial input without further isolation (a container or VM) — see the module docstring in `src/verifiers/python_executor.py` for specifics.

## Research roadmap

```text
Phase 1: Independent sampling               <- current
Phase 2: Better selection (learned/verifier-weighted voting)
Phase 3: Execution-guided search
Phase 4: Adaptive compute allocation
Phase 5: More realistic financial documents
Phase 6: Additional datasets
```

Deliberately out of scope for this version: MCTS, agent/multi-agent frameworks, PRMs, experiment tracking infra (W&B/MLflow), distributed or cloud inference, databases. These get added if and when an experiment's results motivate them.
