# tts-finance

This repo is a framework for examining **test-time scaling** on financial reasoning: how much accuracy can be achieved by spending more inference compute per question (more samples, longer search, more verification) instead of using a bigger model.

For each question, a local LLM reads the table and text from a financial filing and writes a Python program that computes the answer. The program is run in a sandbox, and when you sample several programs per question, their answers are combined by majority vote. Accuracy, token use, and latency are recorded for every run, so you can compare strategies at different compute budgets. Failed samples are tagged with what went wrong (for example a syntax error, a timeout, or a bug in the harness itself).

It supports the [FinQA](https://arxiv.org/abs/2109.00122) dataset, greedy and N-sample voting strategies, and local models through Transformers or MLX. Strategies, verifiers, datasets, and model backends can each be swapped out independently.

## Experimental setup

Every run follows the same procedure: the model writes a program per question, the program is executed, and a strategy turns the samples into one answer. A run is fully defined by its config in `configs/`: strategy and sample count, model, dataset and subset size, and scoring mode. To isolate an effect, change one of these and keep the rest fixed, using the same `output.run_group` so the runs can be compared.

Each run records accuracy, average tokens, average latency, average model calls, and per-sample failure tags. Specific values (which model, which N, how many questions) live in the configs, not in this README.

## Quick start

Requires Python 3.11+ (macOS ships 3.9, so `brew install python@3.12`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python experiments/run_baseline.py --config configs/baseline.yaml
```

The first run downloads FinQA and the model automatically. Results are written to `results/`.

## Running experiments

### One run

```bash
python experiments/run_baseline.py --config configs/baseline.yaml
```

Each run loads the dataset and model, runs the configured strategy on every question, executes and verifies each generated program, and writes one timestamped JSON file to `results/<run_group>/`. Set `output.run_group` in the config to choose the folder. Existing result files are never overwritten.

### Sweeping N

Change one line in the config and rerun:

```yaml
strategy:
  name: sampling
  num_samples: 8   # was 4
```

For the accuracy-vs-compute curve, run N = 1 (`strategy.name: greedy`), then `sampling` with `num_samples` = 2, 4, 8, 16. Keep `dataset.max_examples` the same across the sweep and give every config the same `output.run_group`. Examples are in `configs/validation-20q/` and `configs/validation-200q/`.

### Configuring the model

Model, device, and dtype are set in the YAML config, never hard-coded:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  device: "auto"   # -> mps on Apple Silicon, cuda if available, else cpu
  dtype: "auto"    # -> bfloat16 on mps/cuda, float32 on cpu
  temperature: 0.7
  max_new_tokens: 512
```

| Situation | Setting |
|---|---|
| Default (16GB Apple Silicon, ~6GB in bf16) | `Qwen/Qwen2.5-3B-Instruct` |
| Out of memory | `name: "Qwen/Qwen2.5-1.5B-Instruct"` |
| Larger model on Apple Silicon | `backend: mlx` with a 4-bit build, e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit` (see `configs/validation-200q-7b/`; needs `pip install mlx-lm`) |

To add another backend (vLLM, a hosted API), implement `LLM.generate` in `src/models/`. Strategy and runner code don't change.

### Scoring

Answers are compared in `src/verifiers/matching.py` using the `verifier` block of the config:

```yaml
verifier:
  timeout_seconds: 5
  tolerance: 0.01            # relative tolerance
  match_mode: percent_scale  # or: strict
```

FinQA stores some percentages as `24.69` and others as `0.935`, so no prompt can match both. The default `percent_scale` also accepts an answer that is the gold value ×100 or ÷100. Use `strict` for exact scale matching.

Every failed sample is tagged with an `error_type` (for example `syntax_error`, `timeout`, `truncated`) and an `error_source` (`model` or `harness`), so you can tell model mistakes from pipeline bugs.

## Analyzing results

```bash
jupyter lab analysis/initial_analysis.ipynb
```

Run several values of N first, since the notebook's `RUN_GROUP` folder needs more than one JSON file. The notebook plots accuracy vs. N and accuracy vs. total tokens, and shows a comparison table across runs. It also pulls out specific cases:

- N=1 failures that N=8 fixes
- Questions where every sample failed
- Samples that disagreed
- Cases where execution caught a stated-but-wrong answer

The other notebooks in `analysis/` cover specific run groups (the 200-question set, the 7B vs. 3B comparison, and the fixed-verifier rerun).

## Testing

```bash
pytest tests/ -v                      # full suite
pytest tests/ -v -k "not finqa_task"  # skip the test that needs network access
```

No GPU is required. Only `test_finqa_task.py` downloads data (a few FinQA examples from the Hub). Everything else runs offline with a scripted fake `LLM`.

## How it fits together

```text
config.yaml -> runner.py -> model -> task -> strategy -> verifier -> evaluator -> results/*.json
```

```text
src/
├── models/        LLM interface + backends (local_llm.py: Transformers, mlx_llm.py: MLX)
├── tasks/         Task interface + FinQA loader and prompt
├── strategies/    greedy.py (N=1), sampling.py (N samples + majority vote)
├── verifiers/     sandboxed Python execution (python_executor.py), answer comparison (matching.py)
├── evaluation/    metrics.py, evaluator.py, diagnostics.py (failure tagging)
└── runner.py      wires config -> model -> task -> strategy -> results
configs/           experiment configs (baseline.yaml + one folder per sweep)
experiments/       run_baseline.py (CLI entrypoint)
analysis/          notebooks that read results/
tests/             pytest suite
```

Each piece is swappable on its own:

| To add a new... | Implement |
|---|---|
| Strategy (best-of-N, search, adaptive compute) | `InferenceStrategy.run` |
| Verifier | `Verifier.verify` |
| Dataset | `Task.load` + `Task.build_prompt` |
| Model backend | `LLM.generate` |

The evaluator only consumes `InferenceResult` objects and never knows which strategy produced them.

## The FinQA dataset

`FinQATask` loads [`wujian123/finqa`](https://huggingface.co/datasets/wujian123/finqa) from the Hugging Face Hub. It's a Parquet mirror of the original FinQA release with the paper's schema (`pre_text`, `post_text`, `table`, `qa.program`, `qa.exe_ans`). The original `dreamerdeo/finqa` and `ibm-research/finqa` repos rely on a "dataset script" loader that current `datasets` versions refuse to run, and this mirror avoids that without changing the data.

The data downloads and caches to `~/.cache/huggingface` on first use. No manual download is needed.

## Security note

`PythonExecutorVerifier` runs model-generated code in a separate subprocess with an empty environment, restricted builtins, no `import` (except `math`), no `open`/`eval`/`exec`, a wall-clock timeout, and on POSIX, CPU and memory limits. This is defense-in-depth for a research prototype, **not a hardened sandbox**. Don't point it at adversarial input without further isolation (a container or VM). See the docstring in `src/verifiers/python_executor.py` for specifics.
