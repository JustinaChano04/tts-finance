<!-- Generated: 2026-09-18 | Updated: 2026-09-18 -->

# tts-finance

## Purpose

A small research framework for testing whether **test-time / inference-time scaling** (spending more inference compute per question — more samples, longer search, more verification — instead of a bigger model) improves financial reasoning accuracy. The first experiment runs a local LLM on [FinQA](https://arxiv.org/abs/2109.00122): the model writes a Python program per question, the program is executed, and answers are combined by majority vote across N independent samples (N = 1, 2, 4, 8, 16).

`README.md` has the full human-facing story (motivation, install, model config, running experiments, roadmap). This file is the machine-facing map: where things live, how they connect, and the parts that aren't obvious from reading one file at a time.

## Layout

```text
src/
├── models/local_llm.py      # LLM interface + local Transformers backend
├── tasks/{base,finqa}.py    # Task interface, FinancialQuestion, FinQA loader + prompt
├── strategies/{base,greedy,sampling}.py  # InferenceStrategy interface, N=1 / N-sample+vote
├── verifiers/{base,python_executor,matching}.py  # Verifier interface, sandboxed execution, answer comparison
├── evaluation/{metrics,evaluator,diagnostics}.py # Strategy-agnostic metrics + failure tagging
└── runner.py                # config -> model -> task -> strategy -> results

configs/baseline.yaml        # experiment config (model/dataset/strategy/verifier/output)
experiments/run_baseline.py  # CLI entrypoint
results/                     # one JSON file per experiment run (gitignored)
analysis/initial_analysis.ipynb  # loads results/, plots accuracy vs. compute
tests/                       # pytest suite, no GPU required
data/                        # empty — see "Where the data actually lives" below
```

The architecture is swappable at four seams, each an ABC with one method: a new **strategy** implements `InferenceStrategy.run` (`strategies/`), a new **verifier** implements `Verifier.verify` (`verifiers/`), a new **dataset** implements `Task.load` + `Task.build_prompt` (`tasks/`), a new **model backend** implements `LLM.generate` (`models/`). `Evaluator` only ever consumes `InferenceResult` objects and doesn't know which strategy produced them — don't couple these layers to each other's internals. New datasets/strategies are registered in `runner.py`'s `build_task()` / `build_strategy()` dispatchers.

## Where the data actually lives

`FinQATask.load()` (`src/tasks/finqa.py`) streams `wujian123/finqa` from the Hugging Face Hub and caches it to `~/.cache/huggingface` on first use — **there is no local copy in `data/`** (it's an empty placeholder, `.gitkeep` only). To inspect real rows, instantiate `FinQATask(split="validation", max_examples=3).load()` rather than looking for a file in this repo.

This is a Parquet **mirror** of the original FinQA release, used because the original `dreamerdeo/finqa` / `ibm-research/finqa` repos rely on a HF "dataset script" loader that current `datasets` versions refuse to execute. The row schema (`pre_text`, `post_text`, `table`, `qa.program`, `qa.exe_ans`) matches the paper.

**Gold labels mix percent scales.** `_parse_gold_answer` prefers `qa["exe_ans"]` (the executed gold program's output), but FinQA is inconsistent about percentages: `24.69%` is stored as `24.69` while `93.5%` is stored as `0.935`. No prompt instruction can make the model match both, which is why scoring defaults to `match_mode: percent_scale` (see below). `exe_ans` also occasionally disagrees with the human `answer` text (e.g. `19.2` vs `18.6`); `exe_ans` is treated as ground truth.

`results/*.json` (one file per experiment run) is the only data this repo *generates* itself — see "Results file shape" below.

## The generate → parse → verify contract

`tasks/finqa.py`'s `PROMPT_TEMPLATE` fixes the model's output format: `REASONING:` / fenced ` ```python ` `CODE:` / `ANSWER:`. `strategies/base.py::parse_model_response` regex-parses that exact shape (code comes from the first fenced block *after* `CODE:`, with any language tag stripped) and is deliberately lenient — it returns `None` for any section it can't find rather than raising, since malformed output is an expected, analysis-relevant outcome, not an error. Changing the prompt template without updating the parser regexes (or vice versa) breaks parsing silently.

The extracted `code` must assign its result to a variable literally named `answer` — this is enforced by `verifiers/python_executor.py`'s execution wrapper, not the parser, so the contract spans two files. The prompt's "Rules for the code" (decimals for percentages, keep table units, only `import math`) are likewise enforced or scored elsewhere; keep them in sync with the executor's allowed subset.

**Error attribution.** Every failed sample carries `verification.error_type` (`no_code`, `truncated`, `syntax_error`, `safety_reject`, `runtime_error`, `timeout`, `no_answer_var`, `non_numeric`, `harness_error`) and `error_source` (`model` or `harness`). `harness_error` means our pipeline failed, not the model. `truncated` is decided in `strategies/base.py::generate_and_verify` from `GenerationResult.truncated` or an unclosed fence. When adding a failure path, give it a type rather than only a message string; `evaluation/diagnostics.py` tags samples from `error_type`.

## Verifier security model

`PythonExecutorVerifier` (`src/verifiers/python_executor.py`) is **defense-in-depth for a research prototype, not a hardened sandbox**:
1. Static AST check (`check_code_safety`) rejects imports other than `math`, dunder attribute access, dunder names other than `__name__`, and calls to `open`/`eval`/`exec`/`compile`/`__import__`/`input`.
2. Execution runs in a separate subprocess (`python -I -S`, empty `env={}`, restricted `__builtins__` whose `__import__` only returns `math`, wall-clock timeout). The result comes back through a file in the temp dir, not stdout, so the solution's own `print`s can't corrupt it.
3. On POSIX only, CPU time (5s) and address space (512MB) are additionally capped via `resource.setrlimit`.

It does not stop all resource exhaustion (e.g. a fork bomb where `RLIMIT_NPROC` isn't enforced) and relies on the AST check rather than OS-level blocking. **Don't point this at adversarial input without further isolation (container/VM).** Keep the module docstring in sync if you touch the safety checks.

Answer comparison lives in one place, `verifiers/matching.py`: `within_tolerance` (`tolerance * max(abs(gold), 1.0)`), `matches_gold` (with `match_mode` `strict` or `percent_scale`, which also accepts gold×100 / gold÷100 but never unit factors), and `cluster_answers`. The verifier, `majority_vote`, and `evaluation/diagnostics.py` all use it; strategies score an aggregated answer via `Verifier.matches` so they inherit the configured `match_mode`. Note the clustering is a simple greedy one-pass sort, not true agglomerative clustering, so a chain of near-equal values can transitively merge even if the endpoints exceed tolerance from each other.

## Results file shape

Each `results/<experiment_id>.json` (never overwritten — `experiment_id` includes a uuid suffix):

```json
{
  "experiment_id": "sampling_n4_...", "timestamp": "...", "model": "...",
  "dataset": "finqa:validation", "strategy": "sampling", "num_samples": 4, "temperature": 0.7,
  "max_new_tokens": 512, "verifier": {"timeout_seconds": 5.0, "tolerance": 0.01, "match_mode": "percent_scale"},
  "metrics": {"accuracy": 0.0, "average_tokens": 0.0, "average_latency": 0.0, "average_model_calls": 0.0, "num_examples": 0,
              "diagnostics": {"accuracy_strict": 0.0, "accuracy_percent_scale": 0.0, "pass_at_n": 0.0, "sample_accuracy": 0.0,
                              "harness_error_samples": 0, "failed_but_stated_correct": 0, "sample_tags": {}, "question_tags": {}}},
  "examples": [ /* one serialized InferenceResult per question, via dataclasses.asdict() */ ]
}
```

`analysis/initial_analysis.ipynb` (`RESULTS_DIR = Path("../results")`, relative to the notebook) reads this shape directly — renaming any top-level or `metrics` key means updating the notebook's `summary`/`examples_df` construction too. Its disagreement/`all_failed` analysis only looks at trajectories where `verification.success == True`. Each example also stores `gold_answer`; results files written before it (and before `error_type`) can't be fully diagnosed and should be regenerated rather than compared with new runs.

## For AI Agents

### Working in this repo
- Config (`configs/*.yaml`, schema in `runner.py`'s pydantic models) is the source of truth for experiment parameters — don't hard-code them in source. Only `model` and `strategy` are required blocks; `dataset`/`verifier`/`output` fall back to defaults.
- To sweep N for the accuracy-vs-compute curve, change only `strategy.num_samples` (or `strategy.name: greedy` for N=1) and keep `dataset.max_examples` fixed across the sweep.
- Out of scope per the roadmap in `README.md`: MCTS, multi-agent frameworks, PRMs, experiment tracking infra (W&B/MLflow), distributed/cloud inference, databases.

### Testing
```bash
pytest tests/ -v                        # full suite, no GPU required
pytest tests/ -v -k "not finqa_task"     # skip the one test needing network access
```
Only `test_finqa_task.py` downloads real data from the Hub; everything else runs offline via `ScriptedLLM` (a fake `LLM`, in `test_sampling_strategy.py`) or fake dataclass builders (`make_result`, `make_question`).

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
