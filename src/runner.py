"""Loads a YAML experiment config and runs it end to end.

load config -> load model -> load task -> load strategy -> run every
example -> verify -> compute metrics -> save one JSON file per experiment.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import uuid
from pathlib import Path

import yaml
from pydantic import BaseModel

from evaluation.evaluator import Evaluator
from models.local_llm import LocalLLM
from strategies.base import InferenceResult, InferenceStrategy
from strategies.greedy import GreedyStrategy
from strategies.sampling import SamplingStrategy
from tasks.base import Task
from tasks.finqa import FinQATask
from verifiers.matching import MatchMode
from verifiers.python_executor import PythonExecutorVerifier


class ModelConfig(BaseModel):
    name: str
    device: str = "auto"
    dtype: str = "auto"
    temperature: float = 0.7
    max_new_tokens: int = 512


class DatasetConfig(BaseModel):
    name: str = "finqa"
    split: str = "validation"
    max_examples: int | None = None


class StrategyConfig(BaseModel):
    name: str
    num_samples: int = 1


class VerifierConfig(BaseModel):
    timeout_seconds: float = 5.0
    tolerance: float = 0.01
    # "percent_scale" also accepts answer == gold*100 or gold/100, because
    # FinQA's gold labels mix percent and decimal scales. See verifiers/matching.py.
    match_mode: MatchMode = "percent_scale"


class OutputConfig(BaseModel):
    results_dir: str = "results/"


class ExperimentConfig(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig = DatasetConfig()
    strategy: StrategyConfig
    verifier: VerifierConfig = VerifierConfig()
    output: OutputConfig = OutputConfig()


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(**raw)


def build_task(cfg: DatasetConfig) -> Task:
    if cfg.name == "finqa":
        return FinQATask(split=cfg.split, max_examples=cfg.max_examples)
    raise ValueError(f"unknown dataset: {cfg.name}")


def build_strategy(
    strategy_cfg: StrategyConfig, model_cfg: ModelConfig, verifier_cfg: VerifierConfig
) -> InferenceStrategy:
    if strategy_cfg.name == "greedy":
        return GreedyStrategy(max_new_tokens=model_cfg.max_new_tokens)
    if strategy_cfg.name == "sampling":
        return SamplingStrategy(
            num_samples=strategy_cfg.num_samples,
            temperature=model_cfg.temperature,
            max_new_tokens=model_cfg.max_new_tokens,
            tolerance=verifier_cfg.tolerance,
        )
    raise ValueError(f"unknown strategy: {strategy_cfg.name}")


def _result_to_dict(result: InferenceResult) -> dict:
    return dataclasses.asdict(result)


def run_experiment(config_path: str) -> Path:
    cfg = load_config(config_path)

    print(
        f"Loading model {cfg.model.name} (device={cfg.model.device}, dtype={cfg.model.dtype})..."
    )
    model = LocalLLM(cfg.model.name, device=cfg.model.device, dtype=cfg.model.dtype)

    task = build_task(cfg.dataset)
    print(
        f"Loading dataset {cfg.dataset.name}:{cfg.dataset.split} (max_examples={cfg.dataset.max_examples})..."
    )
    questions = task.load()
    print(f"Loaded {len(questions)} questions.")

    verifier = PythonExecutorVerifier(
        timeout_seconds=cfg.verifier.timeout_seconds,
        tolerance=cfg.verifier.tolerance,
        match_mode=cfg.verifier.match_mode,
    )
    strategy = build_strategy(cfg.strategy, cfg.model, cfg.verifier)

    results: list[InferenceResult] = []
    for i, question in enumerate(questions, start=1):
        result = strategy.run(question, model, verifier, task.build_prompt)
        results.append(result)
        status = "correct" if result.correct else "incorrect"
        print(
            f"[{i}/{len(questions)}] {question.id}: {status} (final_answer={result.final_answer})"
        )

    metrics = Evaluator(
        tolerance=cfg.verifier.tolerance, match_mode=cfg.verifier.match_mode
    ).evaluate(results)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    experiment_id = (
        f"{cfg.strategy.name}_n{cfg.strategy.num_samples}_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )

    output = {
        "experiment_id": experiment_id,
        "timestamp": timestamp.isoformat(),
        "model": cfg.model.name,
        "dataset": f"{cfg.dataset.name}:{cfg.dataset.split}",
        "strategy": cfg.strategy.name,
        "num_samples": cfg.strategy.num_samples,
        "temperature": cfg.model.temperature,
        "max_new_tokens": cfg.model.max_new_tokens,
        "verifier": cfg.verifier.model_dump(),
        "metrics": metrics,
        "examples": [_result_to_dict(r) for r in results],
    }

    results_dir = Path(cfg.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{experiment_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved results to {out_path}")

    return out_path
