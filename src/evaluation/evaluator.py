"""Aggregates InferenceResults into a metrics dict. Strategy-agnostic."""

from __future__ import annotations

from evaluation import diagnostics, metrics
from strategies.base import InferenceResult
from verifiers.matching import MatchMode


class Evaluator:
    def __init__(self, tolerance: float = 0.01, match_mode: MatchMode = "strict"):
        self.tolerance = tolerance
        self.match_mode = match_mode

    def evaluate(self, results: list[InferenceResult]) -> dict:
        return {
            "accuracy": metrics.accuracy(results),
            "average_tokens": metrics.average_tokens(results),
            "average_latency": metrics.average_latency(results),
            "average_model_calls": metrics.average_model_calls(results),
            "num_examples": len(results),
            "diagnostics": diagnostics.summarize(results, self.tolerance, self.match_mode),
        }
