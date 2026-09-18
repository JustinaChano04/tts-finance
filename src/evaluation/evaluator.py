"""Aggregates InferenceResults into a metrics dict. Strategy-agnostic."""

from __future__ import annotations

from evaluation import metrics
from strategies.base import InferenceResult


class Evaluator:
    def evaluate(self, results: list[InferenceResult]) -> dict:
        return {
            "accuracy": metrics.accuracy(results),
            "average_tokens": metrics.average_tokens(results),
            "average_latency": metrics.average_latency(results),
            "average_model_calls": metrics.average_model_calls(results),
            "num_examples": len(results),
        }
