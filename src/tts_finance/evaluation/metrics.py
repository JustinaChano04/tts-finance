"""Pure metric functions over a list of InferenceResult. No strategy knowledge."""

from __future__ import annotations

from tts_finance.strategies.base import InferenceResult


def accuracy(results: list[InferenceResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)


def average_tokens(results: list[InferenceResult]) -> float:
    if not results:
        return 0.0
    return sum(r.total_tokens for r in results) / len(results)


def average_latency(results: list[InferenceResult]) -> float:
    if not results:
        return 0.0
    return sum(r.latency for r in results) / len(results)


def average_model_calls(results: list[InferenceResult]) -> float:
    if not results:
        return 0.0
    return sum(r.model_calls for r in results) / len(results)
