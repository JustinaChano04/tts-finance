"""Verifier interface: executes/checks a generated solution against a task."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from tasks.base import FinancialQuestion
from verifiers.matching import MatchMode

# Why a sample produced no usable answer. Every failure path sets exactly one.
ErrorType = Literal[
    "no_code",  # response had no code block
    "truncated",  # generation hit max_new_tokens before the code block closed
    "syntax_error",  # code does not parse
    "safety_reject",  # code uses something outside the allowed subset
    "runtime_error",  # code raised while running
    "timeout",  # wall-clock or CPU limit exceeded
    "no_answer_var",  # ran, but never assigned `answer`
    "non_numeric",  # `answer` is not a finite number
    "harness_error",  # the executor itself failed; says nothing about the model
]

# "model": the model broke the prompt's contract or wrote failing code.
# "harness": our own pipeline failed; the sample should not count against the model.
ErrorSource = Literal["model", "harness"]


def error_source_for(error_type: ErrorType) -> ErrorSource:
    return "harness" if error_type == "harness_error" else "model"


@dataclass
class VerificationResult:
    success: bool
    answer: float | None
    correct: bool
    execution_error: str | None = None
    stdout: str | None = None
    error_type: ErrorType | None = None
    error_source: ErrorSource | None = None
    # `answer` was never assigned, so the value of the code's final expression was used.
    recovered_from_expression: bool = False

    @classmethod
    def failure(
        cls, error_type: ErrorType, message: str, stdout: str | None = None
    ) -> VerificationResult:
        return cls(
            success=False,
            answer=None,
            correct=False,
            execution_error=message,
            stdout=stdout,
            error_type=error_type,
            error_source=error_source_for(error_type),
        )


class Verifier(ABC):
    match_mode: MatchMode = "strict"

    @abstractmethod
    def verify(self, code: str, question: FinancialQuestion) -> VerificationResult: ...

    @abstractmethod
    def matches(self, answer: float | None, question: FinancialQuestion) -> bool:
        """Whether `answer` counts as correct for `question`. Strategies use
        this to score an aggregated answer (e.g. a vote) the same way the
        verifier scores a single sample.
        """
