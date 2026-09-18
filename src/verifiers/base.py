"""Verifier interface: executes/checks a generated solution against a task."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tasks.base import FinancialQuestion


@dataclass
class VerificationResult:
    success: bool
    answer: float | None
    correct: bool
    execution_error: str | None = None
    stdout: str | None = None


class Verifier(ABC):
    @abstractmethod
    def verify(self, code: str, question: FinancialQuestion) -> VerificationResult:
        ...
