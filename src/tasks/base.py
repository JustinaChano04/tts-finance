"""Dataset-independent task interface.

A `Task` owns everything specific to how its dataset is stored and how a
question should be presented to a model. Strategies only ever see
`FinancialQuestion` objects and a `build_prompt` callable — they never touch
dataset internals directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class FinancialQuestion:
    id: str
    question: str
    pre_text: str
    post_text: str
    table: list[list[str]] = field(default_factory=list)
    gold_answer: float | None = None
    gold_program: str = ""


class Task(ABC):
    @abstractmethod
    def load(self) -> list[FinancialQuestion]:
        ...

    @abstractmethod
    def build_prompt(self, question: FinancialQuestion) -> str:
        ...
