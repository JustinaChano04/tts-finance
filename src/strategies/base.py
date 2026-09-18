"""Inference strategy interface and shared response-parsing utilities.

The evaluator only ever consumes `InferenceResult` objects — it does not
know or care whether they came from `GreedyStrategy`, `SamplingStrategy`,
or anything added later (best-of-N, search, adaptive compute, ...).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from models.local_llm import GenerationResult, LLM
from tasks.base import FinancialQuestion
from verifiers.base import VerificationResult, Verifier

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"REASONING:\s*(.*?)(?=CODE:|$)", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"ANSWER:\s*(.*)", re.DOTALL | re.IGNORECASE)


def parse_model_response(text: str) -> tuple[str | None, str | None, str | None]:
    """Extracts (reasoning, code, stated_answer) from a raw model response.

    Robust to the model omitting sections or the fence missing a language
    tag; falls back to `None` for anything it can't find rather than raising,
    since malformed responses are an expected (and analysis-relevant) case.
    """
    reasoning_match = _REASONING_RE.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    code_match = _CODE_BLOCK_RE.search(text)
    code = code_match.group(1).strip() if code_match else None

    answer_match = _ANSWER_RE.search(text)
    stated_answer = answer_match.group(1).strip() if answer_match else None

    return reasoning, code, stated_answer


@dataclass
class Solution:
    raw_response: str
    reasoning: str | None
    code: str | None
    stated_answer: str | None
    generation: GenerationResult
    verification: VerificationResult | None = None


@dataclass
class InferenceResult:
    question_id: str
    final_answer: float | None
    correct: bool
    trajectories: list[Solution] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0


BuildPrompt = Callable[[FinancialQuestion], str]


class InferenceStrategy(ABC):
    @abstractmethod
    def run(
        self,
        question: FinancialQuestion,
        model: LLM,
        verifier: Verifier,
        build_prompt: BuildPrompt,
    ) -> InferenceResult:
        ...


def generate_and_verify(
    question: FinancialQuestion,
    model: LLM,
    verifier: Verifier,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> Solution:
    generation = model.generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    reasoning, code, stated_answer = parse_model_response(generation.text)
    verification = verifier.verify(code, question) if code else VerificationResult(
        success=False, answer=None, correct=False,
        execution_error="no code extracted from response",
    )
    return Solution(
        raw_response=generation.text,
        reasoning=reasoning,
        code=code,
        stated_answer=stated_answer,
        generation=generation,
        verification=verification,
    )
