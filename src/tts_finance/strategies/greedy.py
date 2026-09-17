"""Greedy (N=1) baseline strategy: one deterministic reasoning program."""

from __future__ import annotations

from tts_finance.models.local_llm import LLM
from tts_finance.strategies.base import BuildPrompt, InferenceResult, InferenceStrategy, generate_and_verify
from tts_finance.tasks.base import FinancialQuestion
from tts_finance.verifiers.base import Verifier


class GreedyStrategy(InferenceStrategy):
    def __init__(self, max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens

    def run(
        self,
        question: FinancialQuestion,
        model: LLM,
        verifier: Verifier,
        build_prompt: BuildPrompt,
    ) -> InferenceResult:
        prompt = build_prompt(question)
        solution = generate_and_verify(
            question, model, verifier, prompt,
            temperature=0.0, max_new_tokens=self.max_new_tokens,
        )
        verification = solution.verification

        return InferenceResult(
            question_id=question.id,
            final_answer=verification.answer if verification and verification.success else None,
            correct=bool(verification and verification.correct),
            trajectories=[solution],
            model_calls=1,
            input_tokens=solution.generation.input_tokens,
            output_tokens=solution.generation.output_tokens,
            total_tokens=solution.generation.input_tokens + solution.generation.output_tokens,
            latency=solution.generation.latency,
        )
