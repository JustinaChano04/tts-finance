"""Independent sampling strategy: N samples, majority vote among executed answers."""

from __future__ import annotations

from models.local_llm import LLM
from strategies.base import (
    BuildPrompt,
    InferenceResult,
    InferenceStrategy,
    Solution,
    generate_and_verify,
)
from tasks.base import FinancialQuestion
from verifiers.base import Verifier
from verifiers.matching import cluster_answers


def majority_vote(trajectories: list[Solution], tolerance: float) -> float | None:
    """Clusters successfully-executed answers within `tolerance` and returns
    the mean of the largest cluster. `None` if nothing executed successfully.
    """
    answers = [
        s.verification.answer
        for s in trajectories
        if s.verification
        and s.verification.success
        and s.verification.answer is not None
    ]
    if not answers:
        return None

    best = max(cluster_answers(answers, tolerance), key=len)
    return sum(best) / len(best)


class SamplingStrategy(InferenceStrategy):
    def __init__(
        self,
        num_samples: int = 4,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        tolerance: float = 0.01,
    ):
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tolerance = tolerance

    def run(
        self,
        question: FinancialQuestion,
        model: LLM,
        verifier: Verifier,
        build_prompt: BuildPrompt,
    ) -> InferenceResult:
        prompt = build_prompt(question)
        trajectories = [
            generate_and_verify(
                question,
                model,
                verifier,
                prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            for _ in range(self.num_samples)
        ]

        final_answer = majority_vote(trajectories, self.tolerance)

        return InferenceResult(
            question_id=question.id,
            final_answer=final_answer,
            correct=verifier.matches(final_answer, question),
            gold_answer=question.gold_answer,
            trajectories=trajectories,
            model_calls=len(trajectories),
            input_tokens=sum(s.generation.input_tokens for s in trajectories),
            output_tokens=sum(s.generation.output_tokens for s in trajectories),
            total_tokens=sum(
                s.generation.input_tokens + s.generation.output_tokens
                for s in trajectories
            ),
            latency=sum(s.generation.latency for s in trajectories),
        )
