"""Failure tagging: why each sample and each question came out the way it did.

Separates four kinds of error that plain accuracy lumps together:
  * model reasoning errors      -> sample `wrong_value`, question `none_correct`
  * code / format failures      -> sample error types (`runtime_error`, ...)
  * scoring artefacts           -> `percent_scale`, `unit_scale`, `sign_flip`, `near_miss`
  * aggregation (voting) losses -> question `vote_lost`, `tie_lost`
plus `harness_error`, which is our pipeline's fault and not the model's.

Strategy-agnostic: works from trajectories and gold answers only.
"""

from __future__ import annotations

import re
from collections import Counter

from strategies.base import InferenceResult, Solution
from verifiers.matching import (
    PERCENT_FACTORS,
    UNIT_FACTORS,
    MatchMode,
    cluster_answers,
    matches_gold,
    within_tolerance,
)

NEAR_MISS_TOLERANCE = 0.05

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?|-?\.\d+")


def sample_tag(solution: Solution, gold: float | None, tolerance: float) -> str:
    """Exactly one tag per sample. Failed samples are tagged with their
    verifier `error_type`; executed samples by how their answer relates to gold.
    """
    v = solution.verification
    if v is None:
        return "no_code"
    if not v.success:
        return v.error_type or "unknown_error"
    if gold is None:
        return "gold_missing"

    answer = v.answer
    if within_tolerance(answer, gold, tolerance):
        return "correct"
    if any(within_tolerance(answer, gold * k, tolerance) for k in PERCENT_FACTORS):
        return "percent_scale"
    if any(within_tolerance(answer, gold * k, tolerance) for k in UNIT_FACTORS):
        return "unit_scale"
    if within_tolerance(answer, -gold, tolerance):
        return "sign_flip"
    if within_tolerance(answer, gold, NEAR_MISS_TOLERANCE):
        return "near_miss"
    return "wrong_value"


def parse_stated_answer(stated: str | None) -> float | None:
    """First number in the model's `ANSWER:` text, ignoring `$`, `,` and `%`.
    Percent signs are dropped rather than converted; compare with
    `percent_scale` matching to absorb the 100x ambiguity.
    """
    if not stated:
        return None
    match = _NUMBER_RE.search(stated.replace(",", "").replace("$", ""))
    return float(match.group()) if match else None


def stated_answer_correct(solution: Solution, gold: float | None, tolerance: float) -> bool:
    return matches_gold(
        parse_stated_answer(solution.stated_answer), gold, tolerance, "percent_scale"
    )


def _executed_answers(result: InferenceResult) -> list[float]:
    return [
        s.verification.answer
        for s in result.trajectories
        if s.verification and s.verification.success and s.verification.answer is not None
    ]


def question_tag(result: InferenceResult, tolerance: float, mode: MatchMode) -> str:
    gold = result.gold_answer
    if gold is None:
        return "gold_missing"
    answers = _executed_answers(result)
    if not answers:
        return "all_failed"
    if result.correct:
        return "vote_correct"

    clusters = cluster_answers(answers, tolerance)
    largest = max(len(c) for c in clusters)
    correct_sizes = [
        len(c) for c in clusters if matches_gold(sum(c) / len(c), gold, tolerance, mode)
    ]
    if not correct_sizes:
        return "none_correct"
    return "tie_lost" if max(correct_sizes) == largest else "vote_lost"


def summarize(results: list[InferenceResult], tolerance: float, mode: MatchMode) -> dict:
    n_questions = len(results)
    sample_tags: Counter[str] = Counter()
    question_tags: Counter[str] = Counter()
    n_samples = n_samples_correct = n_any_correct = n_stated_rescue = 0

    for r in results:
        question_tags[question_tag(r, tolerance, mode)] += 1
        any_correct = False
        for s in r.trajectories:
            n_samples += 1
            sample_tags[sample_tag(s, r.gold_answer, tolerance)] += 1
            v = s.verification
            if v and v.success and matches_gold(v.answer, r.gold_answer, tolerance, mode):
                n_samples_correct += 1
                any_correct = True
            elif not (v and v.success) and stated_answer_correct(s, r.gold_answer, tolerance):
                n_stated_rescue += 1
        n_any_correct += any_correct

    def question_rate(match_mode: MatchMode) -> float:
        if not n_questions:
            return 0.0
        hits = sum(
            matches_gold(r.final_answer, r.gold_answer, tolerance, match_mode)
            for r in results
        )
        return hits / n_questions

    return {
        "accuracy_strict": question_rate("strict"),
        "accuracy_percent_scale": question_rate("percent_scale"),
        "pass_at_n": n_any_correct / n_questions if n_questions else 0.0,
        "sample_accuracy": n_samples_correct / n_samples if n_samples else 0.0,
        "harness_error_samples": sample_tags.get("harness_error", 0),
        "failed_but_stated_correct": n_stated_rescue,
        "sample_tags": dict(sample_tags.most_common()),
        "question_tags": dict(question_tags.most_common()),
    }
