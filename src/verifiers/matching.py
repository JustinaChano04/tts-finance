"""Numeric answer comparison shared by the verifier, majority vote, and diagnostics.

Keeping this in one place means the verifier's notion of "correct" and the
vote's notion of "same answer" cannot drift apart.

Match modes:
  * "strict": answer must be within tolerance of gold.
  * "percent_scale": also accept answer == gold * 100 or gold / 100. FinQA's
    own gold labels are inconsistent about percentages (`24.69%` is stored
    as 24.69, `93.5%` as 0.935), so a model cannot know which scale a given
    question expects. Only the 100x factor is forgiven, never units
    (thousands / millions).
"""

from __future__ import annotations

from typing import Literal

MatchMode = Literal["strict", "percent_scale"]

PERCENT_FACTORS = (100.0, 0.01)
UNIT_FACTORS = (1_000.0, 0.001, 1_000_000.0, 0.000_001)


def within_tolerance(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(abs(b), 1.0)


def matches_gold(
    answer: float | None,
    gold: float | None,
    tolerance: float,
    mode: MatchMode = "strict",
) -> bool:
    if answer is None or gold is None:
        return False
    if within_tolerance(answer, gold, tolerance):
        return True
    if mode == "percent_scale":
        return any(within_tolerance(answer, gold * k, tolerance) for k in PERCENT_FACTORS)
    return False


def cluster_answers(answers: list[float], tolerance: float) -> list[list[float]]:
    """Greedy one-pass clustering over sorted answers: each answer joins the
    first cluster whose most recent member is within tolerance. A chain of
    near-equal values can therefore merge even if its endpoints are further
    apart than tolerance.
    """
    clusters: list[list[float]] = []
    for answer in sorted(answers):
        for cluster in clusters:
            if within_tolerance(answer, cluster[-1], tolerance):
                cluster.append(answer)
                break
        else:
            clusters.append([answer])
    return clusters
