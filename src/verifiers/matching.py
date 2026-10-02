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

Voting treats answers the same way: in "percent_scale" mode, answers 100x
apart count as one answer. Ties between equally large answer groups go to
the group whose member was sampled first (samples are exchangeable, so this
is an unbiased pick rather than a pull toward small values).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Literal

MatchMode = Literal["strict", "percent_scale"]

PERCENT_FACTORS = (100.0, 0.01)
UNIT_FACTORS = (1_000.0, 0.001, 1_000_000.0, 0.000_001)

# Tolerance is relative to the target; this floor only keeps a target of
# exactly 0 (or float noise around it) from needing an exact match. It must
# stay far below FinQA's small values (0.002 vs 0.2 are different answers).
ABS_FLOOR = 1e-4


def within_tolerance(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(abs(b), ABS_FLOOR)


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


@dataclass(frozen=True)
class VoteGroup:
    size: int  # samples in the group
    value: float  # mean of the group's largest exact-match cluster
    first_index: int  # earliest sample in the group; breaks ties


def _same_up_to_scale(a: float, b: float, tolerance: float) -> bool:
    return any(within_tolerance(a, b * k, tolerance) for k in (1.0, *PERCENT_FACTORS))


def vote_groups(
    answers: list[float], tolerance: float, mode: MatchMode = "strict"
) -> list[VoteGroup]:
    """Groups answers into distinct votes.

    First a greedy one-pass clustering over sorted answers: each answer joins
    the first cluster whose most recent member is within tolerance (so a chain
    of near-equal values can merge even if its endpoints are further apart).
    In "percent_scale" mode, clusters 100x apart are then merged into one
    group, anchored on the largest cluster. A group's `value` comes from its
    largest cluster, never a mean across scales.
    """
    clusters: list[list[int]] = []
    for i in sorted(range(len(answers)), key=lambda i: answers[i]):
        for cluster in clusters:
            if within_tolerance(answers[i], answers[cluster[-1]], tolerance):
                cluster.append(i)
                break
        else:
            clusters.append([i])

    def mean(cluster: list[int]) -> float:
        return fmean(answers[i] for i in cluster)

    clusters.sort(key=lambda c: (-len(c), min(c)))
    merged: list[list[list[int]]] = []
    for cluster in clusters:
        if mode == "percent_scale":
            for group in merged:
                if _same_up_to_scale(mean(cluster), mean(group[0]), tolerance):
                    group.append(cluster)
                    break
            else:
                merged.append([cluster])
        else:
            merged.append([cluster])

    return [
        VoteGroup(
            size=sum(len(c) for c in group),
            value=mean(group[0]),
            first_index=min(min(c) for c in group),
        )
        for group in merged
    ]
