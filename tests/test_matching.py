import pytest

from strategies.sampling import majority_vote
from tests.test_evaluation import make_solution
from verifiers.matching import matches_gold, vote_groups, within_tolerance


def vote(answers, mode="strict", tolerance=0.01):
    return majority_vote([make_solution(a) for a in answers], tolerance, mode)


def test_tolerance_is_relative_for_small_values():
    assert not within_tolerance(0.002, 0.2, 0.01)
    assert not within_tolerance(0.0, -0.52485, 0.01)
    assert not matches_gold(0.0, -0.52485, 0.01, "percent_scale")
    assert within_tolerance(0.20000000000000018, 0.2, 0.01)
    assert within_tolerance(1e-12, 0.0, 0.01)


def test_tie_goes_to_first_sampled_group_not_smallest_value():
    assert vote([5.0, 10.0]) == 5.0
    assert vote([10.0, 5.0]) == 10.0
    assert vote([-83.6, -57.4, -52.5]) == -83.6
    assert vote([-52.5, -83.6, -57.4]) == -52.5


def test_percent_scale_votes_merge_in_percent_scale_mode_only():
    answers = [0.109, 10.9, 10.9, 70.0]
    assert vote(answers, mode="percent_scale") == 10.9
    assert vote([0.109, 10.9, 70.0], mode="strict") == 0.109  # three-way tie, first sampled


def test_scale_merge_can_outvote_a_repeated_wrong_answer():
    # 0.935 and 93.5 are one answer; together (3 votes) they beat the repeated 3.3.
    # Strictly, 3.3 and 93.5 tie at 2 each and the first-sampled one (3.3) wins.
    answers = [0.935, 3.3, 93.5, 3.3, 93.5]
    assert vote(answers, mode="percent_scale") == 93.5
    assert vote(answers, mode="strict") == 3.3


def test_group_value_is_not_a_mean_across_scales():
    groups = vote_groups([0.2, 20.0, 20.0], 0.01, "percent_scale")
    assert len(groups) == 1
    assert groups[0].size == 3
    assert groups[0].value == pytest.approx(20.0)


def test_unit_scale_answers_are_never_merged():
    assert len(vote_groups([10.0, 10_000.0], 0.01, "percent_scale")) == 2
