import pytest

from evaluation import diagnostics, metrics
from evaluation.evaluator import Evaluator
from models.local_llm import GenerationResult
from strategies.base import InferenceResult, Solution
from strategies.sampling import majority_vote
from verifiers.base import VerificationResult
from verifiers.matching import matches_gold


def make_result(
    correct: bool, total_tokens: int = 100, latency: float = 1.0, model_calls: int = 1
) -> InferenceResult:
    return InferenceResult(
        question_id="q",
        final_answer=1.0,
        correct=correct,
        trajectories=[],
        model_calls=model_calls,
        input_tokens=total_tokens // 2,
        output_tokens=total_tokens // 2,
        total_tokens=total_tokens,
        latency=latency,
    )


def test_accuracy():
    results = [
        make_result(True),
        make_result(True),
        make_result(False),
        make_result(False),
    ]
    assert metrics.accuracy(results) == 0.5


def test_accuracy_empty():
    assert metrics.accuracy([]) == 0.0


def test_average_tokens_and_latency():
    results = [
        make_result(True, total_tokens=100, latency=2.0),
        make_result(True, total_tokens=200, latency=4.0),
    ]
    assert metrics.average_tokens(results) == 150
    assert metrics.average_latency(results) == 3.0


def test_evaluator_returns_all_fields():
    results = [make_result(True, model_calls=4), make_result(False, model_calls=4)]
    out = Evaluator().evaluate(results)
    assert out["accuracy"] == 0.5
    assert out["num_examples"] == 2
    assert out["average_model_calls"] == 4


# --- diagnostics --------------------------------------------------------------


def make_solution(answer=None, error_type=None, stated=None) -> Solution:
    if error_type:
        verification = VerificationResult.failure(error_type, "err")
    else:
        verification = VerificationResult(success=True, answer=answer, correct=False)
    return Solution(
        raw_response="",
        reasoning=None,
        code="",
        stated_answer=stated,
        generation=GenerationResult(text="", input_tokens=0, output_tokens=0, latency=0),
        verification=verification,
    )


def make_voted_result(answers, gold, error_types=()) -> InferenceResult:
    solutions = [make_solution(a) for a in answers] + [
        make_solution(error_type=e) for e in error_types
    ]
    final = majority_vote(solutions, tolerance=0.01)
    return InferenceResult(
        question_id="q",
        final_answer=final,
        correct=matches_gold(final, gold, 0.01),
        gold_answer=gold,
        trajectories=solutions,
    )


@pytest.mark.parametrize(
    "answer,tag",
    [
        (10.0, "correct"),
        (1000.0, "percent_scale"),
        (0.1, "percent_scale"),
        (10_000.0, "unit_scale"),
        (-10.0, "sign_flip"),
        (10.3, "near_miss"),
        (42.0, "wrong_value"),
    ],
)
def test_sample_tag_for_executed_answers(answer, tag):
    assert diagnostics.sample_tag(make_solution(answer), gold=10.0, tolerance=0.01) == tag


def test_sample_tag_for_failed_sample_uses_error_type():
    solution = make_solution(error_type="safety_reject")
    assert diagnostics.sample_tag(solution, gold=10.0, tolerance=0.01) == "safety_reject"


@pytest.mark.parametrize(
    "answers,error_types,tag",
    [
        ([10, 10, 5], (), "vote_correct"),
        ([10, 5, 5], (), "vote_lost"),
        ([5, 10], (), "tie_lost"),  # tie goes to the smaller-valued cluster
        ([5, 5, 7], (), "none_correct"),
        ([], ("runtime_error", "no_code"), "all_failed"),
    ],
)
def test_question_tag(answers, error_types, tag):
    result = make_voted_result(answers, gold=10.0, error_types=error_types)
    assert diagnostics.question_tag(result, tolerance=0.01, mode="strict") == tag


def test_question_tag_gold_missing():
    result = make_voted_result([10], gold=None)
    assert diagnostics.question_tag(result, tolerance=0.01, mode="strict") == "gold_missing"


@pytest.mark.parametrize(
    "stated,expected", [("0.15", 0.15), ("$1,234.5", 1234.5), ("15%", 15.0), ("n/a", None)]
)
def test_parse_stated_answer(stated, expected):
    assert diagnostics.parse_stated_answer(stated) == expected


def test_summarize_separates_strict_and_percent_scale_accuracy():
    results = [
        make_voted_result([0.15], gold=0.15),
        make_voted_result([15.0], gold=0.15),
        make_voted_result([], gold=0.15, error_types=("runtime_error",)),
    ]
    results[2].trajectories[0].stated_answer = "15%"

    summary = diagnostics.summarize(results, tolerance=0.01, mode="percent_scale")

    assert summary["accuracy_strict"] == pytest.approx(1 / 3)
    assert summary["accuracy_percent_scale"] == pytest.approx(2 / 3)
    assert summary["pass_at_n"] == pytest.approx(2 / 3)
    assert summary["failed_but_stated_correct"] == 1
    assert summary["sample_tags"] == {"correct": 1, "percent_scale": 1, "runtime_error": 1}


def test_evaluator_includes_diagnostics():
    out = Evaluator().evaluate([make_voted_result([10], gold=10.0)])
    assert out["diagnostics"]["question_tags"] == {"vote_correct": 1}
