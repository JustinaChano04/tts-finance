from tts_finance.evaluation.evaluator import Evaluator
from tts_finance.evaluation import metrics
from tts_finance.strategies.base import InferenceResult


def make_result(correct: bool, total_tokens: int = 100, latency: float = 1.0, model_calls: int = 1) -> InferenceResult:
    return InferenceResult(
        question_id="q", final_answer=1.0, correct=correct,
        trajectories=[], model_calls=model_calls,
        input_tokens=total_tokens // 2, output_tokens=total_tokens // 2,
        total_tokens=total_tokens, latency=latency,
    )


def test_accuracy():
    results = [make_result(True), make_result(True), make_result(False), make_result(False)]
    assert metrics.accuracy(results) == 0.5


def test_accuracy_empty():
    assert metrics.accuracy([]) == 0.0


def test_average_tokens_and_latency():
    results = [make_result(True, total_tokens=100, latency=2.0), make_result(True, total_tokens=200, latency=4.0)]
    assert metrics.average_tokens(results) == 150
    assert metrics.average_latency(results) == 3.0


def test_evaluator_returns_all_fields():
    results = [make_result(True, model_calls=4), make_result(False, model_calls=4)]
    out = Evaluator().evaluate(results)
    assert out["accuracy"] == 0.5
    assert out["num_examples"] == 2
    assert out["average_model_calls"] == 4
