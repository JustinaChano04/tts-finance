import subprocess

import pytest

from tasks.base import FinancialQuestion
from verifiers.python_executor import PythonExecutorVerifier, check_code_safety


def make_question(gold_answer: float) -> FinancialQuestion:
    return FinancialQuestion(
        id="q1",
        question="q",
        pre_text="",
        post_text="",
        table=[],
        gold_answer=gold_answer,
    )


def test_valid_code_correct_answer():
    verifier = PythonExecutorVerifier()
    result = verifier.verify("answer = 100 - 90", make_question(gold_answer=10))
    assert result.success
    assert result.correct
    assert result.answer == 10


def test_valid_code_incorrect_answer():
    verifier = PythonExecutorVerifier()
    result = verifier.verify("answer = 100 - 90", make_question(gold_answer=999))
    assert result.success
    assert not result.correct
    assert result.answer == 10


def test_execution_error():
    verifier = PythonExecutorVerifier()
    result = verifier.verify("answer = 1 / 0", make_question(gold_answer=1))
    assert not result.success
    assert not result.correct
    assert "ZeroDivisionError" in result.execution_error


def test_syntax_error_is_its_own_error_type():
    verifier = PythonExecutorVerifier()
    result = verifier.verify("answer = (", make_question(gold_answer=1))
    assert not result.success
    assert result.error_type == "syntax_error"
    assert "SyntaxError" in result.execution_error


def test_no_answer_variable_assigned():
    verifier = PythonExecutorVerifier()
    result = verifier.verify("x = 5", make_question(gold_answer=5))
    assert not result.success
    assert "did not assign" in result.execution_error


def test_timeout():
    verifier = PythonExecutorVerifier(timeout_seconds=1)
    result = verifier.verify("while True:\n    pass", make_question(gold_answer=1))
    assert not result.success
    assert "timed out" in result.execution_error or "CPU" in (
        result.execution_error or ""
    )


def test_disallowed_import_rejected():
    is_safe, reason = check_code_safety("import os\nanswer = 1")
    assert not is_safe
    assert "import" in reason


def test_disallowed_open_call_rejected():
    is_safe, reason = check_code_safety("open('/etc/passwd')\nanswer = 1")
    assert not is_safe


def test_math_module_available_without_import():
    verifier = PythonExecutorVerifier()
    result = verifier.verify("answer = math.sqrt(16)", make_question(gold_answer=4))
    assert result.success
    assert result.answer == 4

    verifier = PythonExecutorVerifier(tolerance=0.01)
    result = verifier.verify("answer = 4.001", make_question(gold_answer=4.0))
    assert result.correct


# --- error attribution -------------------------------------------------------
# Valid code a model would reasonably write must not be reported as a failure.


@pytest.mark.parametrize(
    "code",
    [
        "import math\nanswer = math.sqrt(16)",
        "from math import sqrt\nanswer = sqrt(16)",
        "answer = 4 if all(x > 0 for x in [1, 2]) and any([1]) else 0",
        "answer = list(reversed([1, 4]))[0]",
        "q, r = divmod(9, 5)\nanswer = r",
        "class A:\n    v = 4\nanswer = A.v",
        "if __name__ == '__main__':\n    answer = 4",
        "try:\n    x = 4 / 1\nexcept ZeroDivisionError:\n    x = 0\nanswer = x",
        "print('debug', end='')\nanswer = 4",
        "print('{\"ok\": false}')\nanswer = 4",
    ],
)
def test_valid_code_is_not_reported_as_failure(code):
    result = PythonExecutorVerifier().verify(code, make_question(gold_answer=4))
    assert result.success, result.execution_error
    assert result.answer == 4
    assert result.error_type is None


@pytest.mark.parametrize(
    "code,error_type",
    [
        ("", "no_code"),
        ("answer = (", "syntax_error"),
        ("import os\nanswer = 1", "safety_reject"),
        ("answer = ().__class__", "safety_reject"),
        ("answer = 1 / 0", "runtime_error"),
        ("x = 5", "no_answer_var"),
        ("answer = None", "non_numeric"),
        ("answer = '15%'", "non_numeric"),
        ("answer = True", "non_numeric"),
        ("answer = float('nan')", "non_numeric"),
    ],
)
def test_each_failure_path_sets_error_type(code, error_type):
    result = PythonExecutorVerifier().verify(code, make_question(gold_answer=1))
    assert not result.success
    assert result.error_type == error_type
    assert result.error_source == "model"


def test_timeout_error_type():
    verifier = PythonExecutorVerifier(timeout_seconds=1)
    result = verifier.verify("while True:\n    pass", make_question(gold_answer=1))
    assert result.error_type == "timeout"
    assert result.error_source == "model"


def test_executor_crash_is_attributed_to_harness(monkeypatch):
    def crashed_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", crashed_run)
    result = PythonExecutorVerifier().verify("answer = 1", make_question(gold_answer=1))
    assert not result.success
    assert result.error_type == "harness_error"
    assert result.error_source == "harness"


# --- scoring ------------------------------------------------------------------


def test_strict_mode_rejects_percent_scale():
    verifier = PythonExecutorVerifier(match_mode="strict")
    result = verifier.verify("answer = 15.0", make_question(gold_answer=0.15))
    assert result.success
    assert not result.correct


@pytest.mark.parametrize("answer,gold", [(15.0, 0.15), (0.2469, 24.69)])
def test_percent_scale_mode_accepts_100x(answer, gold):
    verifier = PythonExecutorVerifier(match_mode="percent_scale")
    result = verifier.verify(f"answer = {answer}", make_question(gold_answer=gold))
    assert result.correct


def test_percent_scale_mode_does_not_forgive_units():
    verifier = PythonExecutorVerifier(match_mode="percent_scale")
    result = verifier.verify("answer = 3576000", make_question(gold_answer=3576))
    assert not result.correct


def test_final_expression_is_recovered_when_answer_not_assigned():
    code = "total = 136104\nrisk = 1244659\ntotal / risk"
    result = PythonExecutorVerifier().verify(code, make_question(gold_answer=0.10935))
    assert result.success, result.execution_error
    assert result.recovered_from_expression
    assert result.answer == pytest.approx(136104 / 1244659)


def test_explicit_answer_is_not_flagged_as_recovered():
    result = PythonExecutorVerifier().verify("answer = 1\nanswer", make_question(gold_answer=1))
    assert result.success
    assert not result.recovered_from_expression


@pytest.mark.parametrize("code", ["x = 5", "x = 5\nprint(x)", "x = 5\nif x:\n    x"])
def test_no_recovery_without_a_final_expression_value(code):
    result = PythonExecutorVerifier().verify(code, make_question(gold_answer=5))
    assert not result.success
    assert result.error_type == "no_answer_var"


def test_recovered_expression_still_goes_through_safety_check():
    result = PythonExecutorVerifier().verify("x = 1\n().__class__", make_question(gold_answer=1))
    assert result.error_type == "safety_reject"
