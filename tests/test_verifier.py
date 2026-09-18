from tasks.base import FinancialQuestion
from verifiers.python_executor import PythonExecutorVerifier, check_code_safety


def make_question(gold_answer: float) -> FinancialQuestion:
    return FinancialQuestion(
        id="q1", question="q", pre_text="", post_text="", table=[], gold_answer=gold_answer,
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


def test_syntax_error_rejected_by_safety_check():
    verifier = PythonExecutorVerifier()
    result = verifier.verify("answer = (", make_question(gold_answer=1))
    assert not result.success
    assert "rejected by safety check" in result.execution_error


def test_no_answer_variable_assigned():
    verifier = PythonExecutorVerifier()
    result = verifier.verify("x = 5", make_question(gold_answer=5))
    assert not result.success
    assert "did not assign" in result.execution_error


def test_timeout():
    verifier = PythonExecutorVerifier(timeout_seconds=1)
    result = verifier.verify("while True:\n    pass", make_question(gold_answer=1))
    assert not result.success
    assert "timed out" in result.execution_error or "CPU" in (result.execution_error or "")


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
