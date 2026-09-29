import pytest

from strategies.base import has_unclosed_code_block, parse_model_response


def test_well_formed_response():
    text = """REASONING:
The revenue grew by 10.

CODE:
```python
answer = 110 - 100
```

ANSWER:
10
"""
    reasoning, code, answer = parse_model_response(text)
    assert "revenue grew" in reasoning
    assert code.strip() == "answer = 110 - 100"
    assert answer.strip() == "10"


def test_missing_reasoning_section():
    text = """CODE:
```python
answer = 42
```

ANSWER:
42
"""
    reasoning, code, answer = parse_model_response(text)
    assert reasoning is None
    assert code.strip() == "answer = 42"
    assert answer.strip() == "42"


def test_code_fence_without_language_tag():
    text = """REASONING:
some reasoning

CODE:
```
answer = 1 + 1
```

ANSWER:
2
"""
    _, code, _ = parse_model_response(text)
    assert code.strip() == "answer = 1 + 1"


def test_missing_answer_section():
    text = """REASONING:
r

CODE:
```python
answer = 5
```
"""
    _, code, answer = parse_model_response(text)
    assert code.strip() == "answer = 5"
    assert answer is None


def test_completely_malformed_response():
    text = "I don't know how to solve this."
    reasoning, code, answer = parse_model_response(text)
    assert code is None


@pytest.mark.parametrize("tag", ["python", "py", "python3", "Python", ""])
def test_fence_language_tag_is_not_part_of_code(tag):
    text = f"CODE:\n```{tag}\nanswer = 1\n```\nANSWER:\n1\n"
    _, code, _ = parse_model_response(text)
    assert code == "answer = 1"


def test_code_is_taken_from_block_after_code_marker():
    text = """REASONING:
Use ```x = total / count``` to get the average.

CODE:
```python
answer = 10 / 2
```
"""
    _, code, _ = parse_model_response(text)
    assert code == "answer = 10 / 2"


def test_truncated_code_block_is_detected():
    text = "REASONING:\nr\n\nCODE:\n```python\nrevenue = 100\ncost ="
    _, code, _ = parse_model_response(text)
    assert code is None
    assert has_unclosed_code_block(text)


def test_missing_code_block_is_not_reported_as_truncated():
    assert not has_unclosed_code_block("I don't know how to solve this.")
    assert not has_unclosed_code_block("CODE:\n```python\nanswer = 1\n```\n")
