from strategies.base import parse_model_response


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
