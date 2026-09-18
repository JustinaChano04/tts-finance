"""Requires network access to download/cache `wujian123/finqa` from the HF Hub."""

from tasks.finqa import FinQATask


def test_dataset_loads_small_subset():
    task = FinQATask(split="validation", max_examples=3)
    questions = task.load()

    assert len(questions) == 3
    q = questions[0]
    assert q.id
    assert q.question
    assert isinstance(q.table, list)
    assert q.gold_answer is not None


def test_build_prompt_includes_context_and_question():
    task = FinQATask(split="validation", max_examples=1)
    q = task.load()[0]
    prompt = task.build_prompt(q)

    assert q.question in prompt
    assert "REASONING:" in prompt
    assert "CODE:" in prompt
    assert "ANSWER:" in prompt
