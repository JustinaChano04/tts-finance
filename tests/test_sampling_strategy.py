from models.local_llm import GenerationResult, LLM
from strategies.greedy import GreedyStrategy
from strategies.sampling import SamplingStrategy, majority_vote
from strategies.base import Solution
from tasks.base import FinancialQuestion
from verifiers.python_executor import PythonExecutorVerifier


class ScriptedLLM(LLM):
    """Returns responses from a fixed list, one per call, cycling if exhausted."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    def generate(self, prompt, temperature=0.0, max_new_tokens=512):
        response = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return GenerationResult(
            text=response, input_tokens=10, output_tokens=10, latency=0.01
        )


def make_question() -> FinancialQuestion:
    return FinancialQuestion(
        id="q1",
        question="q",
        pre_text="",
        post_text="",
        table=[],
        gold_answer=10.0,
    )


def response_for(expr: str) -> str:
    return f"""REASONING:
r

CODE:
```python
answer = {expr}
```

ANSWER:
x
"""


def test_sampling_makes_exactly_n_model_calls():
    model = ScriptedLLM([response_for("10")])
    verifier = PythonExecutorVerifier()
    strategy = SamplingStrategy(num_samples=4, temperature=0.7)

    result = strategy.run(
        make_question(), model, verifier, build_prompt=lambda q: "prompt"
    )

    assert model.calls == 4
    assert result.model_calls == 4
    assert len(result.trajectories) == 4


def test_sampling_majority_vote_picks_correct_answer():
    model = ScriptedLLM([response_for("10"), response_for("10"), response_for("999")])
    verifier = PythonExecutorVerifier()
    strategy = SamplingStrategy(num_samples=3, temperature=0.7)

    result = strategy.run(
        make_question(), model, verifier, build_prompt=lambda q: "prompt"
    )

    assert result.final_answer == 10.0
    assert result.correct


def test_greedy_uses_single_call():
    model = ScriptedLLM([response_for("10")])
    verifier = PythonExecutorVerifier()
    strategy = GreedyStrategy()

    result = strategy.run(
        make_question(), model, verifier, build_prompt=lambda q: "prompt"
    )

    assert model.calls == 1
    assert result.correct


def test_majority_vote_returns_none_when_all_fail():
    solutions = [
        Solution(
            raw_response="",
            reasoning=None,
            code=None,
            stated_answer=None,
            generation=GenerationResult(
                text="", input_tokens=0, output_tokens=0, latency=0
            ),
            verification=None,
        )
    ]
    assert majority_vote(solutions, tolerance=0.01) is None


def test_truncated_generation_is_tagged_truncated_not_no_code():
    model = ScriptedLLM(["REASONING:\nr\n\nCODE:\n```python\nx = 1\ny ="])
    strategy = SamplingStrategy(num_samples=1)

    result = strategy.run(
        make_question(), model, PythonExecutorVerifier(), build_prompt=lambda q: "p"
    )

    assert result.trajectories[0].verification.error_type == "truncated"


def test_result_records_gold_answer():
    model = ScriptedLLM([response_for("10")])
    for strategy in (GreedyStrategy(), SamplingStrategy(num_samples=2)):
        result = strategy.run(
            make_question(), model, PythonExecutorVerifier(), build_prompt=lambda q: "p"
        )
        assert result.gold_answer == 10.0


def test_vote_is_scored_with_verifier_match_mode():
    model = ScriptedLLM([response_for("1000")])  # 100x the gold of 10
    strategy = SamplingStrategy(num_samples=2)

    strict = strategy.run(
        make_question(), model, PythonExecutorVerifier(match_mode="strict"), lambda q: "p"
    )
    lenient = strategy.run(
        make_question(),
        model,
        PythonExecutorVerifier(match_mode="percent_scale"),
        lambda q: "p",
    )

    assert not strict.correct
    assert lenient.correct
