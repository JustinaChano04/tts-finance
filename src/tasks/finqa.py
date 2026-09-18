"""FinQA dataset loading and prompt construction.

Loads `wujian123/finqa`, a Parquet mirror of the original FinQA release
(pre_text / post_text / table / qa.program / qa.exe_ans). The original
`dreamerdeo/finqa` and `ibm-research/finqa` repos rely on a Hugging Face
"dataset script" loader that current `datasets` versions no longer execute,
so this mirror is used instead — the row schema matches the FinQA paper.
"""

from __future__ import annotations

import re

from datasets import load_dataset

from tasks.base import FinancialQuestion, Task

HF_DATASET_PATH = "wujian123/finqa"

PROMPT_TEMPLATE = """You are a financial analyst. Answer the question below using ONLY the numbers given in the context and table. Solve it by writing a short Python program.

Context:
{pre_text}

Table:
{table}

{post_text}

Question: {question}

Respond in exactly this format:

REASONING:
<your step-by-step reasoning>

CODE:
```python
<python code that computes the final numeric answer and assigns it to a variable named `answer`>
```

ANSWER:
<the final numeric answer>
"""


def _format_table(table: list[list[str]]) -> str:
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def _parse_gold_answer(qa: dict) -> float | None:
    exe_ans = qa.get("exe_ans")
    if isinstance(exe_ans, (int, float)):
        return float(exe_ans)

    raw = str(qa.gelet("answer", "")).strip()
    raw = raw.replace(",", "").replace("$", "")
    is_percent = raw.endswith("%")
    raw = raw.rstrip("%")
    match = re.search(r"-?\d+\.?\d*", raw)
    if not match:
        return None
    value = float(match.group())
    return value / 100.0 if is_percent else value


class FinQATask(Task):
    def __init__(self, split: str = "validation", max_examples: int | None = None):
        self.split = split
        self.max_examples = max_examples

    def load(self) -> list[FinancialQuestion]:
        dataset = load_dataset(HF_DATASET_PATH, split=self.split)
        if self.max_examples is not None:
            dataset = dataset.select(range(min(self.max_examples, len(dataset))))

        questions = []
        for example in dataset:
            qa = example["qa"]
            questions.append(
                FinancialQuestion(
                    id=example["id"],
                    question=qa["question"],
                    pre_text="\n".join(example["pre_text"]),
                    post_text="\n".join(example["post_text"]),
                    table=example["table"],
                    gold_answer=_parse_gold_answer(qa),
                    gold_program=qa.get("program", ""),
                )
            )
        return questions

    def build_prompt(self, question: FinancialQuestion) -> str:
        return PROMPT_TEMPLATE.format(
            pre_text=question.pre_text,
            table=_format_table(question.table),
            post_text=question.post_text,
            question=question.question,
        )
