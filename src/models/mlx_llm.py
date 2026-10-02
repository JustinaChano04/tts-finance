"""Apple-Silicon MLX backend (mlx-lm), for models too large for bf16 on MPS."""

from __future__ import annotations

import time

from models.local_llm import LLM, GenerationResult


class MLXLLM(LLM):
    """Runs an MLX-format causal LM (e.g. a 4-bit mlx-community build)."""

    def __init__(self, model_name: str):
        from mlx_lm import load

        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> GenerationResult:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(input_text)

        start = time.perf_counter()
        pieces, output_tokens, finish_reason = [], 0, None
        for response in stream_generate(
            self.model,
            self.tokenizer,
            input_ids,
            max_tokens=max_new_tokens,
            sampler=make_sampler(temp=temperature),
        ):
            pieces.append(response.text)
            output_tokens = response.generation_tokens
            finish_reason = response.finish_reason
        latency = time.perf_counter() - start

        return GenerationResult(
            text="".join(pieces),
            input_tokens=len(input_ids),
            output_tokens=output_tokens,
            latency=latency,
            truncated=finish_reason == "length",
        )
