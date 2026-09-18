"""Model interface abstraction plus a local Hugging Face Transformers backend.

Kept abstract enough that a later vLLM or API-backed implementation can drop
in behind the same `LLM.generate` signature without touching strategy code.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency: float


class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> GenerationResult:
        ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


class LocalLLM(LLM):
    """Runs a Hugging Face causal LM locally (CPU/MPS/CUDA)."""

    def __init__(self, model_name: str, device: str = "auto", dtype: str = "auto"):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.torch_dtype = resolve_dtype(dtype, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.torch_dtype
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - start

        new_tokens = output_ids[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return GenerationResult(
            text=text,
            input_tokens=input_len,
            output_tokens=len(new_tokens),
            latency=latency,
        )
