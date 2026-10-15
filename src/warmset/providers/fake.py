"""Deterministic in-process provider for tests, docs and simulations.

Simulates model residency under a memory limit (LRU eviction), load time, and prefill/decode
throughput. Reported ``Usage.seconds`` is simulated; nothing sleeps.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from ..errors import ModelError
from ..types import Usage
from .base import Capabilities, GenerationParams, GenerationResult, Message, ModelInfo, Provider, Resident

Responder = Callable[[list[Message], GenerationParams], str]


@dataclass
class FakeModel:
    respond: Responder
    size_bytes: int = 2_000_000_000
    load_seconds: float = 10.0
    prefill_tps: float = 200.0
    decode_tps: float = 20.0


def tokens(text: str) -> int:
    """Rough token count: ~4 characters per token."""
    return max(1, len(text) // 4)


class FakeProvider(Provider):
    name = "fake"
    capabilities = Capabilities(structured_output="grammar", reports_load_time=True, residency_api=True,
                                can_load_unload=True)

    def __init__(self, models: dict[str, FakeModel], memory_bytes: int | None = None, time_scale: float = 0.0) -> None:
        self.models = models
        self.memory_bytes = memory_bytes
        self.time_scale = time_scale  # real seconds slept per simulated second (0: just yield to the loop)
        self._resident: OrderedDict[str, None] = OrderedDict()
        self.calls: list[tuple[str, list[Message], GenerationParams]] = []

    def _ensure_loaded(self, model: str) -> float:
        if model not in self.models:
            raise ModelError(f"model '{model}' not found")
        if model in self._resident:
            self._resident.move_to_end(model)
            return 0.0
        need = self.models[model].size_bytes
        while self.memory_bytes is not None and self._resident and \
                sum(self.models[m].size_bytes for m in self._resident) + need > self.memory_bytes:
            self._resident.popitem(last=False)  # evict least recently used
        self._resident[model] = None
        return self.models[model].load_seconds

    async def generate(self, messages: list[Message], params: GenerationParams) -> GenerationResult:
        load = self._ensure_loaded(params.model)
        spec = self.models[params.model]
        self.calls.append((params.model, messages, params))
        text = spec.respond(messages, params)
        p_tok = sum(tokens(m.content) for m in messages)
        c_tok = tokens(text)
        truncated = c_tok > params.max_tokens
        if truncated:
            text, c_tok = text[: params.max_tokens * 4], params.max_tokens
        prefill, decode = p_tok / spec.prefill_tps, c_tok / spec.decode_tps
        await asyncio.sleep((load + prefill + decode) * self.time_scale)
        return GenerationResult(text, Usage(p_tok, c_tok, load + prefill + decode, load, prefill, decode, 1),
                                truncated)

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(n, m.size_bytes) for n, m in self.models.items()]

    async def running(self) -> list[Resident]:
        return [Resident(n, self.models[n].size_bytes) for n in self._resident]

    async def load(self, model: str) -> float:
        return self._ensure_loaded(model)

    async def unload(self, model: str) -> None:
        self._resident.pop(model, None)
