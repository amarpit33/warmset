"""Ollama native API (``/api/chat``, ``/api/ps``, ``/api/generate`` for load/unload)."""
from __future__ import annotations

import time
from typing import Any

import httpx

from ..errors import ModelError
from ..types import Usage
from .base import Capabilities, GenerationParams, GenerationResult, Message, ModelInfo, Provider, Resident
from .http import HTTPProvider


def _usage(data: dict[str, Any], wall: float) -> Usage:
    """Build ``Usage`` from Ollama's final response (durations are nanoseconds)."""
    def secs(key: str) -> float | None:
        ns = data.get(key) or 0
        return ns / 1e9 if ns else None

    return Usage(prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                 completion_tokens=data.get("eval_count", 0) or 0, seconds=wall,
                 load_seconds=secs("load_duration") or 0.0, prompt_eval_seconds=secs("prompt_eval_duration"),
                 eval_seconds=secs("eval_duration"), calls=1)


class OllamaProvider(HTTPProvider, Provider):
    name = "ollama"
    capabilities = Capabilities(structured_output="grammar", reports_load_time=True, residency_api=True,
                                can_load_unload=True)

    def __init__(self, base_url: str = "http://localhost:11434", *, timeout: float = 300.0,
                 num_ctx: int | None = 8192, think: bool = False, keep_alive: str | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        super().__init__(base_url, timeout, client)
        self.num_ctx = num_ctx
        self.think = think  # off keeps reasoning models from spending max_tokens thinking
        self.keep_alive = keep_alive

    def _payload(self, messages: list[Message], params: GenerationParams) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": params.temperature, "num_predict": params.max_tokens}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        payload: dict[str, Any] = {"model": params.model, "stream": False, "think": self.think,
                                   "messages": [m.to_dict() for m in messages], "options": options}
        if params.json_schema is not None:
            payload["format"] = params.json_schema
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    async def generate(self, messages: list[Message], params: GenerationParams) -> GenerationResult:
        start = time.perf_counter()
        data = await self._post("/api/chat", self._payload(messages, params))
        wall = time.perf_counter() - start
        return GenerationResult(text=data.get("message", {}).get("content", "") or "", usage=_usage(data, wall),
                                truncated=data.get("done_reason") == "length")

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        data = await self._post("/api/embed", {"model": model, "input": texts})
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise ModelError("embedding response missing or wrong length")
        return vectors

    async def health(self) -> bool:
        try:
            return (await self._http.get("/api/tags", timeout=3.0)).status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[ModelInfo]:
        out = []
        for m in (await self._get("/api/tags")).get("models", []):
            d = m.get("details", {})
            fams = " ".join([d.get("family", ""), *(d.get("families") or [])]).lower()
            name = m["name"]
            out.append(ModelInfo(name, m.get("size", 0), d.get("parameter_size"), d.get("quantization_level"),
                                 is_embedding="bert" in fams or any(h in name.lower() for h in ("embed", "bge-"))))
        return out

    async def running(self) -> list[Resident]:
        return [Resident(m["name"], m.get("size", 0)) for m in (await self._get("/api/ps")).get("models", [])]

    async def load(self, model: str) -> float:
        """An empty generate request loads the model; Ollama reports how long that took."""
        start = time.perf_counter()
        data = await self._post("/api/generate", {"model": model, "prompt": "", "stream": False})
        reported = (data.get("load_duration") or 0) / 1e9
        return reported or (time.perf_counter() - start)

    async def unload(self, model: str) -> None:
        await self._post("/api/generate", {"model": model, "prompt": "", "stream": False, "keep_alive": 0})
