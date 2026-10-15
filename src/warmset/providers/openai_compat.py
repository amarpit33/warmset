"""OpenAI-compatible API (vLLM, llama.cpp ``llama-server``, LM Studio, ...).

These servers typically serve a fixed model and do not report load time or residency, so the
capability flags say so and the controller prices them without swap costs.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from ..errors import ModelError
from ..types import Usage
from .base import Capabilities, GenerationParams, GenerationResult, Message, ModelInfo, Provider
from .http import HTTPProvider, transport_error


class OpenAICompatProvider(HTTPProvider, Provider):
    name = "openai_compat"
    capabilities = Capabilities(structured_output="grammar", reports_load_time=False, residency_api=False,
                                can_load_unload=False)

    def __init__(self, base_url: str = "http://localhost:8000", *, timeout: float = 300.0,
                 api_key: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(base_url, timeout, client)
        if api_key and client is None:
            self._http.headers["Authorization"] = f"Bearer {api_key}"

    def _payload(self, messages: list[Message], params: GenerationParams) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": params.model, "stream": False,
                                   "messages": [m.to_dict() for m in messages],
                                   "temperature": params.temperature, "max_tokens": params.max_tokens}
        if params.json_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": params.schema_name, "schema": params.json_schema}}
        return payload

    async def generate(self, messages: list[Message], params: GenerationParams) -> GenerationResult:
        start = time.perf_counter()
        data = await self._post("/v1/chat/completions", self._payload(messages, params))
        wall = time.perf_counter() - start
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ModelError("malformed chat completion response") from exc
        u = data.get("usage") or {}
        return GenerationResult(text, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), wall, calls=1),
                                truncated=choice.get("finish_reason") == "length")

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        data = await self._post("/v1/embeddings", {"model": model, "input": texts})
        try:
            vectors = [r["embedding"] for r in sorted(data["data"], key=lambda r: r["index"])]
        except (KeyError, TypeError) as exc:
            raise ModelError("malformed embeddings response") from exc
        if len(vectors) != len(texts):
            raise ModelError("embedding response wrong length")
        return vectors

    async def health(self) -> bool:
        try:
            return (await self._http.get("/v1/models", timeout=3.0)).status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[ModelInfo]:
        try:
            resp = await self._http.get("/v1/models", timeout=10.0)
            resp.raise_for_status()
            return [ModelInfo(m["id"]) for m in resp.json().get("data", [])]
        except httpx.TransportError as exc:
            raise transport_error(exc, self._timeout) from exc
        except (httpx.HTTPStatusError, KeyError, ValueError) as exc:
            raise ModelError("malformed models response") from exc
