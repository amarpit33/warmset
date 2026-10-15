"""Backend-agnostic provider interface.

Derived from local-llm-platform's ``ModelClient``. Differences: schema-constrained output is a
first-class parameter (no reaching into private payload builders), every result says whether it
was truncated, and providers expose model residency and load/unload so the controller can price
model swaps.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from ..types import Usage

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class GenerationParams:
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    json_schema: dict[str, Any] | None = None  # constrain output to this JSON schema
    schema_name: str = "output"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    usage: Usage
    truncated: bool = False


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_bytes: int = 0
    parameter_size: str | None = None
    quantization: str | None = None
    is_embedding: bool = False


@dataclass(frozen=True)
class Resident:
    """A model currently loaded in memory."""

    name: str
    size_bytes: int = 0


@dataclass(frozen=True)
class Capabilities:
    """What a backend can do. Runtimes differ; the controller must not assume."""

    structured_output: Literal["grammar", "json_mode", "none"] = "none"
    reports_load_time: bool = False
    residency_api: bool = False       # running() reflects real memory state
    can_load_unload: bool = False


class Provider(ABC):
    """Interface every serving backend implements."""

    name: str = "provider"
    capabilities: Capabilities = Capabilities()

    @abstractmethod
    async def generate(self, messages: list[Message], params: GenerationParams) -> GenerationResult:
        """Complete ``messages``. Raises ``ModelError``/``ModelUnavailable``."""

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        from ..errors import ModelError
        raise ModelError(f"{self.name} does not support embeddings")

    @abstractmethod
    async def health(self) -> bool:
        """True if the backend is reachable."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Models the backend can serve."""

    async def running(self) -> list[Resident]:
        """Models loaded in memory right now. Empty if the backend cannot tell."""
        return []

    async def load(self, model: str) -> float:
        """Load ``model`` into memory; returns the load time in seconds (0 if unsupported)."""
        return 0.0

    async def unload(self, model: str) -> None:  # noqa: B027
        """Evict ``model`` from memory (no-op if unsupported)."""

    async def aclose(self) -> None:  # noqa: B027
        """Release resources."""
