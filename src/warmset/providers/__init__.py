from .base import Capabilities, GenerationParams, GenerationResult, Message, ModelInfo, Provider, Resident
from .fake import FakeModel, FakeProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

__all__ = ["Capabilities", "FakeModel", "FakeProvider", "GenerationParams", "GenerationResult", "Message",
           "ModelInfo", "OllamaProvider", "OpenAICompatProvider", "Provider", "Resident", "create_provider"]


def create_provider(backend: str, base_url: str | None = None, **kwargs: object) -> Provider:
    """Factory: ``ollama`` or ``openai`` (any OpenAI-compatible server)."""
    if backend == "ollama":
        return OllamaProvider(base_url or "http://localhost:11434", **kwargs)  # type: ignore[arg-type]
    if backend in ("openai", "vllm", "openai_compat"):
        return OpenAICompatProvider(base_url or "http://localhost:8000", **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown backend '{backend}', expected 'ollama' or 'openai'")
