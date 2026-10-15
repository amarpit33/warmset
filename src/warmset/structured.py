"""Schema-constrained generation over any ``Provider``.

Derived from resume-jd-matcher's ``StructuredLLM``. Retries only the mechanical failure (a
truncated answer gets double the token budget, once). Retrying an invalid answer at a higher
temperature or on another model is a *policy* decision and belongs to the controller.
"""
from __future__ import annotations

import copy
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import InvalidOutput
from .providers.base import GenerationParams, GenerationResult, Message, Provider
from .types import Usage

T = TypeVar("T", bound=BaseModel)


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``/``$defs`` into one self-contained tree.

    Grammar converters in serving engines handle local refs unevenly; an inlined schema works everywhere.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]]))
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


async def generate_text(provider: Provider, messages: list[Message], model: str, *, temperature: float,
                        max_tokens: int, json_schema: dict[str, Any] | None = None,
                        schema_name: str = "output") -> tuple[GenerationResult, Usage]:
    """Generate, retrying once with double the budget if the answer was cut off."""
    usage = Usage()
    for attempt in range(2):
        params = GenerationParams(model, temperature, max_tokens, json_schema, schema_name)
        result = await provider.generate(messages, params)
        usage = usage + result.usage
        if not result.truncated or attempt == 1:
            return result, usage
        max_tokens *= 2
    raise AssertionError("unreachable")


async def generate_structured(provider: Provider, messages: list[Message], schema: type[T], model: str, *,
                              temperature: float, max_tokens: int) -> tuple[T, Usage]:
    """Generate an instance of ``schema``. Raises ``InvalidOutput`` (carrying usage) on a bad answer."""
    json_schema = inline_refs(schema.model_json_schema())
    result, usage = await generate_text(provider, messages, model, temperature=temperature,
                                        max_tokens=max_tokens, json_schema=json_schema,
                                        schema_name=schema.__name__)
    try:
        return schema.model_validate_json(result.text), usage
    except ValidationError as exc:
        err = InvalidOutput(f"model did not return a valid {schema.__name__}: {exc.errors()[:2]}")
        err.usage = usage  # type: ignore[attr-defined]
        raise err from exc
