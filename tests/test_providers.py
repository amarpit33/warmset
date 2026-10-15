import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from tests.helpers import make_provider
from warmset.errors import InvalidOutput, ModelUnavailable
from warmset.providers import OllamaProvider, OpenAICompatProvider, create_provider
from warmset.providers.base import GenerationParams, Message
from warmset.structured import generate_structured, inline_refs


class Inner(BaseModel):
    x: int


class Outer(BaseModel):
    inner: Inner
    tags: list[Inner]


def mock(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t")


def test_ollama_payload_usage_truncation_and_residency():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        seen.append((req.url.path, body))
        if req.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": '{"x": 1}'}, "done_reason": "length",
                                             "prompt_eval_count": 10, "eval_count": 5, "load_duration": 2_000_000_000,
                                             "prompt_eval_duration": 500_000_000, "eval_duration": 1_000_000_000})
        if req.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "m", "size": 7}]})
        if req.url.path == "/api/generate":
            return httpx.Response(200, json={"load_duration": 3_000_000_000})
        return httpx.Response(404)

    async def go():
        p = OllamaProvider(client=mock(handler))
        r = await p.generate([Message("user", "hi")], GenerationParams("m", 0.1, 9, {"type": "object"}))
        assert r.truncated and r.usage.load_seconds == 2.0 and r.usage.eval_seconds == 1.0
        assert r.usage.prompt_eval_seconds == 0.5 and r.usage.calls == 1
        body = seen[0][1]
        assert body["format"] == {"type": "object"} and body["think"] is False
        assert body["options"] == {"temperature": 0.1, "num_predict": 9, "num_ctx": 8192}
        assert [x.name for x in await p.running()] == ["m"]
        assert await p.load("m") == 3.0
        await p.unload("m")
        assert seen[-1][1]["keep_alive"] == 0

    asyncio.run(go())


def test_openai_compat_response_format_and_errors():
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["response_format"]["json_schema"]["name"] == "Inner"
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    async def go():
        p = OpenAICompatProvider(client=mock(handler))
        r = await p.generate([Message("user", "x")], GenerationParams("m", json_schema={}, schema_name="Inner"))
        assert r.usage.prompt_tokens == 3 and not r.truncated
        assert not p.capabilities.residency_api

        def down(req):
            raise httpx.ConnectError("refused")
        with pytest.raises(ModelUnavailable):
            await OpenAICompatProvider(client=mock(down)).generate([Message("user", "x")], GenerationParams("m"))

    asyncio.run(go())


def test_create_provider():
    assert isinstance(create_provider("ollama"), OllamaProvider)
    assert isinstance(create_provider("vllm"), OpenAICompatProvider)
    with pytest.raises(ValueError):
        create_provider("nope")


def test_inline_refs():
    schema = inline_refs(Outer.model_json_schema())
    assert "$defs" not in json.dumps(schema)
    assert schema["properties"]["inner"]["properties"]["x"]["type"] == "integer"


def test_fake_lru_eviction():
    async def go():
        p = make_provider(memory_bytes=5_000_000_000)
        r1 = await p.generate([Message("user", "Ada is 3")], GenerationParams("small"))
        r2 = await p.generate([Message("user", "Ada is 3")], GenerationParams("small"))
        assert r1.usage.load_seconds == 4.0 and r2.usage.load_seconds == 0.0
        await p.generate([Message("user", "Ada is 3")], GenerationParams("large"))
        assert [r.name for r in await p.running()] == ["large"]

    asyncio.run(go())


def test_structured_invalid_output_carries_usage():
    async def go():
        p = make_provider()
        with pytest.raises(InvalidOutput) as ei:
            await generate_structured(p, [Message("user", "Ada is 3")], Outer, "small", temperature=0, max_tokens=50)
        assert ei.value.usage.calls == 1

    asyncio.run(go())
