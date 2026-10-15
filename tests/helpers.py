"""Shared fixtures: a small extraction pipeline over a simulated two-model machine."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel

from warmset import Deterministic, Model, NotApplicable, StepContext, grounded, step
from warmset.hardware import HardwareProfile, ModelProfile
from warmset.providers import FakeModel, FakeProvider
from warmset.providers.base import GenerationParams, Message

GB = 1_000_000_000


class Person(BaseModel):
    name: str
    age: int


def regex_person(text: str) -> Person:
    m = re.fullmatch(r"(\w+), (\d+)", text.strip())
    if not m:
        raise NotApplicable("not 'Name, age'")
    return Person(name=m.group(1), age=int(m.group(2)))


def _user(messages: list[Message]) -> str:
    return messages[-1].content


def small_responder(messages: list[Message], params: GenerationParams) -> str:
    """Gets the age right but invents a name whenever the text says 'tricky'."""
    text = _user(messages)
    age = int(re.search(r"\d+", text).group()) if re.search(r"\d+", text) else 0
    name = "Zed" if "tricky" in text else re.search(r"[A-Z][a-z]+", text).group()
    return json.dumps({"name": name, "age": age})


def large_responder(messages: list[Message], params: GenerationParams) -> str:
    text = _user(messages)
    names = [w for w in re.findall(r"[A-Z][a-z]+", text) if w not in ("The", "Tricky")]
    return json.dumps({"name": names[0], "age": int(re.search(r"\d+", text).group())})


def make_provider(memory_bytes: int | None = 5 * GB) -> FakeProvider:
    return FakeProvider({
        "small": FakeModel(small_responder, size_bytes=2 * GB, load_seconds=4.0, prefill_tps=400, decode_tps=40),
        "large": FakeModel(large_responder, size_bytes=4 * GB, load_seconds=20.0, prefill_tps=100, decode_tps=10),
    }, memory_bytes=memory_bytes)


def make_profile() -> HardwareProfile:
    return HardwareProfile(8 * GB, 5 * GB, {
        "small": ModelProfile(2 * GB, 4.0, 400, 40),
        "large": ModelProfile(4 * GB, 20.0, 100, 10),
    })


@step(alternatives=[Deterministic(regex_person), Model("small"), Model("large")],
      verify=[grounded("name")])
async def extract(ctx: StepContext, text: str) -> Person:
    return await ctx.generate(system="Extract the person.", user=text, schema=Person)


async def pipeline(text: str) -> Person:
    return await extract(text)
