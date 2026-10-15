"""Quickstart: one step, three alternatives, verified output, and an explanation of every decision.

    python examples/quickstart.py           # needs Ollama with llama3.2 and qwen3.5:4b (or edit MODELS)
    python examples/quickstart.py --fake    # no model server: simulated models
"""
from __future__ import annotations

import asyncio
import json
import re
import sys

from pydantic import BaseModel

from warmset import Budget, Deterministic, Model, NotApplicable, Runtime, StepContext, grounded, numbers_preserved, step
from warmset.hardware import HardwareProfile, ModelProfile
from warmset.lab import explain
from warmset.providers import FakeModel, FakeProvider, OllamaProvider

MODELS = {"small": "llama3.2:latest", "large": "qwen3.5:4b"}


class Order(BaseModel):
    customer: str
    quantity: int
    product: str


def parse_simple(text: str) -> Order:
    """Handles 'Customer: X / Qty: N / Product: P' without any model."""
    m = re.search(r"Customer: (.+)\nQty: (\d+)\nProduct: (.+)", text)
    if not m:
        raise NotApplicable("free-text order")
    return Order(customer=m[1].strip(), quantity=int(m[2]), product=m[3].strip())


@step(alternatives=[Deterministic(parse_simple), Model("small"), Model("large")],
      verify=[grounded("customer"), grounded("product"), numbers_preserved("quantity")])
async def read_order(ctx: StepContext, text: str) -> Order:
    return await ctx.generate(system="Extract the order: who ordered, how many, which product.",
                              user=text, schema=Order, max_tokens=120)


async def pipeline(text: str) -> Order:
    return await read_order(text)


def fake_runtime() -> Runtime:
    def answer(messages, params):
        text = messages[-1].content
        qty = re.search(r"\d+", text)
        return json.dumps({"customer": "Priya Shah", "quantity": int(qty.group()) if qty else 1,
                           "product": "standing desks" if params.model == "large" else "desks"})

    provider = FakeProvider({"small": FakeModel(answer, 2 * 10**9, 5.0, 230, 26),
                             "large": FakeModel(answer, 4 * 10**9, 6.0, 120, 16)}, memory_bytes=5 * 10**9)
    profile = HardwareProfile(8 * 10**9, 5 * 10**9, {"small": ModelProfile(2 * 10**9, 5.0, 230, 26),
                                                     "large": ModelProfile(4 * 10**9, 6.0, 120, 16)})
    return Runtime(provider, profile=profile)


async def main(fake: bool) -> None:
    rt = fake_runtime() if fake else Runtime(OllamaProvider(), models=MODELS,
                                             profile=HardwareProfile.load_or_detect(".warmset/profile.json"))
    for text in ["Customer: Ada Lovelace\nQty: 3\nProduct: brass gears",
                 "Hi, Priya Shah here - please send 12 standing desks to our new office."]:
        handle = rt.start(pipeline, text, policy="cascade", budget=Budget(seconds=120))
        async for event in handle.events():
            if event.type == "decision":
                print(f"  decision: {event.data['chosen'] or 'STOP'}  ({event.data['reason']})")
        res = await handle.result()
        print(f"{res.status.value}: {res.result}  [{res.seconds:.2f}s, {res.calls} model calls]\n")
    print(explain(rt.ledger, res.id))
    await rt.provider.aclose()


if __name__ == "__main__":
    asyncio.run(main("--fake" in sys.argv))
