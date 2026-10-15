import asyncio

import pytest

from tests.helpers import extract, make_profile, make_provider, pipeline
from warmset import Model, Runtime, StepContext, step
from warmset.scheduler import ModelScheduler

TEXTS = ["Bob is 41", "The tricky case of Cy who is 9", "Ann is 5", "The tricky case of Dee who is 22",
         "Eve is 30", "The tricky case of Fay who is 51"]


async def _serve(sched: ModelScheduler, requests: list[str]) -> list[str]:
    served: list[str] = []
    gate = asyncio.Event()

    async def call(model: str, hold: bool) -> None:
        async with sched.slot(model):
            served.append(model)
            if hold:
                await gate.wait()
            await asyncio.sleep(0)

    first = asyncio.create_task(call(requests[0], True))
    await asyncio.sleep(0)
    rest = [asyncio.create_task(call(m, False)) for m in requests[1:]]
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(first, *rest)
    return served


def test_residency_first_serves_loaded_model_queue_before_switching():
    sched = ModelScheduler(residency_first=True, switch_grace=0)
    served = asyncio.run(_serve(sched, ["a", "b", "a", "b", "a"]))
    assert served == ["a", "a", "a", "b", "b"] and sched.switches == 1


def test_fifo_keeps_arrival_order():
    sched = ModelScheduler()
    served = asyncio.run(_serve(sched, ["a", "b", "a", "b", "a"]))
    assert served == ["a", "b", "a", "b", "a"] and sched.switches == 4


def test_cancelled_waiter_is_dropped():
    async def go():
        sched = ModelScheduler(residency_first=True, switch_grace=0)
        gate = asyncio.Event()

        async def hold():
            async with sched.slot("a"):
                await gate.wait()

        async def wait_b():
            async with sched.slot("b"):
                pass

        h = asyncio.create_task(hold())
        await asyncio.sleep(0)
        w = asyncio.create_task(wait_b())
        await asyncio.sleep(0)
        w.cancel()
        gate.set()
        await h
        with pytest.raises(asyncio.CancelledError):
            await w
        async with sched.slot("c"):  # scheduler still usable
            pass
        return sched

    sched = asyncio.run(go())
    assert sched._busy == 0 and not sched._waiting


def _batch(schedule: str):
    rt = Runtime(make_provider(), profile=make_profile())
    res = asyncio.run(rt.run_batch(pipeline, [{"text": t} for t in TEXTS], policy="cascade", schedule=schedule))
    return rt, res


def test_batch_residency_schedule_loads_each_model_once():
    rt, res = _batch("residency")
    assert [r.result.name for r in res] == ["Bob", "Cy", "Ann", "Dee", "Eve", "Fay"]
    assert sum(r.loads for r in res) == 2 and rt.last_batch_switches == 1


def test_single_step_batch_groups_calls_even_in_fifo_order():
    # every small call is queued before any escalation arrives, so arrival order already groups them
    _, fifo = _batch("fifo")
    assert sum(r.loads for r in fifo) == 2 and all(r.ok for r in fifo)


@step(alternatives=[Model("small")])
async def shout(ctx: StepContext, name: str) -> str:
    return await ctx.generate(user=f"Say hi to {name}")


async def two_steps(text: str) -> str:
    person = await extract(text)
    return await shout(person.name)


def _two_step(schedule: str):
    rt = Runtime(make_provider(), profile=make_profile())
    res = asyncio.run(rt.run_batch(two_steps, [{"text": t} for t in TEXTS], policy="cascade", schedule=schedule))
    return rt, res


def test_multi_step_batch_residency_first_avoids_ping_pong():
    # step-2 small calls arrive while large escalations are queued: FIFO swaps back and forth
    _, fifo = _two_step("fifo")
    rt, resid = _two_step("residency")
    assert all(r.ok for r in fifo + resid)
    assert sum(r.loads for r in resid) == 3 and rt.last_batch_switches == 2
    assert sum(r.loads for r in fifo) > sum(r.loads for r in resid)
    assert sum(r.seconds for r in fifo) > sum(r.seconds for r in resid)


def test_batch_amortises_load_in_decisions():
    rt = Runtime(make_provider(), profile=make_profile())
    res = asyncio.run(rt.run_batch(pipeline, [{"text": t} for t in TEXTS], policy="cascade"))
    import json
    view = json.loads(rt.ledger.decisions(res[0].id)[1]["view_json"])
    large = next(c for c in view["candidates"] if c["model"] == "large")
    assert large["est_load_seconds"] < 20.0  # 20s load shared across runs still in flight


def test_batch_rejects_unknown_schedule():
    rt = Runtime(make_provider(), profile=make_profile())
    with pytest.raises(ValueError):
        asyncio.run(rt.run_batch(pipeline, [], schedule="random"))
