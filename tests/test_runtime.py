import asyncio

import pytest

from tests.helpers import Person, extract, make_profile, make_provider, pipeline
from warmset import Budget, Deterministic, Model, NoRunContext, RunStatus, Runtime, StepContext, nonempty, step
from warmset.errors import NoRunContext as _NRC


def rt(**kw) -> Runtime:
    return Runtime(make_provider(), profile=make_profile(), **kw)


def test_deterministic_path_needs_no_model():
    r = rt()
    res = asyncio.run(r.run(pipeline, "Ada, 36"))
    assert res.ok and res.result == Person(name="Ada", age=36)
    assert res.calls == 0
    assert r.provider.calls == []


def test_cascade_escalates_on_failed_verification_and_records_everything():
    r = rt()
    res = asyncio.run(r.run(pipeline, "The tricky case of Bob who is 41", policy="cascade"))
    assert res.ok and res.result.name == "Bob"
    actions = r.ledger.actions(res.id)
    assert [a["action_key"] for a in actions] == ["fn:regex_person", "model:small@t0", "model:large@t0"]
    assert actions[0]["error"].startswith("not applicable")
    assert actions[1]["ok"] == 1 and actions[1]["passed"] == 0
    assert actions[1]["verdicts"][0]["outcome"] == "fail"
    assert actions[2]["passed"] == 1
    assert res.loads == 2 and res.load_seconds == pytest.approx(24.0)
    step_row = r.ledger.steps(res.id)[0]
    assert step_row["status"] == "accepted" and step_row["accepted_key"] == "model:large@t0"


def test_fixed_policy_returns_unverified_output():
    r = rt()
    res = asyncio.run(r.run(pipeline, "The tricky case of Bob who is 41", policy="fixed:small"))
    assert res.ok and res.result.name == "Zed"
    assert res.unverified_steps == 1
    assert r.ledger.steps(res.id)[0]["status"] == "unverified"


def test_budget_prices_out_the_large_model():
    r = rt()
    res = asyncio.run(r.run(pipeline, "The tricky case of Bob who is 41", policy="cascade",
                            budget=Budget(seconds=12)))
    # small fits (4s load + exec), large (20s load) does not: keep the unverified small answer
    assert res.ok and res.result.name == "Zed" and res.unverified_steps == 1
    assert [a["action_key"] for a in r.ledger.actions(res.id)][-1] == "model:small@t0"
    assert res.seconds <= 12


def test_budget_exhausted_without_output_stops_run():
    r = rt(allow_unverified=False)
    res = asyncio.run(r.run(pipeline, "The tricky case of Bob who is 41", policy="cascade",
                            budget=Budget(seconds=12)))
    assert res.status == RunStatus.STOPPED_BUDGET


def test_model_call_budget():
    r = rt()
    res = asyncio.run(r.run(pipeline, "The tricky case of Bob who is 41", policy="cascade",
                            budget=Budget(model_calls=1)))
    assert res.calls == 1 and res.result.name == "Zed"


def test_external_side_effects_are_never_retried():
    sent = []

    def send(msg: str) -> str:
        sent.append(msg)
        return ""  # fails nonempty -> a normal policy would retry

    @step(alternatives=[Deterministic(send), Deterministic(send, name="send_again")], verify=[nonempty()],
          side_effects="external")
    async def notify(ctx: StepContext, msg: str) -> str:
        raise AssertionError("not used")

    async def pipe() -> str:
        return await notify("hi")

    r = rt()
    res = asyncio.run(r.run(pipe, policy="cascade"))
    assert sent == ["hi"]
    assert res.unverified_steps == 1
    reasons = [d["reason"] for d in r.ledger.decisions(res.id)]
    assert "external side effects" in reasons[-1]


def test_step_outside_run_raises():
    with pytest.raises(_NRC):
        asyncio.run(extract("Ada, 36"))
    assert NoRunContext is _NRC


def test_explore_runs_every_alternative_and_keeps_first_verified():
    r = rt()
    res = asyncio.run(r.run(pipeline, "Bob is 41", policy="explore"))
    assert [a["action_key"] for a in r.ledger.actions(res.id)] == \
        ["fn:regex_person", "model:small@t0", "model:large@t0"]
    assert r.ledger.steps(res.id)[0]["accepted_key"] == "model:small@t0"
    assert res.result.name == "Bob"


def test_events_stream_in_order_and_cancel():
    async def go():
        r = rt()
        handle = r.start(pipeline, "The tricky case of Bob who is 41", policy="cascade")
        types = [e.type async for e in handle.events()]
        res = await handle.result()
        assert types[0] == "run_started" and types[-1] == "run_finished"
        assert types.count("decision") == 3 and types.count("action_finished") == 3

        async def slow(text: str):
            await asyncio.sleep(10)

        h2 = r.start(slow, "x")
        await asyncio.sleep(0.01)
        h2.cancel()
        res2 = await h2.result()
        return res, res2

    res, res2 = asyncio.run(go())
    assert res.ok
    assert res2.status == RunStatus.CANCELLED


def test_action_timeout_is_a_failed_action():
    @step(alternatives=[Model("small"), Model("large")])
    async def hang(ctx: StepContext) -> str:
        if ctx.model == "small":
            await asyncio.sleep(5)
        return await ctx.generate(user="Ada is 3")

    async def pipe() -> str:
        return await hang()

    r = rt(action_timeout=0.05)
    res = asyncio.run(r.run(pipe, policy="cascade"))
    actions = r.ledger.actions(res.id)
    assert actions[0]["error"].startswith("timeout")
    assert res.ok and actions[1]["ok"] == 1


def test_invalid_output_is_failed_action_with_usage():
    @step(alternatives=[Model("small")])
    async def bad(ctx: StepContext) -> Person:
        return await ctx.generate(user="Ada is 3", schema=Person, max_tokens=2)

    async def pipe():
        return await bad()

    r = rt()
    res = asyncio.run(r.run(pipe, policy="cascade"))
    assert res.status == RunStatus.FAILED
    a = r.ledger.actions(res.id)[0]
    assert a["ok"] == 0 and "InvalidOutput" in a["error"] and a["calls"] == 2  # truncation retry, then invalid


def test_bug_in_step_body_is_recorded_then_raised_not_escalated():
    @step(alternatives=[Model("small"), Model("large")])
    async def buggy(ctx: StepContext) -> str:
        return {}["missing"]

    async def pipe():
        return await buggy()

    r = rt()
    res = asyncio.run(r.run(pipe, policy="cascade"))
    assert res.status == RunStatus.FAILED and "KeyError" in res.error
    actions = r.ledger.actions(res.id)
    assert len(actions) == 1 and "KeyError" in actions[0]["error"]


def test_aliases_resolve():
    r = Runtime(make_provider(), profile=make_profile(), models={"s": "small", "l": "large"})

    @step(alternatives=[Model("s"), Model("l")])
    async def get(ctx: StepContext, text: str) -> Person:
        return await ctx.generate(user=text, schema=Person)

    async def pipe(text):
        return await get(text)

    res = asyncio.run(r.run(pipe, "Ada is 3", policy="fixed:l"))
    assert r.ledger.actions(res.id)[0]["action_key"] == "model:large@t0"
