import asyncio

from tests.helpers import make_profile, make_provider, pipeline
from warmset import Adaptive, Cascade, Fixed, Runtime
from warmset.lab import branch, calibration, explain, replay_decisions

TRICKY = "The tricky case of Bob who is 41"


def rt() -> Runtime:
    return Runtime(make_provider(), profile=make_profile())


def test_explain_shows_candidates_choice_and_observation():
    r = rt()
    res = asyncio.run(r.run(pipeline, TRICKY, policy="cascade"))
    text = explain(r.ledger, res.id[:6])
    assert "#1 extract  -> accepted via model:large@t0" in text
    assert "model:small@t0" in text and "grounded(name)=fail" in text
    assert "predicted" in text and "load" in text


def test_replay_same_policy_no_divergence_other_policy_diverges():
    r = rt()
    res = asyncio.run(r.run(pipeline, TRICKY, policy="cascade"))
    rows, verdict = replay_decisions(r.ledger, res.id, Cascade())
    assert all(x.same for x in rows) and "no divergence" in verdict
    rows, verdict = replay_decisions(r.ledger, res.id, Fixed("large"))
    assert not rows[-1].same and "diverged" in verdict and len(rows) == 1


def test_branch_is_observed_and_labelled():
    r = rt()
    out = asyncio.run(branch(r, pipeline, [TRICKY], base=Cascade(),
                             force={("extract", 1): "model:large@t0"}, controls=2,
                             check=lambda p: p.name == "Bob"))
    assert out["kind"].startswith("observed")
    assert out["forced"]["correct"] is True and len(out["controls"]) == 2
    first_action = r.ledger.actions(out["forced"]["run_id"])[0]
    assert first_action["action_key"] == "model:large@t0"


def test_calibration_reports_brier_and_verified_correct_rate():
    r = rt()

    async def go():
        for text in ["Bob is 41", TRICKY, "Ann is 5", TRICKY]:
            res = await r.run(pipeline, text, policy="fixed:small")
            r.ledger.label(res.id, res.result.name in text)

    asyncio.run(go())
    cal = calibration(r.ledger)
    assert cal["actions"] == 4 and 0 <= cal["brier"] <= 1
    assert cal["verified_runs"] == {"n": 2, "correct_rate": 1.0}
    assert cal["unverified_runs"] == {"n": 2, "correct_rate": 0.0}


def test_feature_buckets_back_off_then_specialise():
    from tests.helpers import Person, regex_person
    from warmset import Deterministic, Model, StepContext, grounded, step

    @step(alternatives=[Deterministic(regex_person), Model("small"), Model("large")], verify=[grounded("name")],
          features=lambda text: "tricky" if "tricky" in text else "plain")
    async def ex(ctx: StepContext, text: str) -> Person:
        return await ctx.generate(user=text, schema=Person)

    async def pipe(text):
        return await ex(text)

    r = rt()

    async def go():
        for i in range(4):
            await r.run(pipe, f"The tricky case of Bob who is {40 + i}", policy="explore", name="dev")
            await r.run(pipe, f"Ann is {20 + i}", policy="explore", name="dev")
        r.fit(names=["dev"])
        r.learn = False
        pol = Adaptive(value_seconds=100, min_samples=3)
        for m in ("small", "large"):  # cold machine: otherwise the resident large model is the cheap choice
            await r.provider.unload(m)
        return await r.run(pipe, "Cal is 7", policy=pol), await r.run(pipe, TRICKY, policy=pol)

    plain, tricky = asyncio.run(go())
    assert r.success.estimate("ex", "model:small@t0", "plain")[2] == "bucket:plain"
    assert r.success.estimate("ex", "model:small@t0", "unseen")[2] == "step"
    first = lambda res: [a for a in r.ledger.actions(res.id) if a["kind"] == "model"][0]["action_key"]  # noqa: E731
    assert first(plain) == "model:small@t0"   # small passes plain inputs: no need for the big model
    assert first(tricky) == "model:large@t0"  # small fails tricky inputs: go straight to large
    assert "[input bucket: tricky]" in explain(r.ledger, tricky.id)


def test_fit_from_ledger_then_adaptive_uses_estimates():
    r = rt()

    async def go():
        for i in range(6):
            await r.run(pipeline, f"The tricky case of Bob who is {40 + i}", policy="explore", name="dev")
        r.fit(names=["dev"])
        r.learn = False
        return await r.run(pipeline, TRICKY, policy=Adaptive(value_seconds=300, min_samples=5), name="eval")

    res = asyncio.run(go())
    assert r.success.mean("extract", "model:small@t0") < 0.2
    assert r.success.n("extract", "fn:regex_person") == 0  # not-applicable outcomes are not evidence
    first_model = [a for a in r.ledger.actions(res.id) if a["kind"] == "model"][0]
    assert first_model["action_key"] == "model:large@t0"  # skipped the small model it learned fails here
    assert res.result.name == "Bob"
