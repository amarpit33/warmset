import asyncio
import types

from tests import helpers
from warmset import Runtime
from warmset.bench import render_markdown, run_bench, summarize
from warmset.cli import main


def suite() -> types.ModuleType:
    m = types.ModuleType("people")
    m.NAME = "people"
    m.MODELS = {}
    data = [("Ada, 36", "Ada"), ("Bob is 41", "Bob"), ("The tricky case of Cy who is 9", "Cy"),
            ("Dee, 22", "Dee"), ("The tricky case of Eve who is 30", "Eve"), ("Fay is 51", "Fay")]
    tasks = [{"id": f"t{i}", "input": {"text": t}, "expected": n} for i, (t, n) in enumerate(data)]
    m.load_tasks = lambda split: tasks
    m.pipeline = helpers.pipeline
    m.check = lambda out, expected: out.name == expected
    return m


def test_bench_fair_protocol_summary_and_report():
    rt = Runtime(helpers.make_provider(), profile=helpers.make_profile())
    s = suite()
    result = asyncio.run(run_bench(rt, s, ["fixed:small", "fixed:large", "cascade", "adaptive:120"],
                                   split="test", fit_split="dev", repeats=2))
    assert result["fit_actions"] > 0 and rt.learn is False
    assert len(result["rows"]) == 4 * 6 * 2
    summ = summarize(result)
    by = {p["policy"]: p for p in summ["policies"]}
    assert summ["baseline"] == "cascade"
    assert by["cascade"]["success"] == 1.0 and by["fixed:small"]["success"] < 1.0
    assert "vs_baseline" in by["adaptive:120"] and "vs_baseline" not in by["cascade"]
    assert summ["oracle"]["success"] == 1.0
    assert by["adaptive:120"]["vs_baseline"]["p95_diff_ci95"][0] <= by["adaptive:120"]["vs_baseline"]["p95_diff"]
    assert by["cascade"]["makespan_s"] is not None
    md = render_markdown(result, summ)
    assert "| cascade |" in md and "hindsight best" in md


def test_cli_runs_explain_replay_calibrate(tmp_path, capsys):
    db = tmp_path / "l.db"
    rt = Runtime(helpers.make_provider(), profile=helpers.make_profile(), ledger=db)
    res = asyncio.run(rt.run(helpers.pipeline, "The tricky case of Bob who is 41", policy="cascade"))
    rt.ledger.label(res.id, True)
    assert main(["runs", "--db", str(db)]) == 0
    assert main(["explain", res.id, "--db", str(db)]) == 0
    assert main(["replay", res.id, "--policy", "fixed:large", "--db", str(db)]) == 0
    assert main(["calibrate", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert res.id in out and "diverged" in out and '"brier"' in out


def test_batch_policies_reuse_fit_and_pick_value():
    from warmset.bench import pick_value

    rt = Runtime(helpers.make_provider(), profile=helpers.make_profile())
    s = suite()
    first = asyncio.run(run_bench(rt, s, ["cascade"], split="test", fit_split="dev"))
    explore_runs = len(rt.ledger.runs(limit=1000, name="people/dev/explore"))
    result = asyncio.run(run_bench(rt, s, ["cascade", "cascade+batch", "adaptive:120", "adaptive:120+batch",
                                           "adaptive:5"], split="test", fit_split="dev"))
    assert len(rt.ledger.runs(limit=1000, name="people/dev/explore")) == explore_runs  # fit reused
    assert result["fit_actions"] == first["fit_actions"]
    summ = summarize(result)
    by = {p["policy"]: p for p in summ["policies"]}
    assert by["cascade+batch"]["success"] == 1.0
    assert by["cascade+batch"]["loads"] < by["cascade"]["loads"]
    blocks = {b["policy"]: b for b in result["blocks"]}
    assert blocks["cascade+batch"]["switches"] is not None and blocks["cascade"]["switches"] is None
    picked = pick_value(summ)
    assert picked is not None and picked.startswith("adaptive") and picked != "adaptive:5"


def test_drift_detection_flags_slow_blocks():
    from warmset.bench import check_drift
    blocks = [{"exec_median": {"a": 2.0, "b": 8.0}}, {"exec_median": {"a": 2.1, "b": 8.2}},
              {"exec_median": {"a": 2.0, "b": 7.9}}, {"exec_median": {"a": 5.5, "b": 17.0}}]
    assert check_drift(blocks) == {3: 2.68}  # 5.5 / median(2.0, 2.1, 2.0, 5.5) = 5.5 / 2.05
