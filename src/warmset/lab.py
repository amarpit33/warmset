"""Decision lab: explain runs, replay decisions under another policy, branch reruns, calibration.

Three kinds of "what if", never mixed up:

* ``replay_decisions``: deterministic replay. Feeds the *recorded* decision views to another
  policy. Exact up to the first decision where the policies differ; after that the recorded
  observations no longer describe what the other policy would have seen, so replay stops there.
* ``branch``: controlled rerun. Forces an alternative at a decision point and actually runs the
  pipeline again, next to unforced control reruns that measure run-to-run noise. Observed data.
* Anything model-based (estimating outcomes that were never run) is an *estimate* and must be
  labelled as one. Static replay of model swaps is known to mislead (Replay Gap, arXiv 2608.08239).
"""
from __future__ import annotations

import json
import statistics
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .ledger import Ledger
from .policies import Choice, DecisionView, Policy
from .types import Verdict


def _fmt_s(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}s"


def explain(ledger: Ledger, run_id: str, step: str | None = None) -> str:
    """Human-readable account of every decision in a run: what was considered, chosen, and observed."""
    run = ledger.run(run_id)
    if run is None:
        return f"no run '{run_id}'"
    rid = run["id"]
    label = {None: "unlabelled", 1: "correct", 0: "incorrect"}[run.get("correct")]
    lines = [f"run {rid}  policy={run['policy']}  status={run['status']}  {label}",
             f"  action time {_fmt_s(run['seconds'])} (model load {_fmt_s(run['load_seconds'])}, {run['loads']} loads)"
             f"  wall {_fmt_s(run['wall_seconds'])}  calls={run['calls']}  tokens={run['tokens']}"
             f"  unverified steps={run['unverified_steps']}"]
    if run.get("stop_reason"):
        lines.append(f"  stop reason: {run['stop_reason']}")
    actions = {a["decision_id"]: a for a in ledger.actions(rid)}
    steps = {s["id"]: s for s in ledger.steps(rid)}
    current = None
    for d in ledger.decisions(rid):
        if step and d["step_name"] != step:
            continue
        if d["step_id"] != current:
            current = d["step_id"]
            s = steps[current]
            lines.append(f"\n#{s['seq']} {s['name']}  -> {s['status']}" + (f" via {s['accepted_key']}"
                                                                          if s["accepted_key"] else ""))
        view = json.loads(d["view_json"])
        lines.append(f"  decision {d['attempt']}: {d['chosen_key'] or 'STOP'}"
                     + (f"   [input bucket: {view['bucket']}]" if view.get("bucket") else ""))
        lines.append(f"    why: {d['reason']}")
        for c in view["candidates"]:
            flags = [f for f, on in (("resident", c["resident"] is True), ("tried", c["tries"] > 0),
                                     ("exhausted", c["exhausted"]), ("over budget", not c["affordable"])) if on]
            lines.append(f"    - {c['key']:<40} p={c['p_success']:.2f} (n={c['n']}, {c.get('p_source', 'step')})"
                         f"  est {c['est_seconds']:.2f}s"
                         f" [load {c['est_load_seconds']:.1f} + evict {c['est_evict_seconds']:.1f}]"
                         + (f"  {', '.join(flags)}" if flags else ""))
        a = actions.get(d["id"])
        if a:
            verdicts = ", ".join(f"{v['verifier']}={v['outcome']}" + (f" ({v['evidence']})" if v["evidence"] else "")
                                 for v in a["verdicts"])
            lines.append(f"    observed: {'ok' if a['ok'] else 'error'} in {_fmt_s(a['seconds'])}"
                         f" (load {_fmt_s(a['load_seconds'])}; predicted {_fmt_s(a['predicted_seconds'])}),"
                         f" {a['prompt_tokens']}+{a['completion_tokens']} tokens -> {verdicts}")
    return "\n".join(lines)


@dataclass
class ReplayRow:
    step: str
    attempt: int
    recorded: str | None
    replayed: str | None

    @property
    def same(self) -> bool:
        return self.recorded == self.replayed


def replay_decisions(ledger: Ledger, run_id: str, policy: Policy) -> tuple[list[ReplayRow], str]:
    """Deterministic replay of recorded decision views under ``policy``, up to the first divergence."""
    rows: list[ReplayRow] = []
    for d in ledger.decisions(run_id):
        view = DecisionView.from_dict(json.loads(d["view_json"]))
        choice: Choice = policy.choose(view)
        rows.append(ReplayRow(d["step_name"], d["attempt"], d["chosen_key"], choice.key))
        if choice.key != d["chosen_key"]:
            return rows, (f"diverged at {d['step_name']} decision {d['attempt']}: recorded {d['chosen_key']}, "
                          f"{policy.name} would choose {choice.key}. Later observations are not valid for "
                          "this policy; use branch() to observe what actually happens.")
    return rows, "no divergence: this policy makes identical decisions on this run"


class Forced:
    """Wrap a policy, forcing ``key`` at (step name, attempt). Applies to every call of that step."""

    def __init__(self, base: Policy, force: dict[tuple[str, int], str]) -> None:
        self.base = base
        self.force = force
        self.name = f"{base.name}+forced"

    def choose(self, view: DecisionView) -> Choice:
        key = self.force.get((view.step, view.attempt))
        if key is not None:
            return Choice(key, f"forced by branch: {key}")
        return self.base.choose(view)

    def accept(self, view: DecisionView, verdicts: Sequence[Verdict]) -> bool:
        return self.base.accept(view, verdicts)


async def branch(runtime: Any, pipeline: Callable[..., Awaitable[Any]], args: Sequence[Any], *,
                 base: Policy, force: dict[tuple[str, int], str], controls: int = 2,
                 check: Callable[[Any], bool] | None = None) -> dict[str, Any]:
    """Controlled rerun: one forced branch, ``controls`` unforced reruns. All results are observed."""
    async def one(pol: Policy, tag: str) -> dict[str, Any]:
        r = await runtime.run(pipeline, *args, policy=pol, name=f"lab/{tag}")
        correct = check(r.result) if (check and r.ok) else None
        if correct is not None:
            runtime.ledger.label(r.id, correct, "branch-check")
        return {"run_id": r.id, "status": r.status.value, "seconds": round(r.seconds, 3), "correct": correct}

    forced = await one(Forced(base, force), "branch")
    ctrl = [await one(base, "control") for _ in range(controls)]
    secs = [c["seconds"] for c in ctrl]
    return {"kind": "observed (branching rerun)", "forced": forced, "controls": ctrl,
            "control_seconds_median": statistics.median(secs) if secs else None,
            "control_seconds_spread": (max(secs) - min(secs)) if secs else None}


def calibration(ledger: Ledger, *, names: Sequence[str] | None = None, bins: int = 5) -> dict[str, Any]:
    """How well did predicted pass probabilities match observed verifier outcomes?

    Also reports, for labelled runs, how often "every step verified" actually meant correct.
    """
    rows = [r for r in ledger.outcome_rows() if (names is None or r["run_name"] in names)
            and not (r["error"] or "").startswith("not applicable") and r["predicted_p"] is not None]
    out: dict[str, Any] = {"actions": len(rows)}
    if rows:
        brier = statistics.fmean((r["predicted_p"] - r["passed"]) ** 2 for r in rows)
        table, ece = [], 0.0
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            inb = [r for r in rows if lo <= r["predicted_p"] < hi or (b == bins - 1 and r["predicted_p"] == 1.0)]
            if not inb:
                continue
            pred = statistics.fmean(r["predicted_p"] for r in inb)
            obs = statistics.fmean(r["passed"] for r in inb)
            ece += len(inb) / len(rows) * abs(pred - obs)
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(inb), "predicted": round(pred, 3),
                          "observed": round(obs, 3)})
        out.update(brier=round(brier, 4), ece=round(ece, 4), bins=table)
    runs = [r for r in ledger.runs(limit=100000) if r["correct"] is not None and r["status"] == "completed"
            and (names is None or r["name"] in names)]
    verified = [r for r in runs if r["unverified_steps"] == 0]
    unverified = [r for r in runs if r["unverified_steps"] > 0]
    out["verified_runs"] = {"n": len(verified), "correct_rate":
                            round(statistics.fmean(r["correct"] for r in verified), 3) if verified else None}
    out["unverified_runs"] = {"n": len(unverified), "correct_rate":
                              round(statistics.fmean(r["correct"] for r in unverified), 3) if unverified else None}
    return out
