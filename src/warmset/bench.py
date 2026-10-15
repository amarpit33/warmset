"""Benchmark runner and report.

A suite is a Python module exposing:

    NAME: str
    MODELS: dict[str, str]                 # alias -> default model name
    def load_tasks(split: str) -> list[dict]   # each: {"id", "input": {kwargs}, "expected"}
    async def pipeline(**input) -> Any
    def check(output, expected) -> bool

Protocol (deliberately fair): optionally run ``explore`` on a fit split to estimate pass rates and
latencies, freeze the estimators, then run every policy on the evaluation split with the same
models, verifiers and budget. Models are unloaded before each policy block so no policy inherits a
warm model from another. Results are reported per policy with paired bootstrap confidence
intervals against a baseline (the verify-then-escalate cascade by default).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .runtime import Runtime
from .types import Budget, RunStatus


def load_suite(spec: str) -> ModuleType:
    """Import a suite by module path (``bench.suites.invoices``) or file path."""
    if spec.endswith(".py") or "/" in spec:
        path = Path(spec).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot load suite from {spec}")
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[path.stem] = module
        mod_spec.loader.exec_module(module)
        return module
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module(spec)


@dataclass
class Row:
    task: str
    policy: str
    repeat: int
    run_id: str
    status: str
    correct: bool
    seconds: float
    wall_seconds: float
    calls: int
    tokens: int
    loads: int
    load_seconds: float
    unverified_steps: int


async def _cold(rt: Runtime) -> None:
    if rt.provider.capabilities.can_load_unload:
        for r in await rt.provider.running():
            await rt.provider.unload(r.name)


BATCH_SUFFIX = "+batch"


async def run_policy(rt: Runtime, suite: ModuleType, policy: str, tasks: list[dict[str, Any]], *, split: str,
                     repeat: int = 0, budget: Budget | None = None, cold: bool = True,
                     progress: Any = None, blocks: list[dict[str, Any]] | None = None) -> list[Row]:
    """Run ``policy`` over ``tasks``. A spec ending in ``+batch`` runs all tasks as one residency-scheduled
    batch; otherwise tasks run one after another."""
    if cold:
        await _cold(rt)
    batch = policy.endswith(BATCH_SUFFIX)
    spec = policy[: -len(BATCH_SUFFIX)] if batch else policy
    name = f"{suite.NAME}/{split}/{policy}"
    metas = [{"task": t["id"], "split": split, "repeat": repeat} for t in tasks]
    t0 = time.perf_counter()
    if batch:
        results = await rt.run_batch(suite.pipeline, [t["input"] for t in tasks], policy=spec, budget=budget,
                                     name=name, metas=metas)
    else:
        results = []
        for t, meta in zip(tasks, metas, strict=True):
            results.append(await rt.run(suite.pipeline, policy=spec, budget=budget, name=name, meta=meta,
                                        **t["input"]))
    wall = time.perf_counter() - t0
    rows = []
    for t, res in zip(tasks, results, strict=True):
        correct = bool(res.status == RunStatus.COMPLETED and suite.check(res.result, t["expected"]))
        rt.ledger.label(res.id, correct, "suite-check")
        rows.append(Row(t["id"], policy, repeat, res.id, res.status.value, correct, res.seconds, res.wall_seconds,
                        res.calls, res.tokens, res.loads, res.load_seconds, res.unverified_steps))
        if progress:
            progress(rows[-1])
    if blocks is not None:
        blocks.append({"policy": policy, "repeat": repeat, "wall_seconds": round(wall, 2),
                       "action_seconds": round(sum(r.seconds for r in rows), 2), "loads": sum(r.loads for r in rows),
                       "switches": rt.last_batch_switches if batch else None,
                       "exec_median": _exec_medians(rt, [r.run_id for r in rows])})
        drift = check_drift(blocks)
        if progress and drift.get(len(blocks) - 1):
            progress_warn = getattr(progress, "warn", print)
            progress_warn(f"  WARNING: block {policy} r{repeat} ran {drift[len(blocks) - 1]:.2f}x slower per model "
                          "call than the median block: the machine state changed; compare with care")
    return rows


def _exec_medians(rt: Runtime, run_ids: list[str]) -> dict[str, float]:
    """Median execution seconds (load excluded) per model over a block's actions."""
    per: dict[str, list[float]] = defaultdict(list)
    for rid in run_ids:
        for a in rt.ledger.actions(rid):
            if a["kind"] == "model" and a["ok"]:
                per[a["model"]].append(a["exec_seconds"] or 0.0)
    return {m: round(statistics.median(v), 3) for m, v in per.items()}


DRIFT_THRESHOLD = 1.3


def check_drift(blocks: list[dict[str, Any]], threshold: float = DRIFT_THRESHOLD) -> dict[int, float]:
    """Blocks whose per-model execution speed is ``threshold`` x slower than the median block.

    Policies run in blocks; if the machine slows down mid-benchmark (memory pressure, thermal limits,
    other work), whichever policies ran then look worse. Returns {block index: worst slowdown ratio}.
    """
    per_model: dict[str, list[float]] = defaultdict(list)
    for b in blocks:
        for m, v in b.get("exec_median", {}).items():
            per_model[m].append(v)
    ref = {m: statistics.median(v) for m, v in per_model.items() if v}
    flagged = {}
    for i, b in enumerate(blocks):
        ratios = [v / ref[m] for m, v in b.get("exec_median", {}).items() if ref.get(m)]
        worst = max(ratios, default=1.0)
        if worst >= threshold:
            flagged[i] = round(worst, 2)
    return flagged


async def run_bench(rt: Runtime, suite: ModuleType, policies: list[str], *, split: str = "test",
                    fit_split: str | None = "dev", repeats: int = 1, budget: Budget | None = None,
                    seed: int = 0, limit: int | None = None, progress: Any = None,
                    refit: bool = False) -> dict[str, Any]:
    """Fit on ``fit_split`` (reusing explore runs already in the ledger unless ``refit``), then evaluate."""
    fitted = 0
    if fit_split:
        fit_tasks = suite.load_tasks(fit_split)[:limit]
        fit_name = f"{suite.NAME}/{fit_split}/explore"
        done = [r for r in rt.ledger.runs(limit=100000, name=fit_name) if r["status"] == "completed"]
        if refit or len(done) < len(fit_tasks):
            rt.learn = True
            await run_policy(rt, suite, "explore", fit_tasks, split=fit_split, progress=progress)
        fitted = rt.fit(names=[fit_name])
    rt.learn = False  # frozen estimators: every policy sees the same information
    tasks = suite.load_tasks(split)[:limit]
    rng = random.Random(seed)
    rows: list[Row] = []
    blocks: list[dict[str, Any]] = []
    for rep in range(repeats):
        order = policies[:]
        rng.shuffle(order)
        for pol in order:
            rows += await run_policy(rt, suite, pol, tasks, split=split, repeat=rep, budget=budget,
                                     progress=progress, blocks=blocks)
    drift = check_drift(blocks)
    for i, ratio in drift.items():
        blocks[i]["drift"] = ratio
    return {"suite": suite.NAME, "split": split, "fit_split": fit_split, "fit_actions": fitted,
            "repeats": repeats, "tasks": len(tasks), "models": rt.aliases, "rows": [asdict(r) for r in rows],
            "blocks": blocks}


# ---------------------------------------------------------------- report

def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q / 100 * len(s) + 0.5) - 1))] if s else 0.0


def _per_task(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """task -> (mean correct, mean seconds) averaged over repeats."""
    acc: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        acc[r["task"]].append((float(r["correct"]), r["seconds"]))
    return {t: (statistics.fmean(c for c, _ in v), statistics.fmean(s for _, s in v)) for t, v in acc.items()}


def _bootstrap_p95_diff(mine: list[float], base: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of p95(mine) - p95(base), resampling paired tasks."""
    if not mine:
        return 0.0, 0.0
    rng = random.Random(seed)
    idx = range(len(mine))
    stats = sorted(_pct([mine[i] for i in s], 95) - _pct([base[i] for i in s], 95)
                   for s in (rng.choices(idx, k=len(mine)) for _ in range(n)))
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def _bootstrap(diffs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def summarize(result: dict[str, Any], baseline: str | None = None) -> dict[str, Any]:
    by_pol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in result["rows"]:
        by_pol[r["policy"]].append(r)
    if baseline is None:
        baseline = next((p for p in by_pol if p.startswith("cascade")), next(iter(by_pol), None))
    base_tasks = _per_task(by_pol[baseline]) if baseline in by_pol else {}
    walls: dict[str, list[float]] = defaultdict(list)
    for b in result.get("blocks", []):
        walls[b["policy"]].append(b["wall_seconds"])
    table = []
    for pol, rows in by_pol.items():
        secs = [r["seconds"] for r in rows]
        succ = sum(r["correct"] for r in rows)
        entry: dict[str, Any] = {
            "policy": pol, "runs": len(rows), "success": round(succ / len(rows), 3),
            "seconds_per_success": round(sum(secs) / succ, 2) if succ else None,
            "p50_s": round(_pct(secs, 50), 2), "p95_s": round(_pct(secs, 95), 2),
            "calls": round(statistics.fmean(r["calls"] for r in rows), 2),
            "loads": round(statistics.fmean(r["loads"] for r in rows), 2),
            "load_s": round(statistics.fmean(r["load_seconds"] for r in rows), 2),
            "tokens": round(statistics.fmean(r["tokens"] for r in rows)),
            "unverified": sum(1 for r in rows if r["unverified_steps"]),
            "not_completed": sum(1 for r in rows if r["status"] != "completed"),
            "makespan_s": round(statistics.fmean(walls[pol]), 1) if walls.get(pol) else None}
        if base_tasks and pol != baseline:
            mine = _per_task(rows)
            common = sorted(set(mine) & set(base_tasks))
            d_succ = [mine[t][0] - base_tasks[t][0] for t in common]
            d_secs = [mine[t][1] - base_tasks[t][1] for t in common]
            entry["vs_baseline"] = {
                "success_diff": round(statistics.fmean(d_succ), 3) if d_succ else None,
                "success_diff_ci95": [round(x, 3) for x in _bootstrap(d_succ)],
                "seconds_diff": round(statistics.fmean(d_secs), 2) if d_secs else None,
                "seconds_diff_ci95": [round(x, 2) for x in _bootstrap(d_secs)],
                "p95_diff": round(_pct([mine[t][1] for t in common], 95) - _pct([base_tasks[t][1] for t in common], 95),
                                  2) if common else None,
                "p95_diff_ci95": [round(x, 2) for x in _bootstrap_p95_diff([mine[t][1] for t in common],
                                                                           [base_tasks[t][1] for t in common])]}
        table.append(entry)
    # Hindsight bound over the *evaluated* policies: per task, the fastest policy that got it right.
    # It is not a true optimum (no policy may have found the cheapest path), only a headroom indicator.
    evaluated = {p: _per_task(r) for p, r in by_pol.items()}
    oracle = None
    if len(evaluated) >= 2:
        tasks = set.intersection(*(set(v) for v in evaluated.values()))
        succ, secs = [], []
        for t in tasks:
            ok = [v[t][1] for v in evaluated.values() if v[t][0] >= 0.5]
            succ.append(1.0 if ok else 0.0)
            secs.append(min(ok) if ok else min(v[t][1] for v in evaluated.values()))
        oracle = {"policy": "hindsight best (fastest correct evaluated policy per task)",
                  "success": round(statistics.fmean(succ), 3),
                  "seconds_per_success": round(sum(secs) / sum(succ), 2) if sum(succ) else None,
                  "p50_s": round(_pct(secs, 50), 2), "p95_s": round(_pct(secs, 95), 2)}
    return {"baseline": baseline, "policies": table, "oracle": oracle}


def pick_value(summary: dict[str, Any], *, tolerance: float = 0.02, prefix: str = "adaptive") -> str | None:
    """Model selection for the validation split: among ``prefix`` policies whose success is within
    ``tolerance`` of the baseline's, the one with the lowest seconds per success."""
    base = next((e for e in summary["policies"] if e["policy"] == summary["baseline"]), None)
    if base is None:
        return None
    ok = [e for e in summary["policies"] if e["policy"].startswith(prefix)
          and e["success"] >= base["success"] - tolerance and e["seconds_per_success"] is not None]
    return min(ok, key=lambda e: e["seconds_per_success"])["policy"] if ok else None


def render_markdown(result: dict[str, Any], summary: dict[str, Any]) -> str:
    head = (f"## {result['suite']} / {result['split']}  ({result['tasks']} tasks x {result['repeats']} repeats; "
            f"estimators fitted on {result['fit_split'] or 'nothing'}: {result['fit_actions']} actions)\n\n"
            f"Models: {', '.join(f'{k}={v}' for k, v in result['models'].items())}. "
            f"Seconds = provider-reported action time including model loads (per task: the task's own actions; "
            f"in `+batch` rows tasks also queue, so per-task seconds are cost, not latency). "
            f"makespan = wall time of the whole block. Baseline: `{summary['baseline']}`.\n\n")
    cols = ["policy", "success", "seconds_per_success", "p50_s", "p95_s", "makespan_s", "calls", "loads", "load_s",
            "tokens", "unverified", "not_completed"]
    extra_cols = ["vs baseline: success diff [95% CI]", "seconds/task diff [95% CI]", "p95 diff [95% CI]"]
    lines = ["| " + " | ".join(cols + extra_cols) + " |", "|" + "---|" * (len(cols) + len(extra_cols))]

    def ci(v: float, c: list[float], fmt: str) -> str:
        return f"{v:{fmt}} [{c[0]:{fmt}}, {c[1]:{fmt}}]"

    for e in summary["policies"]:
        vb = e.get("vs_baseline")
        extra = (f"{ci(vb['success_diff'], vb['success_diff_ci95'], '+.3f')} | "
                 f"{ci(vb['seconds_diff'], vb['seconds_diff_ci95'], '+.2f')} | "
                 f"{ci(vb['p95_diff'], vb['p95_diff_ci95'], '+.2f')}" if vb else "baseline | baseline | baseline")
        lines.append("| " + " | ".join("" if e.get(c) is None else str(e.get(c)) for c in cols) + f" | {extra} |")
    if summary["oracle"]:
        o = summary["oracle"]
        lines.append(f"| {o['policy']} | {o['success']} | {o['seconds_per_success']} | {o['p50_s']} | {o['p95_s']} |"
                     + " |" * (len(cols) + len(extra_cols) - 5))
    drifted = [b for b in result.get("blocks", []) if b.get("drift")]
    if drifted:
        lines.append("")
        lines.append(f"**Machine drift:** {len(drifted)} block(s) ran >= {DRIFT_THRESHOLD}x slower per model call than "
                     "the median block; their policies are penalised by machine state, not by their decisions:")
        lines += [f"- {b['policy']} (repeat {b['repeat']}): {b['drift']}x" for b in drifted]
    return head + "\n".join(lines) + "\n"


def save(result: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    return p
