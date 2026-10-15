"""Command line: profile hardware, run benchmarks, explain and replay runs, check calibration."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import bench as bench_mod
from .hardware import DEFAULT_PROFILE_PATH, HardwareProfile, measure_model
from .lab import calibration, explain, replay_decisions
from .ledger import Ledger
from .policies import parse_policy
from .providers import create_provider
from .runtime import Runtime
from .types import Budget

DEFAULT_DB = ".warmset/ledger.db"


def _kv(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        k, sep, v = p.partition("=")
        if not sep:
            raise SystemExit(f"expected alias=model, got '{p}'")
        out[k] = v
    return out


async def _profile(args: argparse.Namespace) -> int:
    provider = create_provider(args.backend, args.base_url)
    try:
        if not await provider.health():
            print(f"{args.backend} is not reachable", file=sys.stderr)
            return 1
        models = args.models or [m.name for m in await provider.list_models() if not m.is_embedding]
        profile = HardwareProfile.load_or_detect(args.out)
        for m in models:
            print(f"measuring {m} (unloads everything first) ...", flush=True)
            mp = await measure_model(provider, m)
            profile.models[m] = mp
            print(f"  load {mp.load_seconds:.1f}s  prefill {mp.prefill_tps:.0f} tok/s  decode {mp.decode_tps:.1f} tok/s"
                  f"  size {mp.size_bytes / 1e9:.1f} GB")
        path = profile.save(args.out)
        print(f"saved {path}")
        return 0
    finally:
        await provider.aclose()


async def _bench(args: argparse.Namespace) -> int:
    suite = bench_mod.load_suite(args.suite)
    models = {**getattr(suite, "MODELS", {}), **_kv(args.model)}
    provider = create_provider(args.backend, args.base_url)
    profile = HardwareProfile.load_or_detect(args.profile)
    rt = Runtime(provider, models=models, ledger=args.db, profile=profile, action_timeout=args.timeout)
    budget = Budget(seconds=args.budget_seconds) if args.budget_seconds else None

    def progress(row: bench_mod.Row) -> None:
        print(f"  {row.policy:<28} {row.task:<10} {'ok ' if row.correct else 'BAD'} {row.seconds:7.2f}s "
              f"calls={row.calls} loads={row.loads}", flush=True)

    try:
        result = await bench_mod.run_bench(rt, suite, args.policies, split=args.split,
                                           fit_split=None if args.fit_split == "none" else args.fit_split,
                                           repeats=args.repeats, budget=budget, limit=args.limit,
                                           progress=progress, refit=args.refit)
    finally:
        await provider.aclose()
    summary = bench_mod.summarize(result, args.baseline)
    result["summary"] = summary
    if args.pick:
        result["picked"] = bench_mod.pick_value(summary, tolerance=args.tolerance)
    if args.out:
        print(f"saved {bench_mod.save(result, args.out)}")
    print()
    print(bench_mod.render_markdown(result, summary))
    if args.pick:
        print(f"picked (lowest seconds/success within {args.tolerance:g} of baseline success): {result['picked']}")
    return 0


def _report(args: argparse.Namespace) -> int:
    result = json.loads(Path(args.result).read_text())
    print(bench_mod.render_markdown(result, bench_mod.summarize(result, args.baseline)))
    return 0


def _runs(args: argparse.Namespace) -> int:
    for r in Ledger(args.db).runs(args.limit, args.name):
        label = {None: "", 1: "correct", 0: "incorrect"}[r["correct"]]
        print(f"{r['id']}  {r['status']:<15} {r['policy']:<24} {(r['seconds'] or 0):7.2f}s  "
              f"calls={r['calls']:<3} {r['name'] or ''}  {label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="warmset", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def backend(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--backend", default="ollama", help="ollama | openai")
        sp.add_argument("--base-url", default=None)

    sp = sub.add_parser("profile", help="measure model load time and throughput on this machine")
    backend(sp)
    sp.add_argument("--models", nargs="*")
    sp.add_argument("--out", default=str(DEFAULT_PROFILE_PATH))

    sp = sub.add_parser("bench", help="run a benchmark suite under several policies")
    backend(sp)
    sp.add_argument("suite", help="module path (bench.suites.invoices) or .py file")
    sp.add_argument("--policies", nargs="+", required=True)
    sp.add_argument("--model", action="append", default=[], help="alias=model, repeatable")
    sp.add_argument("--split", default="test")
    sp.add_argument("--fit-split", default="dev", help="split for the explore pass, or 'none'")
    sp.add_argument("--repeats", type=int, default=1)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--budget-seconds", type=float, default=None)
    sp.add_argument("--timeout", type=float, default=300.0)
    sp.add_argument("--baseline", default=None)
    sp.add_argument("--pick", action="store_true", help="report the adaptive policy to carry forward (use on val)")
    sp.add_argument("--tolerance", type=float, default=0.02)
    sp.add_argument("--refit", action="store_true", help="re-run explore even if the ledger already has it")
    sp.add_argument("--profile", default=None)
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("report", help="re-render a saved benchmark result")
    sp.add_argument("result")
    sp.add_argument("--baseline", default=None)

    sp = sub.add_parser("runs", help="list recent runs")
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--limit", type=int, default=30)
    sp.add_argument("--name", default=None)

    sp = sub.add_parser("explain", help="every decision in a run: candidates, estimates, choice, outcome")
    sp.add_argument("run_id")
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--step", default=None)

    sp = sub.add_parser("replay", help="deterministic replay of a run's decisions under another policy")
    sp.add_argument("run_id")
    sp.add_argument("--policy", required=True)
    sp.add_argument("--db", default=DEFAULT_DB)

    sp = sub.add_parser("calibrate", help="predicted vs observed pass rates; verified vs correct")
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--name", action="append", default=None)

    args = p.parse_args(argv)
    if args.cmd == "profile":
        return asyncio.run(_profile(args))
    if args.cmd == "bench":
        return asyncio.run(_bench(args))
    if args.cmd == "report":
        return _report(args)
    if args.cmd == "runs":
        return _runs(args)
    if args.cmd == "explain":
        print(explain(Ledger(args.db), args.run_id, args.step))
        return 0
    if args.cmd == "replay":
        ledger = Ledger(args.db)
        run = ledger.run(args.run_id)
        if run is None:
            print(f"no run '{args.run_id}'", file=sys.stderr)
            return 1
        rows, verdict = replay_decisions(ledger, run["id"], parse_policy(args.policy))
        for r in rows:
            print(f"{'=' if r.same else '!'} {r.step:<20} #{r.attempt}  recorded={r.recorded}  replayed={r.replayed}")
        print(verdict)
        return 0
    if args.cmd == "calibrate":
        print(json.dumps(calibration(Ledger(args.db), names=args.name), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
