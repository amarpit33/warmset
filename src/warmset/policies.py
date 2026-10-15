"""Decision policies.

A policy sees a ``DecisionView`` (eligible candidates with cost and success estimates, the
attempts so far, the budget left) and returns a ``Choice``: run an action, or stop. Views are
plain serialisable data, so recorded decisions can be re-run against another policy offline.

Baselines are deliberately strong (see "AI Agents That Matter", Kapoor et al. 2024): the
verify-then-escalate cascade is the one the adaptive policy has to beat.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .types import Verdict


@dataclass
class Candidate:
    key: str                       # stable id, e.g. "model:llama3.2:latest@t0" or "fn:regex_parse"
    kind: str                      # "model" | "fn"
    model: str | None = None
    temperature: float = 0.0
    est_seconds: float = 0.0       # load + evict penalty + execution
    est_load_seconds: float = 0.0
    est_evict_seconds: float = 0.0
    p_success: float = 0.5         # estimated probability the output passes verification
    n: int = 0                     # observations behind p_success
    p_source: str = "step"         # "step" (step-wide rate) or "bucket:<label>"
    resident: bool | None = None   # None: the backend cannot tell
    tries: int = 0
    exhausted: bool = False        # retrying would repeat the same result (deterministic fn, or t=0 model)
    affordable: bool = True        # fits the remaining budget


@dataclass
class Attempt:
    key: str
    temperature: float
    ok: bool        # produced an output
    passed: bool    # output passed every verifier


@dataclass
class DecisionView:
    step: str
    attempt: int
    candidates: list[Candidate]
    history: list[Attempt] = field(default_factory=list)
    remaining_seconds: float | None = None
    remaining_calls: int | None = None
    side_effects: str = "none"
    bucket: str | None = None      # feature bucket of this step's input, if the step declares features

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecisionView:
        return cls(d["step"], d["attempt"], [Candidate(**c) for c in d["candidates"]],
                   [Attempt(**a) for a in d.get("history", [])], d.get("remaining_seconds"),
                   d.get("remaining_calls"), d.get("side_effects", "none"), d.get("bucket"))

    def open(self) -> list[Candidate]:
        """Candidates worth trying: not exhausted and affordable."""
        return [c for c in self.candidates if not c.exhausted and c.affordable]

    def find_model(self, model: str) -> Candidate | None:
        return next((c for c in self.candidates if c.kind == "model" and c.model == model), None)


@dataclass(frozen=True)
class Choice:
    key: str | None              # None = stop
    reason: str
    temperature: float | None = None  # override the candidate's temperature (retry policies)

    @property
    def stop(self) -> bool:
        return self.key is None


def stop(reason: str) -> Choice:
    return Choice(None, reason)


def all_pass(verdicts: Sequence[Verdict]) -> bool:
    """Default acceptance: nothing failed (``unknown`` does not block)."""
    return not any(v.outcome == "fail" for v in verdicts)


class Policy(Protocol):
    name: str

    def choose(self, view: DecisionView) -> Choice: ...

    def accept(self, view: DecisionView, verdicts: Sequence[Verdict]) -> bool: ...


class _Base:
    name = "base"

    def accept(self, view: DecisionView, verdicts: Sequence[Verdict]) -> bool:
        return all_pass(verdicts)


class Fixed(_Base):
    """B0/B1: one model, one attempt, whatever the verifiers say."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.name = f"fixed:{model}"

    def choose(self, view: DecisionView) -> Choice:
        if view.history:
            return stop("fixed policy: single attempt")
        c = view.find_model(self.model)
        if c is None:
            c = next((x for x in view.candidates if x.kind == "model"), None)
            if c is None:
                return stop(f"no model alternative declared (wanted {self.model})")
            return Choice(c.key, f"fixed: {self.model} not declared for this step, using {c.model}")
        return Choice(c.key, f"fixed: {self.model}")


class Retry(_Base):
    """B4: retry one model with rising temperature until verified."""

    def __init__(self, model: str, temperatures: Sequence[float] = (0.0, 0.5, 0.9)) -> None:
        self.model = model
        self.temperatures = tuple(temperatures)
        self.name = f"retry:{model}:{len(self.temperatures)}"

    def choose(self, view: DecisionView) -> Choice:
        n = len(view.history)
        if n >= len(self.temperatures):
            return stop(f"retry: {n} attempts used")
        c = view.find_model(self.model) or next((x for x in view.candidates if x.kind == "model"), None)
        if c is None:
            return stop("no model alternative declared")
        if not c.affordable:
            return stop("retry: next attempt does not fit the budget")
        t = self.temperatures[n]
        return Choice(c.key, f"retry #{n + 1} at t={t:g}", temperature=t)


class Cascade(_Base):
    """B3: try alternatives in declared order (optionally restricted), escalate on failed verification."""

    def __init__(self, models: Sequence[str] | None = None, include_fn: bool = True) -> None:
        self.models = list(models) if models else None
        self.include_fn = include_fn
        self.name = "cascade" + (f":{','.join(self.models)}" if self.models else "") + ("" if include_fn else ":nofn")

    def _order(self, view: DecisionView) -> list[Candidate]:
        cands = view.candidates
        if not self.include_fn:
            cands = [c for c in cands if c.kind != "fn"]
        if self.models is None:
            return cands
        fns = [c for c in cands if c.kind == "fn"]
        models = [c for m in self.models for c in cands if c.kind == "model" and c.model == m]
        return fns + models

    def choose(self, view: DecisionView) -> Choice:
        for c in self._order(view):
            if c.tries == 0 and not c.exhausted and c.affordable:
                return Choice(c.key, f"cascade: next in order ({c.key})")
        return stop("cascade: alternatives exhausted or unaffordable")


class Cheapest(_Base):
    """B5: cheapest eligible untried alternative first, escalate on failure."""

    name = "cheapest"

    def choose(self, view: DecisionView) -> Choice:
        open_ = [c for c in view.open() if c.tries == 0]
        if not open_:
            return stop("cheapest: alternatives exhausted or unaffordable")
        c = min(open_, key=lambda x: x.est_seconds)
        return Choice(c.key, f"cheapest: est {c.est_seconds:.1f}s")


class Explore(_Base):
    """Data collection: run every alternative once, keep the first verified output.

    Gives unbiased per-alternative pass rates (a cascade only observes the big model on inputs the
    small one failed) and the per-step oracle for headroom studies. Not a production policy.
    """

    name = "explore"

    def choose(self, view: DecisionView) -> Choice:
        for c in view.candidates:
            if c.tries == 0 and c.affordable:
                return Choice(c.key, "explore: collect an observation")
        return stop("explore: every alternative observed")

    def accept(self, view: DecisionView, verdicts: Sequence[Verdict]) -> bool:
        return False  # keep going; the controller returns the first verified output at the end


class Adaptive(_Base):
    """Expected-value choice with local costs.

    For each open candidate: ``EV = p_success * value_seconds - est_seconds`` where est_seconds
    includes model load (if not resident) and the expected cost of reloading whatever it evicts.
    Run the best candidate if its EV is positive, else stop. ``value_seconds`` is the explicit
    exchange rate: how many seconds a verified output is worth to you.

    This is the standard value-of-computation rule (cf. cascade routing, Dekoninck et al. 2025);
    the contribution is pricing it with measured local costs and recording every estimate.

    Abstains (delegates to ``fallback``) while any open candidate has fewer than ``min_samples``
    observations: no adaptivity on guesses.
    """

    def __init__(self, value_seconds: float = 60.0, min_samples: int = 5, fallback: Any = None) -> None:
        self.value_seconds = value_seconds
        self.min_samples = min_samples
        self.fallback = fallback or Cascade()
        self.name = f"adaptive:v{value_seconds:g}"

    def choose(self, view: DecisionView) -> Choice:
        open_ = [c for c in view.open() if c.tries == 0]
        if view.side_effects == "external" and view.history:
            return stop("external side effects: never retried")
        if not open_:
            return stop("no open alternative")
        thin = [c.key for c in open_ if c.n < self.min_samples]
        if thin:
            inner = self.fallback.choose(view)
            return Choice(inner.key, f"abstain (n<{self.min_samples} for {', '.join(thin)}): {inner.reason}",
                          inner.temperature)
        scored = sorted(((c.p_success * self.value_seconds - c.est_seconds, c) for c in open_),
                        key=lambda t: -t[0])
        ev, best = scored[0]
        detail = "; ".join(f"{c.key} p={c.p_success:.2f} cost={c.est_seconds:.1f}s ev={e:.1f}" for e, c in scored)
        if ev <= 0:
            return stop(f"no positive value: {detail}")
        return Choice(best.key, f"max EV: {detail}")


def parse_policy(spec: str, aliases: dict[str, str] | None = None) -> Policy:
    """``fixed:small`` | ``retry:small[:3]`` | ``cascade[:small,large]`` | ``cheapest`` | ``explore`` |
    ``adaptive[:60]``."""
    aliases = aliases or {}
    head, _, rest = spec.partition(":")

    def res(m: str) -> str:
        return aliases.get(m, m)

    if head == "fixed" and rest:
        return Fixed(res(rest))
    if head == "retry" and rest:
        model, _, n = rest.rpartition(":") if rest.rsplit(":", 1)[-1].isdigit() else (rest, "", "")
        temps = (0.0, 0.5, 0.9, 1.1, 1.3)[: int(n) if n else 3]
        return Retry(res(model), temps)
    if head == "cascade":
        return Cascade([res(m) for m in rest.split(",")] if rest else None)
    if head == "cheapest":
        return Cheapest()
    if head == "explore":
        return Explore()
    if head == "adaptive":
        return Adaptive(float(rest) if rest else 60.0)
    raise ValueError(f"unknown policy spec '{spec}'")
