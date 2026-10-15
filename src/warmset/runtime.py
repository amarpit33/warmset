"""Runtime, steps and the controller loop.

A pipeline is a plain async function. Inside it, calls to ``@step`` functions are intercepted:
the controller builds the step's eligible alternatives, prices them, asks the policy which to run,
executes, verifies, records everything, and either accepts the output or escalates.

    @step(alternatives=[Deterministic(regex_parse), Model("small"), Model("large")],
          verify=[grounded("vendor")])
    async def extract(ctx: StepContext, text: str) -> Invoice:
        return await ctx.generate(system=SYSTEM, user=text, schema=Invoice)

    result = await runtime.run(pipeline, text, policy="adaptive", budget=Budget(seconds=120))
"""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .errors import (
    BudgetExhausted,
    InvalidOutput,
    ModelError,
    ModelUnavailable,
    NeedsInput,
    NoRunContext,
    NotApplicable,
    StepFailed,
    WarmsetError,
)
from .estimators import BudgetTracker, CostModel, LatencyEstimator, SuccessEstimator
from .hardware import HardwareProfile, Residency
from .ledger import Ledger
from .policies import Attempt, Candidate, Choice, DecisionView, Policy, parse_policy
from .providers.base import Message, Provider
from .scheduler import ModelScheduler
from .skills import SkillIndex
from .structured import generate_structured, generate_text
from .types import Budget, RunStatus, StepStatus, Usage, Verdict
from .verify import Call, VerifierFn, run_verifier

T = TypeVar("T", bound=BaseModel)

_current_run: contextvars.ContextVar[_RunState | None] = contextvars.ContextVar("warmset_run", default=None)


# ---------------------------------------------------------------- alternatives

@dataclass(frozen=True)
class Model:
    """Run the step body with this model (name or alias)."""

    model: str
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass(frozen=True)
class Deterministic:
    """Run ``fn(*args, **kwargs)`` instead of a model. Raise ``NotApplicable`` to pass the step on."""

    fn: Callable[..., Any]
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or getattr(self.fn, "__name__", "fn")


Alternative = Model | Deterministic


@dataclass
class StepDef:
    fn: Callable[..., Awaitable[Any]]
    name: str
    alternatives: tuple[Alternative, ...]
    verifiers: tuple[VerifierFn, ...]
    side_effects: str
    features: Callable[..., str] | None = None

    def bucket(self, call: Call) -> str | None:
        """The input's feature bucket, if the step declares a featurizer (never fails the step)."""
        if self.features is None:
            return None
        try:
            return str(self.features(*call.args, **call.kwargs))
        except Exception:
            return None

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        run = _current_run.get()
        if run is None:
            raise NoRunContext(f"step '{self.name}' called outside Runtime.run()")
        return await run.execute_step(self, args, kwargs)


def step(*, alternatives: Sequence[Alternative], verify: Sequence[VerifierFn] = (), name: str | None = None,
         side_effects: str = "none", features: Callable[..., str] | None = None,
         ) -> Callable[[Callable[..., Awaitable[Any]]], StepDef]:
    """Declare a step and the alternatives the controller may choose between.

    ``side_effects``: ``none`` | ``idempotent`` | ``external``. External steps run at most one action
    and are never retried.

    ``features``: optional cheap, deterministic function of the step's arguments returning a bucket
    label (e.g. "itemized", "long"). Pass rates are then estimated per bucket, backing off to the
    step-wide rate while a bucket has too few observations.
    """
    if side_effects not in ("none", "idempotent", "external"):
        raise ValueError("side_effects must be 'none', 'idempotent' or 'external'")
    if not alternatives:
        raise ValueError("a step needs at least one alternative")

    def deco(fn: Callable[..., Awaitable[Any]]) -> StepDef:
        return StepDef(fn, name or fn.__name__, tuple(alternatives), tuple(verify), side_effects, features)

    return deco


# ---------------------------------------------------------------- step context

class StepContext:
    """What a step body sees when the controller runs it with a model."""

    def __init__(self, runtime: Runtime, model: str, temperature: float, max_tokens: int | None, call: Call) -> None:
        self.runtime = runtime
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens or runtime.default_max_tokens
        self.call = call
        self.usage = Usage()
        self.context: dict[str, int] = {}  # segment -> approx tokens, for attribution

    def _note(self, segment: str, text: str) -> None:
        self.context[segment] = self.context.get(segment, 0) + max(1, len(text) // 4)

    def skill(self, name: str) -> str:
        """Load a skill's instructions (tier 2) and attribute their tokens."""
        if self.runtime.skills is None:
            raise KeyError("runtime has no skills configured")
        body = self.runtime.skills[name].body()
        self._note(f"skill:{name}", body)
        return body

    async def generate(self, *, user: str, system: str | None = None, schema: type[T] | None = None,
                       max_tokens: int | None = None) -> Any:
        """Schema-constrained output if ``schema`` is given, else text."""
        messages = ([Message("system", system)] if system else []) + [Message("user", user)]
        if system:
            self._note("system", system)
        self._note("user", user)
        return await self.chat(messages, schema=schema, max_tokens=max_tokens)

    async def chat(self, messages: list[Message], *, schema: type[T] | None = None,
                   max_tokens: int | None = None) -> Any:
        mt = max_tokens or self.max_tokens
        timeout = self.runtime.action_timeout
        async with self.runtime.scheduler.slot(self.model):  # queue wait is not part of the timeout
            start = time.perf_counter()
            try:
                if schema is not None:
                    obj, usage = await asyncio.wait_for(
                        generate_structured(self.runtime.provider, messages, schema, self.model,
                                            temperature=self.temperature, max_tokens=mt), timeout)
                    self.usage = self.usage + usage
                    return obj
                result, usage = await asyncio.wait_for(
                    generate_text(self.runtime.provider, messages, self.model, temperature=self.temperature,
                                  max_tokens=mt), timeout)
                self.usage = self.usage + usage
                if not result.text.strip():
                    raise InvalidOutput("model returned an empty answer")
                return result.text
            except InvalidOutput as exc:
                self.usage = self.usage + getattr(exc, "usage", Usage())
                raise
            except asyncio.TimeoutError:
                self.usage = self.usage + Usage(seconds=time.perf_counter() - start, calls=1)
                raise ModelUnavailable(f"model call timed out after {timeout:.0f}s") from None


# ---------------------------------------------------------------- events and results

@dataclass(frozen=True)
class Event:
    type: str
    run_id: str
    data: dict[str, Any]
    ts: float = field(default_factory=time.time)


@dataclass
class RunResult:
    id: str
    status: RunStatus
    result: Any = None
    stop_reason: str | None = None
    error: str | None = None
    seconds: float = 0.0          # action seconds (provider-reported, load included)
    wall_seconds: float = 0.0
    calls: int = 0
    tokens: int = 0
    loads: int = 0
    load_seconds: float = 0.0
    unverified_steps: int = 0

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.COMPLETED


class RunHandle:
    """A started run: stream its events, cancel it, or await its result."""

    def __init__(self, run_id: str, task: asyncio.Task[RunResult], queue: asyncio.Queue[Event | None]) -> None:
        self.id = run_id
        self._task = task
        self._queue = queue

    async def result(self) -> RunResult:
        return await asyncio.shield(self._task)

    def cancel(self) -> None:
        self._task.cancel()

    async def events(self) -> AsyncIterator[Event]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev


# ---------------------------------------------------------------- per-run state and controller

@dataclass
class _Batch:
    """Runs started together by ``Runtime.run_batch``; ``active`` counts those not yet finished."""

    id: str
    active: int


class _RunState:
    def __init__(self, rt: Runtime, run_id: str, policy: Policy, budget: Budget | None,
                 queue: asyncio.Queue[Event | None], batch: _Batch | None = None) -> None:
        self.rt = rt
        self.id = run_id
        self.policy = policy
        self.budget = BudgetTracker(budget)
        self.queue = queue
        self.seq = 0
        self.unverified = 0
        self.batch = batch

    def emit(self, type_: str, **data: Any) -> None:
        self.rt.ledger.record_event(self.id, type_, data)
        self.queue.put_nowait(Event(type_, self.id, data))

    def _key(self, alt: Alternative) -> tuple[str, str, str | None, float]:
        if isinstance(alt, Deterministic):
            return f"fn:{alt.label}", "fn", None, 0.0
        model = self.rt.resolve(alt.model)
        return f"model:{model}@t{alt.temperature:g}", "model", model, alt.temperature

    def _candidates(self, sd: StepDef, history: list[Attempt], bucket: str | None) -> list[Candidate]:
        out = []
        for alt in sd.alternatives:
            key, kind, model, temp = self._key(alt)
            cost = self.rt.cost.estimate(sd.name, key, model, self.rt.residency,
                                         amortize=self.batch.active if self.batch else 1)
            tries = [a for a in history if a.key == key]
            exhausted = bool(tries) and (kind == "fn" or all(a.temperature == 0 for a in tries))
            p, n, p_source = self.rt.success.estimate(sd.name, key, bucket)
            out.append(Candidate(
                key=key, kind=kind, model=model, temperature=temp, est_seconds=round(cost.seconds, 3),
                est_load_seconds=round(cost.load_seconds, 3), est_evict_seconds=round(cost.evict_seconds, 3),
                p_success=round(p, 4), n=n, p_source=p_source,
                resident=self.rt.residency.is_resident(model) if model else None, tries=len(tries),
                exhausted=exhausted, affordable=self.budget.affordable(cost.seconds, kind == "model")))
        return out

    async def _execute(self, sd: StepDef, alt: Alternative, call: Call, model: str | None,
                       temperature: float) -> tuple[Any, Usage, str | None, dict[str, int] | None, Exception | None]:
        """Returns (output or None, usage, error or None, context attribution, fatal exception or None).

        Model and backend failures are ordinary failed actions (the policy may escalate). Any other
        exception from a step body is a bug: it is recorded, then re-raised, never escalated around.
        """
        start = time.perf_counter()
        if isinstance(alt, Deterministic):
            try:
                out = alt.fn(*call.args, **call.kwargs)
                if inspect.isawaitable(out):
                    out = await out
                return out, Usage(seconds=time.perf_counter() - start), None, None, None
            except NotApplicable as exc:
                return None, Usage(seconds=time.perf_counter() - start), f"not applicable: {exc}", None, None
            except Exception as exc:  # a broken deterministic path is a failed action, not a dead run
                return None, Usage(seconds=time.perf_counter() - start), f"{type(exc).__name__}: {exc}", None, None
        assert isinstance(alt, Model) and model is not None
        ctx = StepContext(self.rt, model, temperature, alt.max_tokens, call)
        try:
            body = sd.fn(ctx, *call.args, **call.kwargs)
            # In a batch the body may queue for a model; per-call timeouts apply inside the scheduler instead.
            out = await (body if self.batch else asyncio.wait_for(body, self.rt.action_timeout))
            return out, ctx.usage, None, ctx.context, None
        except asyncio.TimeoutError:
            return None, ctx.usage + Usage(seconds=time.perf_counter() - start), \
                f"timeout after {self.rt.action_timeout:.0f}s", ctx.context, None
        except ModelError as exc:
            usage = ctx.usage if ctx.usage.calls else Usage(seconds=time.perf_counter() - start)
            return None, usage, f"{type(exc).__name__}: {exc}", ctx.context, None
        except (WarmsetError, asyncio.CancelledError):
            raise  # NeedsInput, BudgetExhausted, cancellation: run-level outcomes
        except Exception as exc:
            return None, ctx.usage, f"{type(exc).__name__}: {exc}", ctx.context, exc

    async def execute_step(self, sd: StepDef, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.seq += 1
        call = Call(args, kwargs)
        bucket = sd.bucket(call)
        step_id = self.rt.ledger.start_step(self.id, self.seq, sd.name, {"args": list(args), "kwargs": kwargs})
        self.emit("step_started", step=sd.name, seq=self.seq)
        history: list[Attempt] = []
        cands: list[Candidate] = []
        first_passed: tuple[str, Any] | None = None
        last_output: tuple[str, Any] | None = None
        alts = {self._key(a)[0]: a for a in sd.alternatives}

        while True:
            if self.budget.exhausted:
                stop_reason = "budget exhausted"
                break
            await self.rt.residency.refresh()
            cands = self._candidates(sd, history, bucket)
            view = DecisionView(sd.name, len(history) + 1, cands, list(history), self.budget.remaining_seconds,
                                self.budget.remaining_calls, sd.side_effects, bucket)
            choice: Choice = self.policy.choose(view)
            if choice.key is not None and sd.side_effects == "external" and history:
                choice = Choice(None, f"external side effects: not retried (policy wanted {choice.key})")
            if choice.key is not None and choice.key not in alts:
                choice = Choice(None, f"policy chose unknown alternative {choice.key}")
            decision_id = self.rt.ledger.record_decision(self.id, step_id, view.attempt, choice.key,
                                                         choice.temperature, choice.reason, view.to_dict())
            self.emit("decision", step=sd.name, attempt=view.attempt, chosen=choice.key, reason=choice.reason)
            if choice.key is None:
                stop_reason = choice.reason
                break

            cand = next(c for c in cands if c.key == choice.key)
            temperature = choice.temperature if choice.temperature is not None else cand.temperature
            output, usage, error, context, fatal = await self._execute(sd, alts[choice.key], call, cand.model,
                                                                       temperature)
            self.budget.charge(usage)
            ok = error is None
            verdicts = [run_verifier(v, output, call) for v in sd.verifiers] if ok else \
                [Verdict("execution", "fail", error or "")]
            passed = ok and self.policy.accept(view, verdicts)
            self.rt.ledger.record_action(
                run_id=self.id, step_id=step_id, decision_id=decision_id, step_name=sd.name, key=choice.key,
                kind=cand.kind, model=cand.model, temperature=temperature, ok=ok, passed=ok and all(
                    v.outcome != "fail" for v in verdicts), error=error, usage=usage, predicted_p=cand.p_success,
                predicted_seconds=cand.est_seconds, output=output, context=context, verdicts=verdicts, bucket=bucket)
            if fatal is not None:
                self.rt.ledger.finish_step(step_id, StepStatus.FAILED.value, None, None)
                self.emit("step_finished", step=sd.name, status="failed", reason=f"bug in step body: {error}")
                raise fatal
            is_not_applicable = error is not None and error.startswith("not applicable")
            if self.rt.learn and not is_not_applicable:
                self.rt.success.update(sd.name, choice.key, ok and all(v.outcome != "fail" for v in verdicts), bucket)
                if ok:
                    self.rt.latency.update(sd.name, choice.key, usage.exec_seconds)
            self.emit("action_finished", step=sd.name, key=choice.key, ok=ok, seconds=round(usage.seconds, 3),
                      load_seconds=round(usage.load_seconds, 3),
                      verdicts=[f"{v.verifier}:{v.outcome}" for v in verdicts])
            history.append(Attempt(choice.key, temperature, ok, ok and all(v.outcome != "fail" for v in verdicts)))
            if ok:
                last_output = (choice.key, output)
                if first_passed is None and history[-1].passed:
                    first_passed = (choice.key, output)
            if passed:
                stop_reason = "accepted"
                break

        if first_passed is not None:
            self.rt.ledger.finish_step(step_id, StepStatus.ACCEPTED.value, first_passed[0], first_passed[1])
            self.emit("step_finished", step=sd.name, status="accepted", key=first_passed[0])
            return first_passed[1]
        if last_output is not None and self.rt.allow_unverified:
            self.unverified += 1
            self.rt.ledger.finish_step(step_id, StepStatus.UNVERIFIED.value, last_output[0], last_output[1])
            self.emit("step_finished", step=sd.name, status="unverified", key=last_output[0], reason=stop_reason)
            return last_output[1]
        self.rt.ledger.finish_step(step_id, StepStatus.FAILED.value, None, None)
        self.emit("step_finished", step=sd.name, status="failed", reason=stop_reason)
        priced_out = any(c.tries == 0 and not c.affordable for c in cands)
        if self.budget.exhausted or priced_out:
            raise BudgetExhausted(f"step '{sd.name}': {stop_reason}")
        raise StepFailed(sd.name, stop_reason)


# ---------------------------------------------------------------- runtime

class Runtime:
    """Owns the provider, ledger, estimators and hardware view. One per process is typical."""

    def __init__(self, provider: Provider, *, models: dict[str, str] | None = None,
                 ledger: Ledger | str | Path = ":memory:", profile: HardwareProfile | None = None,
                 success: SuccessEstimator | None = None, latency: LatencyEstimator | None = None,
                 skills: SkillIndex | None = None, max_concurrency: int = 1, action_timeout: float = 300.0,
                 default_max_tokens: int = 512, learn: bool = True, allow_unverified: bool = True,
                 revisit_probability: float = 0.5) -> None:
        self.provider = provider
        self.aliases = dict(models or {})
        self.ledger = ledger if isinstance(ledger, Ledger) else Ledger(ledger)
        self.profile = profile or HardwareProfile.detect()
        self.success = success or SuccessEstimator()
        self.latency = latency or LatencyEstimator()
        self.cost = CostModel(self.profile, self.latency, revisit_probability=revisit_probability)
        self.residency = Residency(provider, self.profile)
        self.skills = skills
        self.action_timeout = action_timeout
        self.default_max_tokens = default_max_tokens
        self.learn = learn
        self.allow_unverified = allow_unverified
        # One model call at a time by default: on local hardware parallel calls to different models thrash memory.
        self.max_concurrency = max_concurrency
        self.scheduler = ModelScheduler(max_concurrency)
        self.last_batch_switches = 0

    def resolve(self, model: str) -> str:
        return self.aliases.get(model, model)

    def fit(self, *, names: Sequence[str] | None = None) -> int:
        """(Re)fit estimators from the ledger; returns the number of actions used."""
        rows = [r for r in self.ledger.outcome_rows() if names is None or r["run_name"] in names]
        rows = [r for r in rows if not (r["error"] or "").startswith("not applicable")]
        self.success = SuccessEstimator.from_rows(
            (r["step_name"], r["action_key"], bool(r["passed"]), r["bucket"]) for r in rows)
        self.latency = LatencyEstimator.from_rows(
            (r["step_name"], r["action_key"], r["exec_seconds"] or 0.0) for r in rows if r["ok"])
        self.cost.latency = self.latency
        return len(rows)

    def start(self, pipeline: Callable[..., Awaitable[Any]], *args: Any, policy: Policy | str = "cascade",
              budget: Budget | None = None, name: str | None = None, meta: dict[str, Any] | None = None,
              _batch: _Batch | None = None, **kwargs: Any) -> RunHandle:
        pol = parse_policy(policy, self.aliases) if isinstance(policy, str) else policy
        if _batch is not None:
            meta = {**(meta or {}), "batch": _batch.id}
        run_id = self.ledger.create_run(policy=pol.name, name=name or getattr(pipeline, "__name__", None),
                                        budget=budget, inputs={"args": list(args), "kwargs": kwargs}, meta=meta)
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        state = _RunState(self, run_id, pol, budget, queue, _batch)
        task = asyncio.get_running_loop().create_task(self._drive(state, pipeline, args, kwargs))
        return RunHandle(run_id, task, queue)

    async def run(self, pipeline: Callable[..., Awaitable[Any]], *args: Any, policy: Policy | str = "cascade",
                  budget: Budget | None = None, name: str | None = None, meta: dict[str, Any] | None = None,
                  **kwargs: Any) -> RunResult:
        return await self.start(pipeline, *args, policy=policy, budget=budget, name=name, meta=meta,
                                **kwargs).result()

    async def run_batch(self, pipeline: Callable[..., Awaitable[Any]], inputs: Sequence[dict[str, Any]], *,
                        policy: Policy | str = "cascade", budget: Budget | None = None, name: str | None = None,
                        metas: Sequence[dict[str, Any]] | None = None, schedule: str = "residency",
                        ) -> list[RunResult]:
        """Run ``pipeline(**inputs[i])`` for every input concurrently and return results in input order.

        ``schedule="residency"``: model calls are served for the loaded model while anyone waits for it,
        so each model is loaded about once per batch instead of once per escalation. Policies see load and
        eviction costs amortised over the runs still in flight (an approximation: it assumes the other runs
        are likely to want the same model). ``schedule="fifo"``: same concurrency, arrival order.
        ``budget`` applies to each run.
        """
        if schedule not in ("residency", "fifo"):
            raise ValueError("schedule must be 'residency' or 'fifo'")
        await self.residency.refresh()
        loaded = next(iter(self.residency.models()), None)
        previous = self.scheduler
        self.scheduler = ModelScheduler(self.max_concurrency, residency_first=schedule == "residency",
                                        current=loaded)
        batch = _Batch(uuid.uuid4().hex[:8], len(inputs))
        try:
            handles = [self.start(pipeline, policy=policy, budget=budget, name=name,
                                  meta=(metas[i] if metas else None), _batch=batch, **inp)
                       for i, inp in enumerate(inputs)]
            return [await h.result() for h in handles]
        finally:
            self.last_batch_switches = self.scheduler.switches
            self.scheduler = previous

    async def _drive(self, state: _RunState, pipeline: Callable[..., Awaitable[Any]], args: tuple[Any, ...],
                     kwargs: dict[str, Any]) -> RunResult:
        token = _current_run.set(state)
        t0 = time.perf_counter()
        res = RunResult(state.id, RunStatus.RUNNING)
        state.emit("run_started", policy=state.policy.name)
        try:
            res.result = await pipeline(*args, **kwargs)
            res.status = RunStatus.COMPLETED
        except BudgetExhausted as exc:
            res.status, res.stop_reason = RunStatus.STOPPED_BUDGET, str(exc)
        except NeedsInput as exc:
            res.status, res.stop_reason = RunStatus.NEEDS_INPUT, str(exc)
        except asyncio.CancelledError:
            res.status, res.stop_reason = RunStatus.CANCELLED, "cancelled"
        except Exception as exc:  # the run fails; the ledger keeps the evidence
            res.status, res.error = RunStatus.FAILED, f"{type(exc).__name__}: {exc}"
            res.stop_reason = str(exc) if isinstance(exc, StepFailed) else None
        finally:
            _current_run.reset(token)
            if state.batch is not None:
                state.batch.active -= 1
        b = state.budget
        res.seconds, res.calls, res.tokens, res.loads, res.load_seconds = (b.seconds, b.calls, b.tokens, b.loads,
                                                                           b.load_seconds)
        res.wall_seconds, res.unverified_steps = time.perf_counter() - t0, state.unverified
        self.ledger.finish_run(state.id, status=res.status.value, stop_reason=res.stop_reason or res.error,
                               result=res.result, seconds=res.seconds, calls=res.calls, tokens=res.tokens,
                               loads=res.loads, load_seconds=res.load_seconds, wall_seconds=res.wall_seconds,
                               unverified_steps=res.unverified_steps)
        state.emit("run_finished", status=res.status.value, stop_reason=res.stop_reason)
        state.queue.put_nowait(None)
        return res
