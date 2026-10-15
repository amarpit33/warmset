"""Success and latency estimates, cost model and budget tracking."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

from .hardware import HardwareProfile, Residency
from .types import Budget, Usage

Key = tuple[str, str]  # (step name, action key)
BucketKey = tuple[str, str, "str | None"]  # (step, action key, feature bucket); None = step-wide


class SuccessEstimator:
    """Beta-Binomial pass rate per (step, action), optionally per feature bucket. Prior Beta(a, b).

    Every observation updates the step-wide count and, if the input had a bucket, the bucket count.
    ``estimate`` uses the bucket once it has ``min_bucket`` observations, else the step-wide rate.
    """

    def __init__(self, prior: tuple[float, float] = (1.0, 1.0), min_bucket: int = 3) -> None:
        self.prior = prior
        self.min_bucket = min_bucket
        self.counts: dict[BucketKey, list[int]] = defaultdict(lambda: [0, 0])

    def update(self, step: str, key: str, passed: bool, bucket: str | None = None) -> None:
        self.counts[(step, key, None)][0 if passed else 1] += 1
        if bucket is not None:
            self.counts[(step, key, bucket)][0 if passed else 1] += 1

    def n(self, step: str, key: str, bucket: str | None = None) -> int:
        s, f = self.counts.get((step, key, bucket), (0, 0))
        return s + f

    def mean(self, step: str, key: str, bucket: str | None = None) -> float:
        s, f = self.counts.get((step, key, bucket), (0, 0))
        a, b = self.prior
        return (s + a) / (s + f + a + b)

    def estimate(self, step: str, key: str, bucket: str | None = None) -> tuple[float, int, str]:
        """(pass probability, observations behind it, "bucket:<b>" or "step")."""
        if bucket is not None and self.n(step, key, bucket) >= self.min_bucket:
            return self.mean(step, key, bucket), self.n(step, key, bucket), f"bucket:{bucket}"
        return self.mean(step, key), self.n(step, key), "step"

    def lcb(self, step: str, key: str, z: float = 1.0, bucket: str | None = None) -> float:
        """Mean minus ``z`` posterior standard deviations (conservative acceptance)."""
        s, f = self.counts.get((step, key, bucket), (0, 0))
        a, b = s + self.prior[0], f + self.prior[1]
        var = a * b / ((a + b) ** 2 * (a + b + 1))
        return max(0.0, a / (a + b) - z * math.sqrt(var))

    @classmethod
    def from_rows(cls, rows: Iterable[tuple]) -> SuccessEstimator:
        """Rows of (step, key, passed) or (step, key, passed, bucket)."""
        est = cls()
        for row in rows:
            est.update(*row)
        return est


class LatencyEstimator:
    """Median execution seconds (model load excluded) per (step, action), last ``window`` samples."""

    def __init__(self, window: int = 50) -> None:
        self.samples: dict[Key, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def update(self, step: str, key: str, exec_seconds: float) -> None:
        self.samples[(step, key)].append(exec_seconds)

    def median(self, step: str, key: str) -> float | None:
        s = self.samples.get((step, key))
        return statistics.median(s) if s else None

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[str, str, float]]) -> LatencyEstimator:
        est = cls()
        for step, key, secs in rows:
            est.update(step, key, secs)
        return est


@dataclass(frozen=True)
class CostEstimate:
    seconds: float
    load_seconds: float
    evict_seconds: float
    exec_seconds: float
    source: str  # where exec_seconds came from: "observed" | "profile" | "default"


class CostModel:
    """``cost = load(m) if not resident + revisit_probability * reload(evicted) + exec``."""

    def __init__(self, profile: HardwareProfile, latency: LatencyEstimator, *, revisit_probability: float = 0.5,
                 default_prompt_tokens: int = 512, default_completion_tokens: int = 128,
                 default_seconds: float = 5.0) -> None:
        self.profile = profile
        self.latency = latency
        self.revisit_probability = revisit_probability
        self.default_tokens = (default_prompt_tokens, default_completion_tokens)
        self.default_seconds = default_seconds

    def estimate(self, step: str, key: str, model: str | None, residency: Residency | None, *,
                 amortize: int = 1) -> CostEstimate:
        """``amortize``: number of runs expected to share one load of the model (batch mode)."""
        observed = self.latency.median(step, key)
        if model is None:  # deterministic function
            ex = observed if observed is not None else 0.001
            return CostEstimate(ex, 0.0, 0.0, ex, "observed" if observed is not None else "default")
        prof = self.profile.model(model)
        if observed is not None:
            ex, src = observed, "observed"
        elif prof and (prof.prefill_tps or prof.decode_tps):
            ex, src = prof.exec_seconds(*self.default_tokens), "profile"
        else:
            ex, src = self.default_seconds, "default"
        load = evict = 0.0
        if residency is not None and residency.is_resident(model) is False:
            load = prof.load_seconds if prof else 0.0
            evict = self.revisit_probability * sum(
                (p.load_seconds if (p := self.profile.model(e)) else 0.0) for e in residency.would_evict(model))
        share = max(1, amortize)
        load, evict = load / share, evict / share
        return CostEstimate(load + evict + ex, load, evict, ex, src)


class BudgetTracker:
    def __init__(self, budget: Budget | None) -> None:
        self.budget = budget or Budget()
        self.seconds = 0.0
        self.calls = 0
        self.tokens = 0
        self.loads = 0
        self.load_seconds = 0.0

    def charge(self, usage: Usage) -> None:
        self.seconds += usage.seconds
        self.calls += usage.calls
        self.tokens += usage.tokens
        self.load_seconds += usage.load_seconds
        if usage.load_seconds >= 0.5:  # sub-second "loads" are cache touches, not real loads
            self.loads += 1

    @property
    def remaining_seconds(self) -> float | None:
        return None if self.budget.seconds is None else self.budget.seconds - self.seconds

    @property
    def remaining_calls(self) -> int | None:
        return None if self.budget.model_calls is None else self.budget.model_calls - self.calls

    def affordable(self, est_seconds: float, is_model: bool) -> bool:
        if self.budget.seconds is not None and est_seconds > self.budget.seconds - self.seconds:
            return False
        if is_model and self.budget.model_calls is not None and self.calls >= self.budget.model_calls:
            return False
        if is_model and self.budget.tokens is not None and self.tokens >= self.budget.tokens:
            return False
        return True

    @property
    def exhausted(self) -> bool:
        b = self.budget
        return ((b.seconds is not None and self.seconds >= b.seconds)
                or (b.model_calls is not None and self.calls >= b.model_calls)
                or (b.tokens is not None and self.tokens >= b.tokens))
