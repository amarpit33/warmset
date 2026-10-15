"""Plain data types shared across modules."""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal


class RunStatus(str, enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED_BUDGET = "stopped_budget"
    NEEDS_INPUT = "needs_input"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, enum.Enum):
    ACCEPTED = "accepted"        # an output passed every verifier
    UNVERIFIED = "unverified"    # an output exists but failed (or skipped) verification
    FAILED = "failed"            # no output at all


@dataclass(frozen=True)
class Usage:
    """Resource accounting for one action (possibly several backend calls)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0               # wall time, including any model load
    load_seconds: float = 0.0          # time the backend spent loading the model into memory
    prompt_eval_seconds: float | None = None
    eval_seconds: float | None = None  # pure generation time, when the backend reports it
    calls: int = 0                     # backend calls made

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def exec_seconds(self) -> float:
        """Wall time minus model load: what the action costs once the model is resident."""
        return max(0.0, self.seconds - self.load_seconds)

    def __add__(self, other: Usage) -> Usage:
        def opt(a: float | None, b: float | None) -> float | None:
            return None if a is None and b is None else (a or 0.0) + (b or 0.0)

        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens,
                     self.seconds + other.seconds, self.load_seconds + other.load_seconds,
                     opt(self.prompt_eval_seconds, other.prompt_eval_seconds),
                     opt(self.eval_seconds, other.eval_seconds), self.calls + other.calls)


Outcome = Literal["pass", "fail", "unknown"]


@dataclass(frozen=True)
class Verdict:
    """Result of one verifier on one output."""

    verifier: str
    outcome: Outcome
    evidence: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"


@dataclass(frozen=True)
class Budget:
    """Limits for one run. ``None`` means unlimited.

    ``seconds`` counts action time as reported by the provider (model load included), not
    controller overhead, so simulated providers can be budgeted exactly.
    """

    seconds: float | None = None
    model_calls: int | None = None
    tokens: int | None = None
