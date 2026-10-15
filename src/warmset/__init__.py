"""warmset: local-first adaptive execution for LLM pipelines."""
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
from .hardware import HardwareProfile
from .ledger import Ledger
from .policies import Adaptive, Cascade, Cheapest, Explore, Fixed, Retry, parse_policy
from .runtime import Deterministic, Event, Model, RunHandle, RunResult, Runtime, StepContext, step
from .skills import SkillIndex
from .types import Budget, RunStatus, StepStatus, Usage, Verdict
from .verify import Call, grounded, grounded_ids, matches, nonempty, numbers_preserved, verifier

__version__ = "0.0.1"

__all__ = [
    "Adaptive",
    "Budget",
    "BudgetExhausted",
    "Call",
    "Cascade",
    "Cheapest",
    "Deterministic",
    "Event",
    "Explore",
    "Fixed",
    "HardwareProfile",
    "InvalidOutput",
    "Ledger",
    "WarmsetError",
    "Model",
    "ModelError",
    "ModelUnavailable",
    "NeedsInput",
    "NoRunContext",
    "NotApplicable",
    "Retry",
    "RunHandle",
    "RunResult",
    "RunStatus",
    "Runtime",
    "SkillIndex",
    "StepContext",
    "StepFailed",
    "StepStatus",
    "Usage",
    "Verdict",
    "grounded",
    "grounded_ids",
    "matches",
    "nonempty",
    "numbers_preserved",
    "parse_policy",
    "step",
    "verifier",
]
