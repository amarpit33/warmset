"""Exception hierarchy. Everything warmset raises derives from ``WarmsetError``."""
from __future__ import annotations


class WarmsetError(Exception):
    """Base class."""


class ModelError(WarmsetError):
    """The model backend failed or returned something unusable."""


class ModelUnavailable(ModelError):
    """The model backend could not be reached."""


class InvalidOutput(ModelError):
    """The model answered, but the answer did not match the requested schema."""


class NotApplicable(WarmsetError):
    """A deterministic alternative cannot handle this input; the controller moves on without penalty."""


class BudgetExhausted(WarmsetError):
    """The run's budget is spent."""


class StepFailed(WarmsetError):
    """No alternative produced any output for a step."""

    def __init__(self, step: str, reason: str) -> None:
        super().__init__(f"step '{step}' failed: {reason}")
        self.step = step
        self.reason = reason


class NeedsInput(WarmsetError):
    """The input is ambiguous or incomplete; a human (or caller) must supply more."""


class NoRunContext(WarmsetError):
    """A step was called outside ``Runtime.run``."""
