"""Verifiers: cheap, deterministic checks that decide whether an output can be accepted.

A verifier is any callable ``(output, call) -> Verdict | bool | str | None`` where ``call`` holds
the step's arguments. ``True``/``None`` mean pass, ``False`` fail, a string is a failure reason.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .types import Verdict


@dataclass(frozen=True)
class Call:
    """The arguments a step was invoked with."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def text(self) -> str:
        """All string-like arguments joined, for grounding checks."""
        parts = [a for a in (*self.args, *self.kwargs.values()) if isinstance(a, str)]
        return "\n".join(parts)


VerifierFn = Callable[[Any, Call], Any]


def verifier(name: str | None = None) -> Callable[[VerifierFn], VerifierFn]:
    """Name a verifier function for the ledger."""
    def deco(fn: VerifierFn) -> VerifierFn:
        fn.verifier_name = name or fn.__name__  # type: ignore[attr-defined]
        return fn
    return deco


def name_of(v: VerifierFn) -> str:
    return getattr(v, "verifier_name", getattr(v, "__name__", "verifier"))


def run_verifier(v: VerifierFn, output: Any, call: Call) -> Verdict:
    name = name_of(v)
    try:
        res = v(output, call)
    except Exception as exc:  # a crashing verifier is a failed check, never an accepted output
        return Verdict(name, "fail", f"verifier raised {type(exc).__name__}: {exc}")
    if isinstance(res, Verdict):
        return res
    if res is None or res is True:
        return Verdict(name, "pass")
    if res is False:
        return Verdict(name, "fail")
    return Verdict(name, "fail", str(res))


def _get(output: Any, field: str | None) -> Any:
    if field is None:
        return output
    cur = output
    for part in field.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return cur


# ---------------------------------------------------------------- built-ins

def nonempty(field: str | None = None) -> VerifierFn:
    @verifier(f"nonempty({field or ''})")
    def check(output: Any, call: Call) -> Any:
        value = _get(output, field)
        return bool(value.strip()) if isinstance(value, str) else value not in (None, [], {}, "")
    return check


def grounded(field: str, source: Callable[[Call], str] | None = None, *, case: bool = False) -> VerifierFn:
    """The field's text (or every item, for lists) must appear verbatim in the source text."""
    @verifier(f"grounded({field})")
    def check(output: Any, call: Call) -> Any:
        value = _get(output, field)
        hay = source(call) if source else call.text()
        if not case:
            hay = hay.lower()
        items = value if isinstance(value, list) else [value]
        missing = [str(i) for i in items if str(i).strip() and (str(i) if case else str(i).lower()) not in hay]
        return f"not in source: {', '.join(missing[:5])}" if missing else True
    return check


def grounded_ids(field: str, allowed: Callable[[Call], Iterable[str]]) -> VerifierFn:
    """Every id the output cites must exist (drops hallucinated citations)."""
    @verifier(f"grounded_ids({field})")
    def check(output: Any, call: Call) -> Any:
        ids = _get(output, field) or []
        known = {str(a).upper() for a in allowed(call)}
        bad = [i for i in ids if str(i).upper() not in known]
        return f"unknown ids: {', '.join(map(str, bad[:5]))}" if bad else True
    return check


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[float]:
    """Numeric values in ``text`` ('1,234.50' and '1234.5' are the same number)."""
    out = set()
    for n in _NUMBER.findall(text):
        try:
            out.add(float(n.replace(",", "")))
        except ValueError:
            continue
    return out


def numbers_preserved(field: str | None = None, source: Callable[[Call], str] | None = None) -> VerifierFn:
    """Every number in the output must appear in the source (no invented figures)."""
    @verifier(f"numbers_preserved({field or ''})")
    def check(output: Any, call: Call) -> Any:
        value = _get(output, field)
        text = value.model_dump_json() if hasattr(value, "model_dump_json") else str(value)
        invented = _numbers(text) - _numbers(source(call) if source else call.text())
        return f"numbers not in source: {', '.join(f'{x:g}' for x in sorted(invented)[:5])}" if invented else True
    return check


def matches(pattern: str, field: str | None = None) -> VerifierFn:
    rx = re.compile(pattern)

    @verifier(f"matches({field or ''})")
    def check(output: Any, call: Call) -> Any:
        value = _get(output, field)
        return bool(rx.fullmatch(str(value))) or f"{value!r} does not match {pattern}"
    return check
