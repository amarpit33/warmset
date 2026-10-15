"""Measured hardware profile and live model residency.

On a memory-constrained machine the expensive part of escalating to a bigger model is often
loading it (and evicting what the rest of the pipeline needs), not generating. This module
measures those costs once per machine and tracks which models are resident.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .providers.base import GenerationParams, Message, Provider

DEFAULT_PROFILE_PATH = Path.home() / ".warmset" / "profile.json"
_PROBE = ("Summarise in two sentences why caching helps web applications. " * 12).strip()


def detect_memory_bytes() -> int | None:
    """Total physical memory, or None if it cannot be read."""
    try:
        if platform.system() == "Darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                      check=True, timeout=5).stdout.strip())
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


@dataclass
class ModelProfile:
    size_bytes: int = 0
    load_seconds: float = 0.0
    prefill_tps: float = 0.0   # prompt tokens per second
    decode_tps: float = 0.0    # generated tokens per second
    measured_at: str = ""

    def exec_seconds(self, prompt_tokens: int, completion_tokens: int) -> float:
        pre = prompt_tokens / self.prefill_tps if self.prefill_tps else 0.0
        dec = completion_tokens / self.decode_tps if self.decode_tps else 0.0
        return pre + dec


@dataclass
class HardwareProfile:
    memory_bytes: int | None = None
    memory_budget_bytes: int | None = None  # how much the model server may hold at once
    models: dict[str, ModelProfile] = field(default_factory=dict)

    @classmethod
    def detect(cls, budget_fraction: float = 0.6) -> HardwareProfile:
        mem = detect_memory_bytes()
        return cls(mem, int(mem * budget_fraction) if mem else None)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROFILE_PATH) -> HardwareProfile:
        data = json.loads(Path(path).read_text())
        return cls(data.get("memory_bytes"), data.get("memory_budget_bytes"),
                   {k: ModelProfile(**v) for k, v in data.get("models", {}).items()})

    @classmethod
    def load_or_detect(cls, path: str | Path | None = None) -> HardwareProfile:
        p = Path(path) if path else DEFAULT_PROFILE_PATH
        return cls.load(p) if p.exists() else cls.detect()

    def save(self, path: str | Path = DEFAULT_PROFILE_PATH) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    def model(self, name: str) -> ModelProfile | None:
        return self.models.get(name)


async def measure_model(provider: Provider, model: str, max_tokens: int = 64) -> ModelProfile:
    """Measure load time (from cold) and prefill/decode throughput for one model."""
    for r in await provider.running():
        await provider.unload(r.name)
    load = await provider.load(model)
    msgs = [Message("user", _PROBE)]
    start = time.perf_counter()
    res = await provider.generate(msgs, GenerationParams(model, 0.0, max_tokens))
    wall = time.perf_counter() - start
    u = res.usage
    prefill = u.prompt_tokens / u.prompt_eval_seconds if u.prompt_eval_seconds else 0.0
    decode = u.completion_tokens / u.eval_seconds if u.eval_seconds else (u.completion_tokens / wall if wall else 0.0)
    size = next((r.size_bytes for r in await provider.running() if r.name == model), 0)
    if not size:
        size = next((m.size_bytes for m in await provider.list_models() if m.name == model), 0)
    return ModelProfile(size, round(load, 3), round(prefill, 1), round(decode, 1),
                        datetime.now(timezone.utc).isoformat(timespec="seconds"))


class Residency:
    """Which models are in memory, refreshed from the provider when it can tell."""

    def __init__(self, provider: Provider, profile: HardwareProfile) -> None:
        self._provider = provider
        self._profile = profile
        self._resident: dict[str, int] = {}
        self.known = provider.capabilities.residency_api

    async def refresh(self) -> dict[str, int]:
        if self.known:
            self._resident = {r.name: r.size_bytes for r in await self._provider.running()}
        return dict(self._resident)

    def models(self) -> list[str]:
        """Resident models as of the last refresh."""
        return list(self._resident)

    def is_resident(self, model: str) -> bool | None:
        return model in self._resident if self.known else None

    def would_evict(self, model: str) -> list[str]:
        """Models that loading ``model`` would push out under the memory budget (best effort)."""
        budget = self._profile.memory_budget_bytes
        if not self.known or budget is None or model in self._resident:
            return []
        prof = self._profile.model(model)
        need = prof.size_bytes if prof else 0
        if not need:
            return []
        used = sum(self._resident.values())
        evicted: list[str] = []
        for name, size in self._resident.items():  # assume oldest-first eviction
            if used + need <= budget:
                break
            evicted.append(name)
            used -= size
        return evicted
