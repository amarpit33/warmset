"""On-demand skills in the open Agent Skills format (https://agentskills.io).

Three tiers of progressive disclosure: the catalog (name + description) is cheap and always
available; a skill's ``SKILL.md`` body loads only when used; referenced files load only when
asked for. Skills are instructions, never executable code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


def parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    """Split ``SKILL.md`` into frontmatter (flat ``key: value`` lines) and body."""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t", "#")):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, m.group(2)


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    _body: str | None = field(default=None, repr=False)

    def body(self) -> str:
        """Tier 2: the instructions (read from disk on first use)."""
        if self._body is None:
            self._body = parse_skill_md((self.path / "SKILL.md").read_text())[1].strip()
        return self._body

    def resource(self, relative: str) -> str:
        """Tier 3: a referenced file inside the skill folder."""
        target = (self.path / relative).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise ValueError(f"resource '{relative}' escapes the skill folder")
        return target.read_text()


class SkillIndex:
    """Discovers ``*/SKILL.md`` folders under one or more roots, reading only frontmatter."""

    def __init__(self, *roots: str | Path) -> None:
        self.skills: dict[str, Skill] = {}
        for root in roots:
            for md in sorted(Path(root).glob("*/SKILL.md")):
                meta, _ = parse_skill_md(md.read_text())
                name = meta.get("name") or md.parent.name
                self.skills[name] = Skill(name, meta.get("description", ""), md.parent)

    def __getitem__(self, name: str) -> Skill:
        try:
            return self.skills[name]
        except KeyError:
            raise KeyError(f"unknown skill '{name}'; known: {sorted(self.skills)}") from None

    def catalog(self) -> str:
        """Tier 1 text suitable for a system prompt."""
        return "\n".join(f"- {s.name}: {s.description}" for s in self.skills.values())

    def rank(self, query: str, k: int = 3) -> list[Skill]:
        """Deterministic lexical ranking of skills for a query (no model call)."""
        words = set(re.findall(r"[a-z0-9]{3,}", query.lower()))

        def score(s: Skill) -> int:
            return len(words & set(re.findall(r"[a-z0-9]{3,}", f"{s.name} {s.description}".lower())))

        ranked = sorted(self.skills.values(), key=lambda s: -score(s))
        return [s for s in ranked[:k] if score(s) > 0]
