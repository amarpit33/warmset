import pytest
from pydantic import BaseModel

from warmset.skills import SkillIndex, parse_skill_md
from warmset.verify import Call, grounded, grounded_ids, matches, nonempty, numbers_preserved, run_verifier


def test_skill_tiers(tmp_path):
    d = tmp_path / "invoice-extraction"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: invoice-extraction\ndescription: Extract invoice fields\n---\n"
                                "Read the invoice. Use ISO dates.\n")
    (d / "ref.md").write_text("currency codes")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "SKILL.md").write_text("no frontmatter here")
    idx = SkillIndex(tmp_path)
    assert set(idx.skills) == {"invoice-extraction", "other"}
    sk = idx["invoice-extraction"]
    assert sk._body is None  # tier 2 not loaded until used
    assert "invoice-extraction: Extract invoice fields" in idx.catalog()
    assert sk.body() == "Read the invoice. Use ISO dates."
    assert sk.resource("ref.md") == "currency codes"
    with pytest.raises(ValueError):
        sk.resource("../other/SKILL.md")
    assert idx.rank("extract fields from this invoice")[0].name == "invoice-extraction"
    with pytest.raises(KeyError):
        idx["missing"]


def test_parse_skill_md_without_frontmatter():
    assert parse_skill_md("just text") == ({}, "just text")


class Out(BaseModel):
    vendor: str
    total: float
    ids: list[str]


CALL = Call(("Acme Corp invoice INV-7, total 1,234.50, items R1 R2",), {})


def outcome(v, out):
    return run_verifier(v, out, CALL).outcome


def test_builtin_verifiers():
    good = Out(vendor="acme corp", total=1234.5, ids=["R1"])
    bad = Out(vendor="Globex", total=99.0, ids=["R9"])
    assert outcome(grounded("vendor"), good) == "pass"
    assert outcome(grounded("vendor"), bad) == "fail"
    assert outcome(grounded_ids("ids", lambda c: ["r1", "r2"]), good) == "pass"
    assert outcome(grounded_ids("ids", lambda c: ["r1", "r2"]), bad) == "fail"
    assert outcome(numbers_preserved("total"), good) == "pass"
    assert outcome(numbers_preserved("total"), bad) == "fail"
    assert outcome(nonempty("vendor"), good) == "pass"
    assert outcome(matches(r"[A-Z][a-z]+", "vendor"), bad) == "pass"


def test_crashing_verifier_fails_closed():
    def boom(out, call):
        raise RuntimeError("bug")
    v = run_verifier(boom, None, CALL)
    assert v.outcome == "fail" and "RuntimeError" in v.evidence
