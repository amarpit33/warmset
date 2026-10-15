import pytest

from warmset.policies import (
    Adaptive,
    Attempt,
    Candidate,
    Cascade,
    Cheapest,
    DecisionView,
    Explore,
    Fixed,
    Retry,
    parse_policy,
)


def cand(key, model=None, *, p=0.5, n=10, cost=1.0, kind="model", tries=0, exhausted=False, affordable=True):
    return Candidate(key=key, kind=kind, model=model, est_seconds=cost, p_success=p, n=n, tries=tries,
                     exhausted=exhausted, affordable=affordable)


def view(*cands, history=()):
    return DecisionView("s", len(history) + 1, list(cands), list(history))


def test_adaptive_prefers_cheap_resident_model_when_good_enough():
    v = view(cand("small", "small", p=0.85, cost=3), cand("large", "large", p=0.95, cost=30))
    assert Adaptive(value_seconds=60).choose(v).key == "small"


def test_adaptive_escalates_when_small_rarely_passes():
    v = view(cand("small", "small", p=0.1, cost=3), cand("large", "large", p=0.9, cost=30))
    assert Adaptive(value_seconds=60).choose(v).key == "large"


def test_adaptive_stops_when_no_positive_value():
    v = view(cand("large", "large", p=0.3, cost=30))
    c = Adaptive(value_seconds=60).choose(v)
    assert c.stop and "no positive value" in c.reason


def test_adaptive_abstains_to_fallback_on_thin_data():
    v = view(cand("small", "small", p=0.1, n=2, cost=3), cand("large", "large", p=0.9, cost=30))
    c = Adaptive(min_samples=5).choose(v)
    assert c.key == "small" and c.reason.startswith("abstain")


def test_adaptive_skips_tried_exhausted_and_unaffordable():
    v = view(cand("small", "small", p=0.9, cost=3, tries=1, exhausted=True),
             cand("large", "large", p=0.9, cost=30, affordable=False))
    assert Adaptive().choose(v).stop


def test_cascade_order_and_restriction():
    v = view(cand("fn", kind="fn"), cand("small", "small"), cand("large", "large"))
    assert Cascade().choose(v).key == "fn"
    assert Cascade(["large", "small"], include_fn=False).choose(v).key == "large"
    v.candidates[0].tries = 1
    assert Cascade().choose(v).key == "small"


def test_fixed_single_attempt():
    v = view(cand("small", "small"), cand("large", "large"))
    assert Fixed("large").choose(v).key == "large"
    v.history.append(Attempt("large", 0.0, True, False))
    assert Fixed("large").choose(v).stop


def test_retry_raises_temperature():
    pol = Retry("small", (0.0, 0.7))
    v = view(cand("small", "small"))
    assert pol.choose(v).temperature == 0.0
    v.history.append(Attempt("small", 0.0, True, False))
    assert pol.choose(v).temperature == 0.7
    v.history.append(Attempt("small", 0.7, True, False))
    assert pol.choose(v).stop


def test_cheapest_and_explore():
    v = view(cand("a", "a", cost=9), cand("b", "b", cost=2))
    assert Cheapest().choose(v).key == "b"
    assert Explore().choose(v).key == "a"
    assert Explore().accept(v, []) is False


def test_view_roundtrip():
    v = view(cand("a", "a"), history=[Attempt("a", 0.0, True, False)])
    assert DecisionView.from_dict(v.to_dict()) == v


@pytest.mark.parametrize("spec,name", [
    ("fixed:s", "fixed:small"), ("retry:s:2", "retry:small:2"), ("retry:s", "retry:small:3"),
    ("cascade", "cascade"), ("cascade:s,l", "cascade:small,large"), ("cheapest", "cheapest"),
    ("explore", "explore"), ("adaptive:30", "adaptive:v30"), ("adaptive", "adaptive:v60"),
])
def test_parse_policy(spec, name):
    assert parse_policy(spec, {"s": "small", "l": "large"}).name == name


def test_parse_policy_rejects_unknown():
    with pytest.raises(ValueError):
        parse_policy("magic")
