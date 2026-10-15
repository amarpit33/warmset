import asyncio

import pytest

from tests.helpers import make_profile, make_provider
from warmset.estimators import BudgetTracker, CostModel, LatencyEstimator, SuccessEstimator
from warmset.hardware import Residency
from warmset.types import Budget, Usage


def test_beta_estimator():
    e = SuccessEstimator()
    assert e.mean("s", "k") == 0.5 and e.n("s", "k") == 0
    for passed in (True, True, True, False):
        e.update("s", "k", passed)
    assert e.mean("s", "k") == pytest.approx(4 / 6)
    assert e.lcb("s", "k") < e.mean("s", "k")
    assert SuccessEstimator.from_rows([("s", "k", True)]).n("s", "k") == 1


def test_cost_model_prices_load_and_eviction():
    async def go():
        provider = make_provider()
        profile = make_profile()
        res = Residency(provider, profile)
        await provider.load("small")
        await provider.load("large")  # 2 + 4 GB > 5 GB budget -> small evicted in the fake
        await res.refresh()
        assert res.is_resident("large") and not res.is_resident("small")
        lat = LatencyEstimator()
        cm = CostModel(profile, lat, revisit_probability=0.5)
        c_small = cm.estimate("s", "model:small@t0", "small", res)
        assert c_small.load_seconds == 4.0
        assert c_small.evict_seconds == pytest.approx(0.5 * 20.0)  # loading small evicts large (5 GB budget)
        c_large = cm.estimate("s", "model:large@t0", "large", res)
        assert c_large.load_seconds == 0.0 and c_large.source == "profile"
        lat.update("s", "model:large@t0", 7.0)
        assert cm.estimate("s", "model:large@t0", "large", res).exec_seconds == 7.0
        assert cm.estimate("s", "fn:x", None, res).seconds < 0.01

    asyncio.run(go())


def test_budget_tracker():
    b = BudgetTracker(Budget(seconds=10, model_calls=2))
    assert b.affordable(9.9, True) and not b.affordable(10.1, True)
    b.charge(Usage(seconds=6, load_seconds=4, calls=1))
    assert b.loads == 1 and b.remaining_seconds == 4
    b.charge(Usage(seconds=1, calls=1))
    assert not b.affordable(0.1, True) and b.affordable(0.1, False)
    assert b.exhausted
