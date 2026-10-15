# Contributing

Thanks for helping. The project is small on purpose: a core that can be read in an afternoon.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest            # no model server needed: tests use the simulated FakeProvider
.venv/bin/ruff check src tests bench examples
```

## Good first contributions

* **Verifiers** (`src/warmset/verify.py`): deterministic checks with clear failure messages and tests for
  both a passing and a failing output.
* **Providers** (`src/warmset/providers/`): implement `Provider`, set honest `Capabilities` (do not claim
  residency or load-time reporting the backend does not have), add `httpx.MockTransport` tests.
* **Benchmark suites** (`bench/suites/`): machine-checkable ground truth, a dev/val/test split, and
  verifiers that pass on every ground-truth answer (there is a check for that in the invoices suite).

## Rules for benchmark claims

* Never tune on the test split. Fit estimators on `dev`, choose settings on `val`, report `test`.
* Compare against the strongest simple baseline (`cascade`), not only against `fixed:<large>`.
* Report losses as well as wins, with the hardware, OS, model digests and repeats used.
* Results produced by replay or estimation are labelled as such; only reruns are "observed".

## Pull requests

Keep changes focused, add tests, and update `docs/design.md` if you change a contract (step,
policy, provider, ledger schema). Ledger schema changes need a version bump and a migration.
