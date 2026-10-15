# warmset

[![CI](https://github.com/amarpit33/warmset/actions/workflows/ci.yml/badge.svg)](https://github.com/amarpit33/warmset/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Pre-alpha (0.0.x): APIs and the ledger schema will change.

**Local-first adaptive execution for LLM pipelines.** Each step declares its alternatives: a deterministic
function, a small model, a bigger model. warmset picks one per input using what that choice *actually costs
on your machine* (model load, eviction, prefill and decode speed), checks the output with deterministic
verifiers, escalates only when it is worth it, and records every decision with its prediction so you can
see why it chose what it chose.

Real `warmset explain` output (cascade policy, 8 GB M1; trimmed):

```text
#1 extract_invoice  -> accepted via model:qwen3.5:4b@t0
  decision 2: model:llama3.2:latest@t0   [input bucket: line_items]
    - model:llama3.2:latest@t0   p=0.33 (n=10, bucket:line_items)  est 2.79s [load 0.0 + evict 0.0]  resident
    - model:qwen3.5:4b@t0        p=0.75 (n=10, bucket:line_items)  est 14.24s [load 5.6 + evict 2.7]
    observed: ok in 3.41s -> total_supported=fail (total 3699.66 is neither printed nor the sum of line amounts)
  decision 3: model:qwen3.5:4b@t0   [input bucket: line_items]
    observed: ok in 15.81s (load 7.86s; predicted 14.24s) -> all verifiers pass
```

## Why

On an 8 GB laptop, two small models do not fit in memory together. "Escalate to the bigger model" means
unloading one and loading the other, and the rest of your pipeline pays to load it back. Routers and
cascades usually price models in dollars per token and assume everything is loaded. warmset measures the
local costs (`warmset profile`) and prices decisions with them.

None of the mechanisms (routing, cascades, value-of-computation stopping, budgets, replay) are new; see
[docs/design.md](docs/design.md#related-work). What warmset adds is the local cost model, an auditable
decision ledger, residency-aware scheduling of model calls across a batch, and a benchmark protocol that
compares against strong simple baselines.

## Quickstart

```bash
pip install -e .
python examples/quickstart.py --fake     # simulated models, no server needed
ollama pull llama3.2 && ollama pull qwen3.5:4b
warmset profile --models llama3.2:latest qwen3.5:4b --out .warmset/profile.json
python examples/quickstart.py
```

```python
from warmset import Budget, Deterministic, Model, Runtime, StepContext, grounded, step
from warmset.providers import OllamaProvider

@step(alternatives=[Deterministic(parse_form), Model("small"), Model("large")],
      verify=[grounded("vendor"), total_supported],
      features=shape)                       # optional: cheap input buckets for per-bucket pass rates
async def extract_invoice(ctx: StepContext, text: str) -> Invoice:
    return await ctx.generate(system=SYSTEM, user=text, schema=Invoice)

rt = Runtime(OllamaProvider(), models={"small": "llama3.2:latest", "large": "qwen3.5:4b"},
             ledger=".warmset/ledger.db")
result = await rt.run(extract_invoice, text, policy="adaptive:60", budget=Budget(seconds=60))

# many independent inputs: model calls grouped by the loaded model, so each model loads about once
results = await rt.run_batch(extract_invoice, [{"text": t} for t in texts], policy="cascade")
```

## Commands

```bash
warmset profile                 # measure load time and throughput per model on this machine
warmset bench bench.suites.invoices --policies cascade cascade+batch adaptive:60 adaptive:60+batch --repeats 3
warmset runs                    # recent runs
warmset explain <run-id>        # every decision: candidates, estimates, choice, observed outcome
warmset replay <run-id> --policy fixed:large   # what another policy would have chosen, up to divergence
warmset calibrate               # predicted vs observed pass rates; how often "verified" meant "correct"
```

## Results

On an 8 GB M1 with two local models that cannot be loaded together (40 held-out synthetic invoices,
3 repeats, settings tuned on a separate validation split; details and caveats in
[docs/results.md](docs/results.md)):

| | success | seconds per success | model loads per 40 tasks |
|---|---|---|---|
| always the large model | 0.95 | 7.25 | 1 |
| verify-then-escalate cascade | 0.95 | 5.68 | 19 |
| cascade, **batched by loaded model** | 0.95 | **3.96** (-30%, CI excludes 0) | 2 |
| adaptive policy, batched | 0.95 | 3.91 | 3 |

The gain comes from **scheduling model calls by what is loaded**, not from the adaptive decision rule:
adaptive vs cascade within batch mode is a tie. Sequentially, the adaptive policy cut p95 per-request cost
from 18.1 s to 7.4 s by avoiding model swaps, with no significant change in the mean.

## Status and limits

Structured pipelines only (no open-ended agents). No sandbox. Pass-rate estimates are per step,
alternative and optional input bucket, not conditional on earlier failures. Full list in
[docs/design.md](docs/design.md#known-limitations-v0).

MIT licensed.
