# Design

## What problem this solves

On a single machine with local models, the expensive part of "use a bigger model when the small one
fails" is often not generation. It is **loading the bigger model and evicting the one the rest of the
pipeline needs**. On an 8 GB Apple M1 with Ollama, `llama3.2` (3B) and `qwen3.5:4b` cannot be resident
at the same time; every switch costs a reload (measured with `warmset profile`).

Routers and cascades in the literature mostly price models in dollars per token and assume every model
is already loaded. warmset prices each decision in **measured local cost** (load time, eviction, prefill
and decode rates), gates acceptance on **deterministic verifiers**, and records **every decision with
its prediction**, so the controller itself can be audited and its estimates calibrated.

None of the individual mechanisms are new; see [Related work](#related-work). The contribution is the
local cost model, the auditable decision record, and an honest benchmark against strong simple baselines.

## Concepts

```
pipeline (plain async Python)
  └─ @step(alternatives=[Deterministic(fn), Model("small"), Model("large")], verify=[...], features=...)
        controller loop per call:
          candidates  = eligible alternatives (budget, tried/exhausted)
          estimates   = p(pass | step, alternative[, input bucket])  from the ledger (Beta-Binomial)
                        cost = load if not resident + p_revisit * reload(evicted) + exec  (hardware profile)
          choice      = policy.choose(view)            -> run one alternative, or stop
          verdicts    = verifiers(output)              -> accept, or ask the policy again
          ledger      <- decision view (all candidates + estimates), action usage, verdicts
```

* **Step**: a unit with declared alternatives. Step bodies are ordinary async functions that call
  `ctx.generate(...)`; the controller decides which model `ctx` is bound to.
* **Alternative**: `Deterministic(fn)` (raise `NotApplicable` to pass) or `Model(name_or_alias)`.
* **Verifier**: `(output, call) -> Verdict | bool | str | None`. Crashing verifiers fail closed.
* **Policy**: `choose(view) -> Choice`, `accept(view, verdicts) -> bool`. Views are plain data, so a
  recorded decision can be fed to a different policy offline.
* **Ledger**: SQLite. Runs, steps, decisions (full candidate list with estimates and the reason), actions
  (usage including load seconds, predicted vs observed), verdicts, events, content-addressed blobs, labels.

## Policies

| spec | what it does | role |
|---|---|---|
| `fixed:<model>` | one model, one attempt | B0/B1 |
| `retry:<model>[:n]` | same model, rising temperature until verified | B4 ("AI Agents That Matter" baseline) |
| `cascade[:m1,m2]` | declared order, escalate on failed verification | **B3, the baseline to beat** |
| `cheapest` | cheapest eligible first, escalate on failure | B5 |
| `explore` | run every alternative once; keep first verified | data collection and hindsight analysis |
| `adaptive[:V]` | `EV = p * V - cost`; run best positive EV, else stop | the policy under test |
| `<spec>+batch` (bench) | run the whole split as one residency-scheduled batch | execution-order variant of any policy |

`adaptive` abstains (delegates to `cascade`) while any open alternative has fewer than `min_samples`
observations. `V` is an explicit exchange rate: how many seconds a verified output is worth.

## Batches and model scheduling

Every model call goes through a `ModelScheduler` (one call at a time by default). `Runtime.run_batch`
starts every input concurrently and switches the scheduler to **residency-first** mode: while anyone is
waiting for the loaded model it keeps serving that model, and it switches only when that queue is empty
(after a 20 ms grace so just-finished tasks can queue again). With a batch in flight, all small-model work
runs, the large model is loaded once for every escalation, and so on.

Two effects, worth keeping apart:

* **Concurrency alone groups calls** in a single-step pipeline: every first attempt is queued before any
  escalation arrives, so even FIFO order loads each model about once.
* **Residency-first ordering** matters when work arrives while another model is queued (multi-step
  pipelines): FIFO swaps back and forth, residency-first does not (`tests/test_scheduler.py`).

In a batch, policies see load and eviction costs divided by the number of runs still in flight. That is an
approximation (it assumes the other runs will want the same model) and is recorded in every decision view.
Per-call timeouts apply inside the scheduler, so queueing does not count against them.

Per-task numbers mean different things in the two modes: sequentially, a task's seconds are its latency;
in a batch, they are its share of the cost and the task may wait in a queue. Compare batches by seconds
per success and makespan, not by per-task latency.

## Three kinds of "what if"

| | what it is | valid for |
|---|---|---|
| `warmset replay` | recorded decision views fed to another policy | exact up to the first divergent decision, then stops |
| `lab.branch` | force an alternative and actually rerun, next to unforced controls | observed outcomes, with noise measured |
| model-based estimates | predicted outcomes of things never run | must be labelled as estimates |

Static replay of model switches is known to mislead for agents (Replay Gap, arXiv 2608.08239), so warmset
never presents replayed observations after a divergence as results.

## Safety

* `side_effects="external"` steps run at most one action; never retried by any policy.
* Retries happen only by explicit policy decision; the only automatic retry is a truncated answer
  getting double the token budget once.
* Bugs in step bodies are recorded and re-raised, never "escalated around" to a bigger model.
* Budgets (`seconds`, `model_calls`, `tokens`) are checked before every action against its estimated
  cost; an alternative that does not fit is ineligible.
* No sandbox in v0. Deterministic alternatives and verifiers run in-process.

## Decisions (short ADRs)

1. **Library, not a service.** Plain async Python steps; dependencies are `httpx` and `pydantic`.
2. **Structured pipelines first.** Open-ended agents are out of scope for v0 (no verifiers, unreliable
   counterfactuals).
3. **Agent Skills format** (`SKILL.md`, progressive disclosure) for on-demand instructions.
4. **SQLite ledger**, versioned schema (`PRAGMA user_version`). Exporters (OTel, Langfuse) later and optional.
5. **Measured cost, not price tables.** `warmset profile` writes a per-machine profile.
6. **Explicit value rule with abstention** instead of a learned black-box router.
7. **Ledger before controller.** Estimators are fitted from recorded outcomes (`explore` on a dev split),
   then frozen for evaluation so every policy sees the same information.
8. **Never call replayed data counterfactual results** (see above).

## Known limitations (v0)

* Pass-rate estimates are per step, alternative and optional hand-written feature bucket. They are not
  conditional on earlier failures (P(large passes | small failed) is treated as P(large passes)).
* Verifier pass is a proxy for correctness. `warmset calibrate` reports how good a proxy it is when labels exist.
* Residency comes from the provider (`/api/ps` for Ollama). OpenAI-compatible servers do not report it,
  so they are priced without load costs.
* The eviction model assumes oldest-first eviction under a memory budget (default 60% of RAM).
* Concurrency defaults to 1 model call at a time; parallel calls to different models thrash memory.

## Related work

* Routing: RouteLLM (lm-sys), Arch-Router, vLLM Semantic Router, LiteLLM, NVIDIA llm-router.
* Cascades: FrugalGPT; AutoMix (NeurIPS 2024); *A Unified Approach to Routing and Cascading for LLMs*
  (Dekoninck et al., ICML 2025, eth-sri/cascade-routing); Conformal Cascade (2026).
* Stopping and value of computation: Rational Metareasoning for LLMs; Adaptive-Consistency (EMNLP 2023);
  CAM-DF cost-aware stopping (2026); BATS budget-aware tool use (Google, COLM 2026).
* Step-wise routing in workflows: ProgRouter (2026), MasRouter (ACL 2025).
* Evaluation: *AI Agents That Matter* (Kapoor et al.); HAL (ICLR 2026); EcoAgent-Bench (2026);
  *The Replay Gap* (2026).
* Switching cost on edge devices: quality-latency aware edge routing with KV-cache switching costs (2025).
