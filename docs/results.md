# Results

## Run 2: batch scheduling, values tuned on val, 3 repeats (2026-10-14/15)

**Setup.** Same machine and models as run 1 below (Apple M1 8 GB, macOS 26.6.2, Ollama 0.35.0,
`llama3.2:latest` a80c4f17acd5, `qwen3.5:4b` 2a654d98e6fb; the two cannot be resident together).

**Protocol.**
1. Estimators fitted from the existing `explore` pass on the 40-task dev split (reused from run 1).
2. **Tuning on val only** (20 tasks, 1 repeat): `cascade` and `adaptive:{15,30,60,120}`, each sequential
   and `+batch`. Rule fixed in advance: lowest seconds per success with success within 2 points of the
   baseline. Picks: `adaptive:30` (sequential), `adaptive:15+batch` (batch).
   Raw: [results/invoices-val-m1-8gb-2026-10-14.json](../results/invoices-val-m1-8gb-2026-10-14.json).
3. **Test** (40 tasks x **3 repeats**, policy order shuffled per repeat, models unloaded before each block).
   The test split had been run once before (run 1); nothing was tuned on it.
   Raw: [results/invoices-test-r3-m1-8gb-2026-10-14.json](../results/invoices-test-r3-m1-8gb-2026-10-14.json).
4. **Drift guard**: every block records median model execution time per model; blocks >= 1.3x slower than
   the median block are flagged. **No block was flagged.** A first attempt at this run was stopped and
   discarded: from about 23:05 the machine slowed 2-3x (swap nearly full, cause not identified), hitting
   the `cascade` blocks of repeat 2. The guard was added after that, and it flags exactly those blocks when
   applied to the discarded data.

| policy | success | s / success | makespan (s) | loads / task | vs cascade: s/task [95% CI] | vs cascade: p95 [95% CI] |
|---|---|---|---|---|---|---|
| fixed:small | 0.775 | 3.54 | 110.7 | 0.03 | -2.65 [-4.56, -0.83] | -15.22 [-15.44, -9.82] |
| fixed:large | 0.950 | 7.25 | 276.8 | 0.03 | +1.49 [-0.35, +3.28] | -10.63 [-11.13, -2.97] |
| **cascade** (baseline) | 0.950 | 5.68 | 217.0 | 0.47 | | |
| adaptive:30 | 0.950 | 5.14 | 196.6 | 0.05 | -0.51 [-2.09, +0.92] | **-11.09 [-11.55, -1.67]** |
| cascade+batch | 0.950 | **3.96** | 151.6 | 0.05 | **-1.63 [-2.56, -0.85]** | n/a (see note) |
| adaptive:15+batch | 0.950 | **3.91** | 149.8 | 0.07 | **-1.68 [-2.85, -0.54]** | n/a (see note) |
| hindsight best* | 0.950 | 2.73 | | | | |

Seconds = provider-reported action time including model loads, averaged over 3 repeats per task; CIs
are paired bootstrap over the 40 tasks. Makespan = wall time for the whole 40-task block (mean of 3).
\* fastest correct evaluated policy per task, in hindsight; not achievable.
Note: in `+batch` rows a task's seconds are its share of cost and tasks wait in a queue, so per-task p95 is
not latency; compare batches by seconds per success and makespan.

Within batch mode, `adaptive:15+batch` vs `cascade+batch`: -0.04 s/task [-0.78, +0.68], same success.

### What this shows

1. **Batch scheduling is the win, and it is significant.** Running the 40 tasks as one batch, with model
   calls grouped by the loaded model, cuts seconds per success from 5.68 to 3.96 (**-30%**, CI excludes
   zero) and makespan from 217 s to 152 s, at the same success rate. Model loads per task drop from 0.47 to
   0.05 (19 loads per block to 2). This holds for both policies.
2. **The adaptive policy adds nothing measurable on top of batching** (3.91 vs 3.96, CI spans zero).
   Once loads are amortised, the choice between the models matters little on this suite.
3. **Sequentially, the adaptive policy cuts tail cost, not mean cost.** `adaptive:30` vs `cascade`:
   p95 per-task seconds 7.4 vs 18.1 (CI excludes zero) by avoiding swaps (2 loads per block vs 19), but the
   mean difference (-0.51 s/task) is within noise. The value of that depends on whether you care about tail
   latency of individual requests.
4. **Headroom remains.** Hindsight best is 2.73 s/success, vs 3.91 for the best policy.
5. **Gates** (targets from the plan): success parity holds for every non-fixed policy (G1). The cost gate
   (G2: >= 20% lower seconds per success or >= 25% lower p95) is met by both batch policies (-30% s/success)
   and by `adaptive:30` on p95 (-59%). It is met by *scheduling*, not by the adaptive decision rule. G3
   (safety, budgets) was not exercised. G4 calibration: see run 1.

### Caveats

* One machine, one model pair, synthetic invoices; 40 test tasks. Three repeats measure run-to-run noise,
  not generality.
* Batch mode needs many independent inputs available at once; it does not help a single interactive request.
* The amortised-cost approximation in batch mode is untested beyond this suite.

## Run 1: first benchmark (2026-10-14), sequential only

**Setup.** macOS 26.6.2, Apple M1, 8 GB RAM, Ollama 0.35.0. Models: `small` = `llama3.2:latest`
(a80c4f17acd5), `large` = `qwen3.5:4b` (2a654d98e6fb), thinking off, temperature 0. The two models
**cannot be resident together** on this machine (verified with `/api/ps`), so every switch is a reload.
Profile: [results/profile-m1-8gb-2026-10-14.json](../results/profile-m1-8gb-2026-10-14.json)
(llama3.2: load 5.4 s, 231 prompt tok/s, 26 tok/s decode; qwen3.5:4b: load 5.6 s, 120 / 16 tok/s).

**Protocol.** `explore` on the 40-task dev split (every alternative on every task) to fit pass rates
and latencies; estimators frozen; then each policy on the 40-task held-out test split, policies in
shuffled order, models unloaded before each policy block. **One repeat.** Seconds are provider-reported
action time including model loads. Raw rows: [results/invoices-test-m1-8gb-2026-10-14.json](../results/invoices-test-m1-8gb-2026-10-14.json).

Reproduce:

```bash
warmset profile --models llama3.2:latest qwen3.5:4b --out .warmset/profile.json
warmset bench bench.suites.invoices --policies fixed:small fixed:large cascade retry:small adaptive:15 adaptive:60 \
  --split test --fit-split dev --profile .warmset/profile.json
```

| policy | success | s / success | p50 s | p95 s | model calls | loads | vs cascade: success [95% CI] | vs cascade: s/task [95% CI] |
|---|---|---|---|---|---|---|---|---|
| fixed:small | 0.775 | 3.82 | 2.82 | 3.18 | 1.00 | 0.03 | -0.175 [-0.300, -0.075] | -3.07 [-5.25, -1.02] |
| retry:small | 0.775 | 5.37 | 3.01 | 8.21 | 1.45 | 0.03 | -0.175 [-0.300, -0.075] | -1.87 [-3.48, -0.35] |
| fixed:large | 0.950 | 8.81 | 8.01 | 10.00 | 1.00 | 0.03 | +0.000 [+0.000, +0.000] | +2.34 [+0.07, +4.44] |
| **cascade** (baseline) | 0.950 | 6.35 | 2.54 | 19.93 | 0.97 | 0.47 | | |
| adaptive:15 | 0.825 | 3.59 | 2.54 | 8.18 | 0.78 | 0.05 | -0.125 [-0.225, -0.025] | -3.07 [-5.23, -1.23] |
| adaptive:60 | 0.950 | 6.56 | 7.12 | 13.65 | 0.82 | 0.12 | +0.000 [+0.000, +0.000] | +0.20 [-1.56, +1.75] |
| hindsight best* | 0.950 | 3.01 | 2.36 | 7.99 | | | | |

\* Per task, the fastest evaluated policy that got it right. A headroom indicator, not an achievable policy.

Per layout (correct / 10, mean seconds per task, total model loads):

| policy | form | email | itemized | distractor |
|---|---|---|---|---|
| fixed:small | 10, 2.7 s | 9, 2.8 s | **2**, 3.0 s | 10, 3.2 s |
| fixed:large | 10, 8.2 s | 10, 8.3 s | 8, 8.3 s | 10, 8.7 s |
| cascade | 10, **0.0 s** | 10, 4.1 s, 2 loads | 8, **15.9 s, 12 loads** | 10, 4.2 s, 5 loads |
| adaptive:60 | 10, 0.0 s | 10, 8.5 s | 8, 9.3 s, 3 loads | 10, 7.1 s |

## What this shows

1. **The verify-then-escalate cascade is a strong baseline.** It matches the large model's success rate
   (0.95) at 28% fewer seconds per success, mostly because the regex path handles the form layout for free.
2. **adaptive:60 did not beat it on mean cost.** Same success, +0.20 s/task (CI includes zero). It
   did cut model loads from 0.47 to 0.12 per task and P95 from 19.9 s to 13.7 s, but with one repeat and
   no CI on P95 that is not a claim, only a lead.
3. **Why it did not win: myopic stickiness.** Once `qwen3.5:4b` is resident, switching back to llama costs
   load plus the expected reload of qwen, so adaptive:60 stays on the large model even for inputs the small
   model always passes (single_amount: chose qwen first 9/10 times; llama passed 20/20 on dev). It
   trades swaps for slower generation and roughly breaks even.
4. **The waste is in the sequence, not the single decision.** Cascade's itemized tasks cost 15.9 s each
   because each one swaps to qwen and the next task swaps back. Hindsight-best is 3.0 s/success vs 6.35
   for cascade: real headroom, but a per-call policy cannot capture it. Grouping work by the model it
   needs (residency-aware batch scheduling) is the next hypothesis to test.
5. **adaptive:15 shows the trade-off dial.** Valuing a verified output at 15 s, it accepts small-model
   answers more often: 3.07 s/task cheaper, 12.5 points less success.
6. **Verifiers were a perfect proxy here.** Across all 240 test runs, every fully verified run was
   correct (209/209) and every run with an unverified step was wrong (31/31). Estimator calibration on cascade + adaptive:60: ECE 0.063,
   Brier 0.082. The suite is synthetic, so expect a weaker proxy on real documents.

## Against the plan's gates (targets, not achievements)

| gate | target | adaptive:60 here |
|---|---|---|
| G1 success not worse than cascade by > 2 points | CI lower bound >= -0.02 | met (0.000 [0.000, 0.000]) |
| G2 >= 20% lower median s/success or >= 25% lower P95 | | **not met** on s/success; P95 -31% point estimate, single repeat, no CI |
| G3 zero safety failures, no budget overrun | | not exercised (no budget set, no side-effect steps) |
| G4 ECE <= 0.10 | | met (0.063), one run |

## Caveats

* One repeat, 40 test tasks, one machine, synthetic invoices. Not enough for a public claim.
* `V` (15 and 60) was not tuned on the validation split; both values were run once on test and both
  are reported. Future tuning happens on `val` only.
* The `explore` fit observes each model from a different residency state than evaluation; latency
  estimates exclude load, which is priced separately from the profile.
