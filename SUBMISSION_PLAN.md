# ATLAS — Phased execution plan: fix the code, defend the claims, run the continual test

## Context

Deadline ~85h. Budget ~$90 (Modal L4 $0.80/GPU-h, T4 $0.59/GPU-h).

The `research_audit/` pass is complete and trustworthy on arithmetic: ~60 headline
numbers independently recomputed, **zero errors**; `atlas/stats.py` clean; pairing
genuine; encoder genuinely frozen; the 2026-08-25 rollout fix correct. The problems are
in **experimental design**, in **never-run code**, and in **a set of open items the
audit itself flagged as unreached**.

This plan is organised as **7 phases, each owned by a delegated sonnet sub-agent**, so
no single context has to carry the whole job. Every finding from `CODE_AUDIT.md`,
`PROPOSAL_CODE_ALIGNMENT.md`, `RESULTS_AUDIT.md`, `NEXT_ACTIONS.md`, `REDTEAM.md`,
`PAPER_FACT_CHECK.md` and `OPUS_REMAINING_TASKS.md` appears in the register in Part B
with an **owning phase and a concrete fix** — not just a description.

**Measured throughput** (computed from `wall_time` in the JSONLs this session, not from
any doc): **~148 s per CEM replan** at `300×30×horizon 6` on L4, invariant across every
run on disk. nas=6 → ~150 s/ep; nas=2 → ~350 s/ep; nas=1 → ~890 s/ep at iterations=30,
~300 s/ep at iterations=10. Peak VRAM 6.45 GB. Sharding N ways costs the same total
GPU-h at 1/N wall clock.

**Guiding rule.** No silent fixes. Every change is logged in `research_audit/FIXLOG.md`
(defect, `file:line`, change, claim affected, re-run needed y/n). Numbers invalidated by
a fix are superseded in place with a dated banner, never deleted. `atlas/score.py::umf`,
`atlas/stats.py`'s existing functions and `run_e0_planning.py`'s planning loop are
**not** modified — they produced the results on disk. Additions only.

**Verification rule.** A sub-agent reporting "done" is not evidence. Each phase has an
**exit gate** I re-run and read myself. For fixes that repair a *dead* mechanism, the
assertion must be shown to **fail before the fix** — otherwise it is vacuous, which is
exactly what made gates G2 and G5 worthless here.

**Agent dispatch rule.** Sub-agents run **strictly sequentially, one at a time** — never
in parallel. I dispatch a phase's agent, wait for it, verify its exit gate myself, and
only then dispatch the next. This keeps rate-limit pressure low and, more importantly,
means a defect introduced in one phase cannot silently propagate into work another agent
is doing concurrently against the same files.

---

# PART A — Findings closed by research during this planning session

Four items the audit listed as never-checked, now checked. Two cleared, two are new
findings.

**A-i. Shard merging is CLEAN — concern cleared.** `CODE_AUDIT.md` flagged
`scripts/merge_planning_shards.py` as never audited and "a classic place to duplicate or
drop episodes." Read in full and verified empirically: it hard-fails with `ValueError`
on any duplicate episode index (`:50-57`), sorts by episode (`:58`), and filters
nothing. For both `e0_planning_sweep_60` and `e0_planning_sweep_100`, merged indices are
exactly the union of shards 0-19 and 20-39, contiguous, sorted, **0 content mismatches**
against the shards. **N4's pairing against `e0_planning_n100/baseline_R2.jsonl`
episodes 0-39: 0 mismatches across 120 field comparisons per sweep.** N4's data is
sound. Only gap: the script does not check contiguity, so an upstream-missing episode
would pass silently — add an assertion.

**A-ii. `make_t5` does not exist.** `scripts/make_tables.py` offers `--table T5` in its
argparse choices (`:179`) but `main()` (`:186-189`) dispatches only T1 and T2. Passing
`--table T5` **silently does nothing** — no error, no output. The E0 capacity table the
proposal calls supplementary has never been generated. Same class of defect in
`scripts/make_figures.py`: `--fig` accepts only `{F1, F2}` (`:114`) while
`atlas/plots.py` defines `umf_traces`, `crosspolicy` and `umf_vs_sr` (S1/S2/S3) that
nothing dispatches. **Fix:** wire the missing dispatches, or remove the dead choices so
they cannot silently no-op.

**A-iii (NEW, not in any audit file). The released artifacts record the discredited
LoRA parameter count.** `ATLAS_SUMMARY.md` §3 states the count was corrected from
10,292,640 to 118,176 and that "**every comparison in this project now uses
118,176**". That is false at the artifact level: **both** `atlas_out/e0_v4_lora4/results.json`
and `atlas_out/e0_v6_R1/results.json` still record `params=10292640`.
*Mechanism, confirmed:* `Chart.n_params()` (`atlas/chart.py:169-170`) sums `_params`,
which at construction holds the 12 **full base matrices**; the `.lora_A`/`.lora_B`
entries are only added later by `update_from_predictor_` (`:172-190`). `run_e0.py`
records `params` at construction, so it captures 10.3M. `PAPER_DRAFT.md` Appendix B
says 118,176 — so **the paper and the released artifact disagree**, and a reviewer
checking the artifacts finds the number the summary says was retracted.
*Fix:* record the trainable count (numel over `requires_grad` params, or over
`lora_A`/`lora_B` only) and regenerate the affected `results.json` files with a dated
supersede note. $0.
*(Also historical: `atlas_out/e0/results.json` and `e0_pre_regime_fix_*` record
`params=26/12/69` — parameter *group* counts, an older bug. Superseded, leave as is.)*

**A-iv. Every input for the dissociation figure (`OPUS_REMAINING_TASKS.md` #21) is
local, and the picture is sharper than the doc claims.** Assembled this session:

| chart | eval_umf | planning SR (N=20) |
|---|---:|---:|
| baseline `c0` | 0.367 *(from `umf_locality.json`)* | 0.450 |
| `ln_act`/dataset | 0.336 | **0.500** |
| `lora4`/dataset | **0.329 (best UMF)** | 0.400 |
| `ln_act`/hybrid | 0.367 | 0.400 |
| `ln_act`/closed_loop | 0.423 | 0.350 |
| `full`/dataset | 0.728 (worst) | 0.200 (worst) |

UMF ranks `lora4` best and it plans *worse* than `ln_act`. The only chart where UMF and
SR agree is `full`, the outlier — **drop `full` and the ranking over the remaining five
is uninformative or inverted.** That is the paper's central claim in one figure, it
costs $0, and `atlas/plots.py::umf_vs_sr` already exists (it just needs a cross-arm
variant and a dispatch). Elevate this from a to-do item to a body figure.

**A-v. `harness_e4.py` does NOT inherit E1's three defects — cleared, and it validates
Phase 5's design.** `CODE_AUDIT.md` flagged that `harness_e4.py` claims to copy
`run_e1_episode` "verbatim" and that the copy was never diffed. Diffed in full: the
"verbatim" claim scopes only to the chunk re-encoding block (`harness_e4.py:158-162`
says so explicitly), and on the two defects that matter E4 uses the **corrected** path —
`sample_dataset_init_goal` (`harness_e4.py:177`) not E1's random-goal sampler, and
`block_success()` (`:269-271`) not `goal_utils.eval_state()`. E4 also adds a
`max_raw_steps` early-exit E1 lacks. Env construction/reset, proprio threading and
action chunking are genuinely identical. **This is why Phase 5 can route around C10.**

**A-vi. The encode → squeeze → flatten tensor chain is safe, and safe by construction
rather than by luck.** Both squeezes target *fixed, known-constant* axes: `.squeeze(0)`
removes a batch dim that is 1 by call-site construction, `.squeeze(1)` removes a tubelet
dim that is a **hardcoded literal `1`** in `EncPredWM.encode`'s einops pattern — so
neither can accidentally collapse `T` even when `T==1`. All three call sites
(`harness.py:413-415`, `harness_e4.py:288-290`, `run_e0_planning.py:338-340`) are
character-identical, and the output matches what `score.py::umf` and `_make_z_ctxt`
require. *Residual:* safety rests on the unasserted invariant "these sites never batch
multiple episodes". **Fix: assert it** (Phase 2, cheap).

**A-vii. RESULTS_AUDIT §8's open question is CLOSED, in the favourable direction.** The
question was whether E2 Cell C's two tau-crossing chunks producing 0 commits meant the
probe fired and correctly rejected, or never fired at all — "plausible but **not
verified**". Traced definitively: at `q=1`, `record()` sets `strikes=1 >= q` on the same
call, and `run_e2.py:328-334` then calls `maybe_expand()` **unconditionally**, so the
crossing chunk reaches the probe on that same episode. Library is 2/10 (not full), the
chunks were already confirmed ungated, and no intervening chunk resets the strike. The
probe was genuinely fit and verified against the next episode's chunk, and returned
`"rejected_score"`. **So Cell C's story is "the probe fired and correctly declined to
commit", not "it never fired"** — which `PAPER_FACT_CHECK` A4 already noted is the
*stronger* demonstration. Now verified rather than assumed; report it that way.

**A-viii (NEW). The ~100% contact rate comes from rejection sampling, not from the
aimed-walk scheme — and that conditions the training distribution.**
`ACTION_SAMPLING_REVIEW.md` frames the persistent-target aimed walk as reliable "by
construction". The implementation's own inline math (`run_e0.py:326-341`) says
otherwise: after the `ACTION_GAIN=0.25` tuning needed to match the checkpoint's action
distribution, the aimed walk alone reaches **~43% single-attempt contact**, and the
headline ~100% is the **retry-until-contact loop** compounding it
(`1−(1−0.43)^8 ≈ 0.989`, `max_tries=8`, accept condition at `:351-352`).
*Why this matters scientifically, not just editorially:* every training trajectory is
**rejection-sampled on having made contact**, so the chart's training distribution is
conditioned on contact, while CEM's candidate distribution makes contact ~80% of the
time and is not conditioned at all. That is a fourth, previously undocumented axis of
train/deploy mismatch, and it belongs in Part C-1's table.
*Fix:* disclose it; record the realised single-attempt contact rate in the seed manifest
so the conditioning is measurable rather than inferred.

**A-ix (NEW, CRITICAL, would have wasted $38). `modal_e4.py`'s `seed_run` does not vary
the stream — it only relabels the output.** `modal_e4.py:109` always passes
`--seeds 1` to the subprocess (a *count*, not the requested seed). Inside `run_e4.py`,
`for seed_run in range(profile_seeds)` therefore only ever runs local `seed_run=0`;
`get_stream(...)` is built with `seeds=1` so `stream_s2` only generates its `seed_run=0`
stream; and `local_seed` is hardcoded to `0`. `modal_e4.py:126,138` then **rewrites each
record's `seed_run` field after the fact**. So a "3-seed" sweep launched this way
produces **bit-identical episode data under three different labels** — not independent
replications. This is exactly the failure mode that produces a plausible-looking
variance band from one run. **It directly invalidates Stretch A as specified**; the fix
(pass the real seed through to `--seeds`/stream construction and to `local_seed`) is
small but must land before any multi-seed spend.

**A-x (NEW, CRITICAL). Every ATLAS-arm container on Modal deterministically hits the
frozen-predictor bug.** `CODE_AUDIT.md` rated B4 "HIGH — ATLAS arms work only if an
AdaJEPA arm ran earlier in the same process," which reads as an ordering hazard. But
`modal_e4.py:108` invokes `run_e4.py --arms <single arm>` in a **fresh subprocess per
container**, so there is *no possibility* of an AdaJEPA arm having run first. The
per-container fan-out that makes Phase 5 affordable simultaneously **guarantees** the
bug fires on every `atlas_fixed`/`atlas_detect`/`atlas` container. B4 is therefore a
hard blocker for Phase 5, not a nice-to-have.

**A-xi. The key-namespace gate on fix B2 is CLEAR.** For `kind="ln_act"`,
`pretrained_state` (`adajepa.py:83-87`, from `state_dict()`) and `param_names`
(`chart.py:198-238`, from `named_parameters()`) come from the same predictor object with
no intervening parametrization; `VisionTransformerPredictorAC` is a plain `nn.Module`
with no custom `state_dict()` and no prefix wrapping. Verified by code read plus a
runnable check that the two key sets coincide exactly for an ordinary module. **The
"reset silently restores nothing" failure mode does not occur, so B2's fix will
work** — the Phase 2 gate on it passes in advance.

**A-xii (NEW). `smoke_e4.py` — the only validator for the E4 path — is substantially
weaker than it looks.** Of eight assertions: **two are structurally vacuous** (`frozen`'s
`library_size==1` tests a hardcoded literal, since `build_arm_state` sets
`library=None` for that arm; `atlas_fixed`'s `probe_outcome != "committed"` tests a
control-flow guarantee, since `expansion_mode="fixed"` reaches neither expansion
branch); **one is near-vacuous and contradicts its own docstring** (the docstring claims
`raw_steps_per_replan` entries are checked as multiples of `frameskip`, but the code
checks only `n >= 0` on a Python list length, which cannot be negative — the divisibility
property is not tested anywhere); **one has a latent vacuous mode** (the AdaJEPA-reset
check filters both sides by the same `param_names`, so a total namespace divergence
would compare two empty dicts and pass — not triggered today, per A-xi). The remaining
four (G5 pairing, UMF finiteness, oracle routing, JSONL key contract) are genuine.
**Fix before Phase 5 relies on it**: implement the divisibility check the docstring
promises, assert non-empty coverage in the reset and oracle checks, and replace the two
structural assertions with dynamic ones.

**A-xiii. `Library` has no `evict()` at all.** `add()` raises `RuntimeError` when full
(`library.py:58-74`); both production callers pre-check `is_full()`, so growth simply
**halts** at `K_max=10` — nothing is ever evicted, retired or replaced, and no
`utilisation()` method exists. The proposal's "the hard cap — not the probe — bounds
library size" is technically satisfied, but `OPUS_REMAINING_TASKS` #14's "K_max pressure
is completely untested" is understated: there is no eviction policy to test. `atlas_detect`
is the arm most likely to reach the cap in Phase 5. **Fix:** log a
`library_full` flag per episode so cap-hits are visible, and state plainly in the paper
that eviction is unimplemented and out of scope.

---

# PART B — The complete finding register

Every item from all seven audit documents, with its fix and owning phase. **Nothing
below is description-only; each row has an action.**

## B1 — Corrupts numbers already on disk → fix + cheap re-run (Phase 1)

| ID | Finding | Fix | Cost |
|---|---|---|---:|
| A1 | Hysteresis margin provably inert at K=2 (`router.py:94-101`); "the fix halved the margin" cannot be true. Affects **N7**, contributes to **N6**. | Normalise by the incumbent's own score, not the batch spread: `(current−best)/current < m`. Keep m=0.05. Re-run all E2 configs. | $1 |
| A2 | `maybe_expand()` picks the incumbent by argmin **on the verification chunk itself** (`expand.py:131-142`) — look-ahead advantage. Affects **N9**. | Select the incumbent as the currently-selected chart, or argmin over the *deficit* chunks. Re-run E2 q=1. | $0.2 |
| A3 | Per-chunk expander log never written, so "3 charts committed" is unreproducible (`RESULTS_AUDIT` §8; `PAPER_FACT_CHECK` A4). | `run_e2.py` logs `strikes`, `probe_outcome`, `relative_gap`, `hysteresis_held`, `committed` per decision. Folded into A1/A2's re-run. Also closes `PAPER_FACT_CHECK` D4 (K=3 hysteresis binding, currently "unverified"). | — |
| A4 | Reported `eval_umf` measured on the set that selected the checkpoint (`run_e0.py:489-493`, consulted up to 80×). **Biases every UMF number in the paper.** | Add `--num-test-trajs`, a third untouched split. **No retraining** — re-score saved charts, forward-only. | $1 |
| A5 | `analyze_n100.py` never computes the CI/McNemar that **N1** cites it for. | Add both computations, regenerate `analysis_n100.json`. | $0 |
| A6 | Oracle−random CI structurally one-sided: `d_i ≥ 0` by construction, so it can never contain a negative (`RESULTS_AUDIT` §7(j-3), `PAPER_FACT_CHECK` A3). | Add `oracle_gap_permutation()` to `stats.py`; permute chart labels within episode. Report effect size + permutation *p*. | $0 |
| A7 | Partial-Kendall p-value anticonservative (Kendall's null ignores the estimated OLS coefficients). | Permutation test on residuals. Point estimates unaffected. | $0 |
| A8 | `sr_by_bucket` silently drops episodes >300px to NaN. | Open-ended top bucket + assert *n* sums to the episode count. | $0 |
| A9 | 20-trajectory chart has **no seed manifest** — held-out status unverifiable, and it is the anchor of **N4** *and* the chart behind **N1** (`REDTEAM` N4 Attack 1). | Retrain with the manifest saved; confirm UMF reproduces. T4. | $1 |
| A10 | R2 trio trained at unrecorded, unmatched budgets; `lora4` documented as once retrained at a reduced budget (`PROPOSAL_CODE_ALIGNMENT` H.3). `full`'s 0.728 is the shape of a budget confound. | **(a)** report descriptively, drop any capacity *ordering* (what `REDTEAM` RQ0 says is the only defensible framing); **(b)** if time: retrain all three R2 kinds at one recorded matched budget. | $0 / $3 |
| **A11** | **NEW** — artifacts record `params=10292640` while the paper says 118,176 (Part A-iii). | Record trainable count; regenerate `results.json` with a supersede note. | $0 |
| A12 | `--table T5` and `--fig S1/S2/S3` silently no-op (Part A-ii). | Wire the dispatches or remove the dead choices. | $0 |
| A13 | `merge_planning_shards.py` does not check contiguity (Part A-i; merging is otherwise clean). | Add a contiguity assertion. | $0 |
| A14 | `ATLAS_SUMMARY.md` states the converged-CEM spread as "3.8–8.3px" unqualified; the real range over all 6 seed/kind cells is **3.77–27.15px** (`ln_act` seeds 1,2 = 17.9, 27.2). `E0_RESULTS.md` correctly hedges "for most seed/kind pairs"; the summary dropped the hedge (`RESULTS_AUDIT` §6). | Restate with the full range; main text must not say "tight cluster". `PAPER_DRAFT.md` Appendix C already has the per-seed table. | $0 |
| A15 | The "+55.6pp pre-fix" Cell B number has **no surviving raw records** — it is L0 (`RESULTS_AUDIT` §2(h)). | Drop the before/after framing entirely. Institute never-reuse-a-directory-name. | $0 |

**Phase 1 total: ~$3.20 (+$3 for A10b).**

## B2 — Blocks the continual run; never-run code, free to fix (Phase 3)

| ID | Finding | Fix |
|---|---|---|
| B1 🔴 | ATLAS arm can **never** commit: `harness_e4.py:216-217` hard-codes `next_encoder_output=None`, so `loop.py:140-150`'s guard is unsatisfiable. **Under the default `expansion_start_library="full"`, correct behaviour and this bug both give "0 commits"** — indistinguishable. | Implement the two-deep chunk buffer the comment at `:198-203` already describes, copying `run_e2.py:296-339` (verified leakage-free, `CODE_AUDIT` §9.3). |
| B2 🔴 | Arm 2 ≡ arm 3: `AdaJEPA.reset()` has no production caller, so plain AdaJEPA never re-initialises and the persistence rung measures zero. Buffer also never cleared, so its window spans regime boundaries. | Call `reset()` per episode for `variant=="adajepa"`, and clear the buffer. **Key-namespace gate now CLEAR (A-xi)** — the fix will take effect; no longer blocked. |
| B3 🔴 | Motion gate: 10th percentile of **whole-trajectory** displacement from **3** trajectories, applied to **1-model-step** chunks at nas=1. If it gates everything, routing freezes, no strikes, arms 4-7 collapse into arm 1 while writing a plausible table. | **Measure first**, then recalibrate at the chunk size actually scored, over ≥30 trajectories. |
| B4 🔴 | **Upgraded from HIGH to CRITICAL (A-x).** Not an ordering hazard — `modal_e4.py` runs one arm per fresh subprocess, so **every** ATLAS-arm container is guaranteed to hit the frozen-predictor crash. Hard blocker for Phase 5. | Re-enable `requires_grad` on the chart surface inside `atlas_refine`/`_fit_candidate`. |
| **B12** 🔴 | **NEW (A-ix)** — `modal_e4.py:109` passes `--seeds 1` always; `seed_run` is only relabelled onto output records afterwards. A multi-seed sweep yields **bit-identical data under different labels**. Invalidates Stretch A as specified. | Thread the real seed through to `--seeds`/`get_stream`/`local_seed`. **Must land before any multi-seed spend.** |
| **B13** 🟠 | **NEW (A-xii)** — `smoke_e4.py`, the only E4 validator, has 2 structurally vacuous assertions, 1 near-vacuous one contradicting its own docstring, and 1 latent-vacuous one. | Implement the promised `frameskip` divisibility check; assert non-empty coverage in the reset and oracle checks; replace the two structural assertions with dynamic ones. **Do this before Phase 5 leans on it.** |
| **B14** 🟢 | **NEW (A-xiii)** — no `evict()` exists; growth halts at `K_max` and nothing is retired. `atlas_detect` is most likely to reach the cap. | Log a `library_full` flag per episode; state in the paper that eviction is unimplemented and out of scope. |
| B5 🟡 | Recall is unpaired: `streams.py:86-87` puts `segment_idx` in the seed key, so first visit and revisit get different init/goal. **RQ4 is unmeasurable as wired.** | Key on regime-visit index so revisits reuse the first visit's episode set. |
| B6 🟡 | `atlas_detect` commits a byte-identical clone with no gradient step — it ties its parent, loses the argmin tie-break, and is inert. Not what the compared methods do. *(Decided: fit it.)* | Call `_fit_candidate()` on deficit chunks before `library.add()`. Makes 5→6 differ by **exactly** "verifies". |
| B7 🟡 | `make_tables.py:126` pairs by equal **length**, not equal key set — with resume, silently mis-paired bootstrap/McNemar. | Intersect on `(arm, seed_run, global_episode_idx)` and assert. |
| B8 🟡 | Arms 4/5/6 skip refinement when `current_idx==0` while 2/3 always refine → the 3→4 rung differs by two mechanisms. | Refine a **clone of c₀** on first selection rather than skipping, holding "adapts" constant. Record the decision. |
| B9 🟡 | Arm 3→4 also differs by buffer size (5 vs 1), violating plan §7.6's explicit "same buffer size" requirement. | Give `atlas_refine` the same 5-chunk buffer. The offline-pre-training difference is *specified* by §7.4 and stays — but must be named in the paper. |
| B10 🟢 | `n_replans_target` unit error (`run_e4.py:154`) — raw-step budget ÷ chunk count. Inert at horizon=6 but the planner never shortens near episode end. | Correct the arithmetic. |
| B11 🟢 | E4 not reproducible on resume — CEM generator seeded once, resume doesn't replay draws. | Re-seed the planner generator per episode from `spec.seed`. |

## B3 — Dead gates and latent hazards (Phase 2)

| ID | Finding | Fix |
|---|---|---|
| C1 🔴 | **Gate G2 asserts nothing** — a literal `if…: pass` printing PASSED unconditionally, on i.i.d. `randn` with no learnable structure. Claim **P-2** and every "G2 passes" statement have zero support. | Rewrite on structured data (a learnable predictor-weight perturbation, as G3a uses) with a real assertion, driven through `_open_loop_rollout`. **Must fail on a deliberately leaked chart.** |
| C2 🟠 | **Gate G5 is a tautology** — it only checks that `paired_seed()` ignores its `arm` arg, which is true by inspection. Builds no env, samples no goal. | Construct two envs at the same seed; assert identical init states and goals. **Must fail on deliberately mismatched seeds.** |
| C3 🟡 | G1 tests only *unrefined* charts (where `restore_()==apply_()` is trivially identical) and never `kind="full"`. | Extend to refined charts and all three kinds. |
| C4 🟠 | `chart.restore_()` does not restore pretrained weights for `ln_act`/`full` — it re-applies the same chart. Falsifies **P-1**'s literal wording; leaves the predictor permanently dirty. 10 call sites depend on current behaviour. | Add explicit `restore_pretrained_()`; rename the existing one to `reapply_`; fix the false comment at `loop.py:247`. C3's gate then covers it. |
| C5 🟠 | `base_env.step()` bypasses the wrapper. Harmless for `PhysicsRegime`, **fatal and silent for `VisualCorruption`** — any corrupted planning run would measure nothing. | Step the wrapper; assert loudly if a corruption wrapper is present and bypassed. |
| C6 🟡 | `lora4` online refinement raises — after parametrization the base name leaves `named_parameters()`, so the optimizer list is **empty** and `Adam([])` raises. E0's offline path handles this; the online path doesn't. | Select by `requires_grad` after enabling `lora_A`/`lora_B`, as `harness.py:118-125` already does. |
| C7 🟡 | `diagnose_cem_costs.py` writes output only at the end — the documented cause of two lost dose-response runs. | Write/append per seed. |
| C8 🟢 | `smoke_gates.py` loads from the **remote** hub while production uses the patched local cache. | Use `source="local"`. |
| C9 🟢 | `configs/atlas/*.yaml` are loaded by nothing, and two disagree with the scripts' real defaults. | Load them, or add a "documentation only" header matching `configs/regimes/pusht.yaml:3-5`. |
| C10 🟡 | E1 harness samples random goals + uses a success criterion requiring the *agent's* position to match — why `e1_reduced_v2` returned **0% for all routers including the oracle**. Prescribed fix never applied. | Port `sample_dataset_init_goal` + `block_success()`. Only needed if E1's harness is revived; Phase 5 routes around it. |

## B4 — Audit items the audit itself never finished (Phase 2)

Straight from `RESULTS_AUDIT.md`'s and `NEXT_ACTIONS.md`'s own "did not get to" lists.

| ID | Open item | Action |
|---|---|---|
| D1 | Step 4 red-flag sweep **never done as a dedicated pass**: exact-zero variance across seeds, the 360-paired-episodes spec check, NaN sweeps beyond the two already found. | Run it as a script over every `atlas_out/` directory; emit a report. |
| D2 | **`CLAIMS_MATRIX.md` Section C's S-1…S-8 are completely unverified** — the rollout-fix claim, full gate status, the "E0 closed" framing, the two-parallel-sessions claim. | Dedicated process-claims pass; S-2 and S-5 are already partly resolved, the rest are not. |
| D3 | `OPUS_REMAINING_TASKS.md` item 1 marked DONE but its own text flags an unchecked internal gap (knock-aways confirmed at N=20, "never re-verified at N=100"). | Re-verify at N=100 from the local JSONL. $0. |
| D4 | Full inventory of the `e0*` training-only directories never characterised one by one. | Enumerate; mark each current / superseded / smoke, so the release bundle is defensible. |
| D5 | Whether `expand.py`'s strike logic explains Cell C's 2 tau-crossings producing 0 commits. | **CLOSED (A-vii)** — the probe *did* fire and rejected on merits. Report Cell C as "fired and correctly declined", the stronger story. |
| D6 | Exact re-simulation of the K=3 hysteresis condition in `e2_confusion_matrix`. | Closed by A3's per-decision logging + re-run. |
| D7 | `ACTION_SAMPLING_REVIEW.md` never independently re-derived. | **CLOSED (A-viii)** — its "reliable by construction" framing is not supported; ~43% single-attempt, ~100% from retry. Disclose the contact-conditioning; record the realised rate in the manifest. `README.md` and `code-review.md` still unchecked — low priority, assign to Phase 2 if time. |
| D8 | `smoke_e4.py`'s assertions never audited — **it is the only validator for the E4 path**. `modal/modal_e4.py` never read — **it is what will launch Phase 5**. | *(Research agent in flight; folds into Phase 3's spec.)* |
| D10 | Tensor-chain safety rests on an unasserted "never batch multiple episodes" invariant (A-vi). | Add the assertion at all three call sites. |
| D9 | 470 KB of AI-authored markdown vs 190 KB of Python; a claim repeated across six docs reads as corroborated when asserted once. | Consolidate: `EVIDENCE_LEDGER.md` becomes the single source of truth for numbers; other docs point at it. |

## B5 — `OPUS_REMAINING_TASKS.md` Section B: 14 un-applied write-up corrections (Phase 6)

Items 9-22, still standing in the source docs. Most are wording, but four have work
attached: **#12** → A10; **#13** → E-E; **#17** → C5; **#18** → A4; and **#21 — build
the dissociation figure**, which Part A-iv shows is free and is the paper's central
claim in one panel. The rest (#9 retract the "strict superset" argument, #10 the
`closed_loop` retraction, #11 the `hybrid` two-variable confound, #14 `K_max` untested,
#15 the R1-vs-R2 note, #16 stop calling E0/E2 "orthogonal", #19 single-env/checkpoint/
seed scope, #20 don't lean on "pre-registered", #22 reframe the invented 15pp bar) are
applied in Phase 6.

---

# PART C — The two scientific questions you raised

## C-1. Is the training action distribution correct? No — on three axes at once.

The default collector (`--data-source dataset`, `run_e0.py:261-277`) replays **recorded
expert demo actions**, recorded under **R0**, **open-loop** under the shifted regime.
Deployment queries the model with CEM candidates: 300 sequences from a zero-mean
unit-variance Gaussian at iteration 0, refined over 30 iterations, **re-drawn cold every
replan** (no warm start — `planner.py:276-277`; `_prev_mean` is written and never read).

| Axis | Training | Deployment query |
|---|---|---|
| Generator | expert human | CEM's own proposal distribution |
| Goal-directed | yes, optimally | no at iter 0; increasingly by iter 30 |
| Physics it was recorded under | **R0** | R2 / R1 |
| Reacts to the shift | no — open-loop replay | n/a (a query, not a rollout) |
| Contact | **conditioned on it** — rejection-sampled until contact occurs (A-viii) | ~80% of candidates, unconditioned |

So the chart is fit to predict *what an expert would have done under unshifted physics*
and asked to rank *what a planner is considering under shifted physics*. The code
reaches this conclusion itself (`run_e0.py:520-528`); `CLAUDE.md` §0.1 calls it an
"open, unresolved concern". **The honest name is the distribution half of objective
mismatch** — the covariate-shift problem on-policy data aggregation exists to solve. A
reviewer who knows model-based RL asks this on the first read and the paper has no
answer today.

Not fatal: the *primary* result (N3, cost-ranking collapse) uses no chart at all — it is
the frozen baseline with R0 as control. This threatens the *secondary* negative (N1).

### E-A — the distribution-matched chart *(Phase 4, ~$6)*

`diagnose_cem_costs.py` already draws CEM's candidate batch under the shifted regime and
**rolls every candidate out for real** (`rollout_true_outcomes()`, resetting to the
identical init state before each). That is exactly (context, action sequence, true
outcome) from the planner's query distribution. It saves costs and true distances but
**not** the encoder outputs.

Add `--save-latents`; collect ~30 seeds × 300 candidates under R2; fit `ln_act` with the
existing `run_e0_finetune` loss; evaluate on held-out seeds against **three** read-outs:
UMF on held-out candidates; **cost-ranking ρ** (the quantity that decides what executes,
and far more sensitive than binary SR — n=20 seeds suffices where SR needs n=100); and
planning SR at N=100 paired against the existing baseline.

- **ρ recovers** → a *positive* contribution: *"predictive-fitness adaptation works, but
  only when fitness is trained and measured on the distribution the planner queries."*
  Turns the paper from a diagnosis into a fix.
- **ρ does not recover** → the strongest alternative explanation for the null is killed
  by direct test; the failure localises to the frozen latent geometry.

### E-B — does UMF measure the right thing? *(Phase 4, ~$0)*

Your instinct is right and there is **already unreported evidence**.
`atlas_out/umf_locality.json` recomputes UMF on the top-*k* **most-moving** DINOv2
tokens — block and pusher, not the ~97% static white background global UMF averages
over. The localized metric **also fails to rank charts by planning competence**, and
`lora4` is the clean counterexample (best moving-token UMF 0.168, worst planning). That
is a *stronger* result than the diagnostic's own hypothesis, and it is unreported.
Extensions: correlate per-candidate UMF against per-candidate cost-rank error on arrays
already on disk; score UMF on the CEM candidate distribution. Both ~$0, and the second
is a go/no-go for E-A.

## C-2. The closed-loop train/eval mismatch — confirmed on all four counts

| | Collection (`closed_loop`) | Evaluation |
|---|---|---|
| CEM samples × iterations | **100 × 10** (`run_e0.py:532,539`) | **300 × 30** |
| `num_act_stepped` | **1**, hardcoded, no flag (`run_e0.py:613`) | **6** |
| Replans/episode | 6 (maximally closed-loop) | **1** (maximally open-loop) |
| Whose policy | the **frozen** predictor's plans, collected once and reused for all three kinds | the chart being evaluated |

A **9× search-budget gap** and the two *opposite extremes* of replan frequency, in the
one collector built to close the train/deploy gap. Trajectories were collected once
outside the per-kind loop (`run_e0.py:642-649` vs `:685`), so no chart's corrections
ever influenced its own training data — a single non-reactive round, not DAgger.
**`E0_RECOVERY_PLAN.md`'s "clean rejection" framing is not supportable**; the experiment
cannot separate "closed-loop replay doesn't help" from "the collector's CEM was too weak
to demonstrate anything."

### E-C — matched-instrument re-test *(Phase 4, ~$9)*
Add `--collect-num-act-stepped`; default the collection budget to the eval budget;
re-collect at 300×30 keeping nas=1; retrain; evaluate at N=100. Leaves exactly **one**
deliberate mismatch — replan frequency — which E-D then measures directly.

### E-D — separating feedback from compute *(Phase 4, ~$12)*
N5's +10pp at nas=2 is confounded: `plan_length` stays pinned at `horizon=6` regardless
of `steps_left`, so each of nas=2's three replans runs a **full** 6-step search — 3× the
compute over the same 30 raw steps. Run three paired arms at N=40:
(1) nas=6, iters=30 *(exists)*; (2) **nas=6, iters=90** — 1 replan, 3× compute *(the
control nobody ran)*; (3) nas=2, iters=30 *(exists at N=20, extend to 40)*.
If (2)≈(1), the nas=2 gain is feedback. If (2)≈(3), it was compute.

### E-E — does the one positive result mean anything? *(Phase 4, ~$4)*
`REDTEAM` N6 Attack 1: E2 scores "correct" by **regime label**, not by which chart plans
better — in direct tension with the title *"Measure Fitness, Don't Infer the Regime"* —
and there is no planner in E2's loop at all. Take E2's collected trajectories and seeds,
run the CEM planner under **each** chart on the same seed, and ask: **is the UMF-argmin
chart also the better-planning chart?** Reuses `run_e0_planning.py`'s validated sampler
and success criterion, sidestepping C10. High agreement upgrades the routing result to
what C1 actually needs; chance agreement is itself a sharp finding — the dissociation
appearing a third time, inside the positive result. **Run before Phase 5**: it de-risks
the $19 stream by confirming the routing signal does anything in a planning loop.

---

# PART D — The phases

Each phase = one delegated **sonnet** sub-agent, one spec file, one exit gate I verify.

### Phase 0 — Setup *(me, ~30 min, $0)*
Copy this plan to repo root as `SUBMISSION_PLAN.md`; pointer from `CLAUDE.md`. Create
`research_audit/FIXLOG.md`, `EVIDENCE_LEDGER.md`, `FIX_SPEC.md`. Write four new sonnet
agent definitions into `.claude/agents/`: `atlas-fixer`, `atlas-runner`,
`atlas-analyst`, `atlas-process-auditor`. *(`paper-drafter`, `paper-fact-checker`,
`code-bughunter`, `results-auditor`, `scientific-redteam` already exist and are reused.)*

### Phase 1 — Tier A fixes + cheap re-runs *(`atlas-fixer`, ~5 h, ~$3.20)*
Register B1 rows A1–A15. **Exit gate:** I re-run the E2 configs and confirm N6/N7/N9
reproduce under the corrected hysteresis and incumbent selection; every superseded
number carries a dated banner; `EVIDENCE_LEDGER.md` maps every paper number → file →
recomputed value.

### Phase 2 — Gates, latent hazards, unfinished audit items *(`atlas-process-auditor`, ~4 h, $0)*
Register B3 (C1–C10) and B4 (D1–D10). **Exit gate:** rewritten G2 **fails** on a
deliberately leaked chart; rewritten G5 **fails** on mismatched seeds; the
`squeeze`-safety invariant is asserted at all three call sites (D10).
*(The key-namespace check that was to gate B2 is already CLEAR — see A-xi.)*

### Phase 3 — E4/continual fixes *(`atlas-fixer`, ~5 h, $0.50)*
Register B2 (B1–B14), now including the three new items the research agents surfaced:
**B12** (Modal's fake multi-seed), **B13** (`smoke_e4.py`'s vacuous assertions), **B14**
(no eviction). **Exit gate:** for B1, B2, B3, B6 I confirm each assertion **fails
before** the fix and passes after; `--arms atlas --episodes 2` runs standalone (B4);
segment 0 ep *i* and segment 2 ep *i* share init/goal (B5); and two containers launched
at `seed_run=0` and `seed_run=1` produce **different** episode data, not relabelled
copies (B12).

### Phase 4 — Defence experiments *(`atlas-runner`, ~9 h wall, ~$31)*
E-B ($0, first, as go/no-go) → E-E ($4) → E-A ($6) → E-D ($12) → E-C ($9).
Every run writes a seed manifest and per-episode records **from the start**, and
downloads artifacts immediately — the archive gap that started this session must not
recur. **Exit gate:** each result recomputed by me from the downloaded raw file.

### Phase 5 — The continual stream *(`atlas-runner`, ~4 h wall, ~$19)*
S2 = `A,B,A,B` (R0/R2), **10 episodes/segment, 1 seed**, nas=1 (6 replans — the only
setting where routing, refinement and next-chunk verification can occur), iterations=10
(a uniform cut across all arms per plan §7.0's cut ladder item 1, recorded as such),
num_samples=300, horizon=6. **40 paired episodes/arm.** Seven arms, one Modal container
each: frozen · adajepa *(now genuinely resets)* · adajepa_persist · atlas_fixed ·
atlas_detect *(candidate now fitted)* · atlas *(verification now reachable)* · oracle_id.

Pre-register the read-out **before looking**, into `research_audit/E4_PREREGISTRATION.md`:
charts committed vs. the true regime count of 2 (RQ3); routing accuracy and SR per
segment; paired first-visit-vs-final-revisit Δ (RQ4, measurable only after B5);
gated-chunk fraction as a validity check on B3.

**Risk control:** `--profile` one arm 2 episodes (after B4) → 2-episode-per-segment
smoke across **all** arms → field-by-field inspection of `gated`, `strikes`,
`probe_outcome`, `charts_committed_cumulative`, `selected_idx`, and that arms 2 and 3
now differ → only then launch. `run_e4.py` flushes per episode and resumes, so a timeout
costs one episode.

**Honest sizing, stated in the paper:** 40 paired episodes/arm supports "the mechanism
runs end to end and here is what it does" and a charts-committed count. It does **not**
power a success-rate comparison between adjacent rungs. Report Δ with CIs and let the
CIs speak.

### Phase 6 — Write-up *(`paper-drafter` → `paper-fact-checker`, parallel, $0)*
All `PAPER_FACT_CHECK` findings; `OPUS_REMAINING_TASKS` §B items 9-22; the dissociation
figure (Part A-iv); strip the nine internal-provenance HTML comments before any LaTeX
conversion; relabel S-dyn as a **dynamics-fingerprint** baseline; state N1 and N3 as one
mechanism seen as cause and consequence; fix the MBCD venue (AAMAS 2021); add Lambert
2020, Grimm 2020, Singh 2026, Vakalis 2026, framing the dissociation as a **replication
in a new substrate**. **Exit gate:** `paper-fact-checker` returns **zero** section-A
findings.

---

# PART E — Budget and order

| Phase | Wall | $ | Cut if short? |
|---|---|---:|---|
| 0 setup | 0.5 h | 0 | no |
| 1 Tier A + re-runs | 5 h | 3.2 | no — makes existing claims honest |
| 2 gates + unfinished audit | 4 h | 0 | no — gates G2/G5 currently assert nothing |
| 4a E-B, E-E | 3 h | 4 | no — defends the only positive result |
| 3 E4 fixes | 5 h | 0.5 | no — blocks Phase 5 |
| 5 continual stream | 4 h | 19 | no — the venue's core gap |
| 4b E-A | 4 h | 6 | keep — the action-distribution defence |
| 4c E-D | 3 h | 12 | drop first if time-bound |
| 4d E-C | 4 h | 9 | drop second |
| 6 write-up | parallel | 0 | no |
| **Total** | **~33 h** | **~$54** | leaves ~$36, ~50 h |

**Stretches, at most two fit:** continual seeds 1-2 (+$38, best buy if Phase 5 is
informative — **but only after B12 lands, otherwise it buys three relabelled copies of
one run**) · E-D at N=100 rather than N=40 (+$18, `REDTEAM`'s own prescription) ·
A10(b) matched-budget R2 trio retrain (+$3).

Phases 0-2 alone reach `REDTEAM`'s recommended zero-new-GPU paper. Phases 0-5 (~$27) add
the continual result and the properly-scoped positive result.

---

# PART F — Not doing

- **Full 7-arm × 20 ep × 6 segments × 3 seeds E4.** Not budget — *validation time*.
  Phase 5 is the same code path at 1/9 the episodes, leaving room to smoke it properly.
- **`full` × R1 training** — dropped by prior decision; restates a known result.
- **τ/q sensitivity sweep**, **E5** — out of scope in the plan's own cut ladder.
- **Reviving `harness.py::run_e1_episode`** — C10's defects returned 0% for every router
  including the oracle. Phase 5 gets routing-with-planner via `run_e4.py`.
- **Touching `score.py::umf`, `stats.py`'s existing functions, `run_e0_planning.py`'s
  loop** — they produced the results on disk. Additions only.

---

# PART G — Verification

- **Phase 1:** every fix followed by its re-run; superseded numbers get dated banners;
  `EVIDENCE_LEDGER.md` maps claim → file → recomputed value, regenerable by script.
- **Phase 2:** G2 must fail on a leaked chart; G5 must fail on mismatched seeds; the
  key-namespace check must return non-zero overlap before B2 is trusted.
- **Phase 3:** B1/B2/B3/B6 assertions confirmed to **fail before** the fix.
- **Phases 4-5:** every result recomputed by me from the downloaded raw file, not from a
  summary or an agent's report; artifacts downloaded immediately after each run.
- **Phase 6:** `paper-fact-checker` returns zero section-A findings.
