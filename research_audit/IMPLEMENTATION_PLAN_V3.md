# ATLAS — Implementation Plan v3

*Written 2026-08-28. Target: NeurIPS 2026 Workshop on **World Models in Physical AI** (Sydney),
5 Sept 2026 AoE, full paper 6–7 pages. Substrate: `dino_wm_pusht` in `jepa-wms`. Push-T only.*

**Status of this document.** This is a **replacement** for `ATLAS_implementation_plan_v2.md`, not a
patch on it. v2 carries two in-place CORRECTION sections (§6.1a regimes, §7.0a planner budget) that
override its own §6.1/§7.0; a third layer would leave no self-consistent reading. v3 is written as a
standalone spec. It does **not** redesign the science: RQ0–RQ4, C1–C4, the SCORE → SELECT → EXPAND →
EXECUTE → REFINE ordering, and the "inherits unchanged" component list from `ATLAS_proposal_v7.md`
are carried through verbatim (§1). What is redesigned is the set of open numeric/protocol choices
that v2 got wrong twice: the planner budget, the regime parameters, the chart-training data
objective, and every threshold.

**Nothing in this session ran code or changed experiment code.** Every empirical number quoted below
is either read directly from a raw file on disk (path given) or taken from a dated audit entry
(citation given). Where I computed something myself this session it is marked *(computed here)*.

---

## §0. Decisions at a glance

A reviewer of this plan should be able to read this section alone and see the reasoning. Each item
states the choice, the reason, and — per the reuse rule — whether it costs new code and what it
invalidates.

### C.1 — Planner budget: `num_act_stepped = 2`, `horizon = 6`, `num_samples = 300`, `iterations` set by a pre-registered calibration gate, `MAX_MPC_STEPS = 30`, `frameskip = 5`

The two configurations already tried both fail, but not for the reason the dissection assumes.
AdaJEPA-literal failed because `horizon=25` means 125 raw steps of model lookahead for a task whose
goals are sampled 30 raw steps apart — a lookahead/task mismatch, not a replan-count problem. The
substrate-validated `nas=6` config does not fail on competence; it fails because executing all six
planned model-steps consumes the whole 30-step episode, so exactly one CEM search happens per
episode and the prequential loop has nothing to run on. The fix is to **decouple these two axes**:
keep `horizon=6` (the lookahead the task was built for) and cut only how much of each plan is
committed before replanning. That is textbook receding-horizon MPC, and it is *substrate-native* —
`jepa-wms`'s own Metaworld config for this same DINO-WM backbone is `horizon=6, num_act_stepped=3,
iterations=15` (`vendor/jepa-wms/configs/evals/simu_env_planning/mw/dino-wm/reach_L2_cem_sourcexp_H6_nas3_...yaml:202–207`),
and RoboCasa's is `H3_nas1`. So `nas < horizon` is not a deviation from the substrate; the Push-T
`nas=6` config is simply the substrate's *open-loop* variant, and DINO-WM's own Table 8 reports
replanning MPC (0.90) **above** execute-the-whole-plan CEM (0.86) on Push-T. `nas=2` is chosen
rather than `nas=1` for a reason that is provable, not empirical: `atlas/score.py::umf` sets
`T = actions.shape[0]`, and the UMF denominator `Σ‖z_k − z_0‖²` is identical for every chart on a
given chunk, so at `T=1` **argmin-UMF is arithmetically identical to argmin-`e1`** — RQ1's own
ablation baseline. `nas=1` would make the paper's central comparison vacuous by construction.
`nas=2` gives `T=2` chunks (UMF stays a multi-step score), three replans per 30-raw-step episode
(the minimum for a complete SCORE → SELECT → REFINE → SCORE cycle within one episode), and leaves
episode length, goal sampling and the success threshold untouched so every E0 number on disk stays
comparable. It is already measured: `atlas_out/e0_planning_nas2/` shows frozen baseline 8/20 vs.
`nas=6`'s 44/100 — no evidence of competence loss — and the chart at +10pp (`ATLAS_SUMMARY.md`
§4.4). The plan's own unresolved suggestion (scale `MAX_MPC_STEPS` instead) is **rejected**: with
`GOAL_TRAJ_LEN = 31` raw steps fixed, lengthening the episode alone inflates success toward ceiling
for every arm and destroys discrimination; lengthening the goal too creates a different, harder task
that is incomparable to everything on disk, costs the same per replan, and still commits 30 raw
steps open-loop per plan — the regime where JEPA-WM says compounding error bites hardest. **Cost:
config only** (`--num-act-stepped 2`; `run_e1.py`'s `N_WARMUP_REPLANS` drops 2 → 1, one constant).
No new code. **Invalidates:** nothing on disk is deleted — the `nas=6` results remain valid *as
measurements of the single-open-loop-plan protocol*, which is precisely how the paper's N1 result
must be scoped. E0's headline RQ0 answer must be re-measured under the new protocol (§8.1).

### C.2 — Regimes: keep `R1 = shape.friction 2.0` and `R2 = space.damping 0.5`; promote **R2 to the primary stream regime**

The dissection's description of the current state is stale: the code is not friction 0.8 / elasticity
0.9. `atlas/regimes.py:55–66` is `R1 = {"friction": 2.0}`, `R2 = {"damping": 0.5}`, and elasticity
was dropped for a *third* mechanical-deadness reason of the same family as mass — `pusht_env.py:687`
hardcodes `space.damping = 0`, which annihilates restitution velocity before it can displace
anything (`E0_RECOVERY_PLAN.md` §0.3). Unlike every earlier regime choice, the current pair was
actually calibrated against a stated criterion (`E0_RECOVERY_PLAN.md` §0.4: UMF-ratio under shift,
with saturation checked — friction 2.0 vs 5.0 measured identical at 1.67× vs 1.66×, so 2.0 is the
axis ceiling; damping 0.5 at 2.47×, with 0.9 held in reserve at 3.18×). I am not redesigning a part
that was, for once, derived. The one change is **which regime carries the S2 stream**. v2 §6.1 says
S2 uses R0 and R1. R2 (damping) is the better choice for RQ2 and RQ4 for three measured reasons:
its baseline-SR drop is −40pp vs friction's −20pp; it has a characterised, *directional* failure
mechanism (the predictor was trained entirely at `damping=0`, so the block glides past the goal —
systematic overshoot, `E0_RECOVERY_PLAN.md` §0.5), which is exactly the kind of deficit a
regime-specific adapter can absorb; and it is the only regime with a **dose-response ladder already
on disk** (`atlas_out/cost_ranking_dose_0125|025|0375`), which is what makes the N3 mechanism result
defensible. Friction is retained as E0's second capacity regime and as E2's cell-B replicate, so the
"two regimes, because gain-like and contact-like shifts may need different capacity" argument in the
proposal survives intact — and the two axes are now genuinely different *mechanisms* (tangential
contact vs. post-contact glide), which is stronger than the original mass-vs-damping pair ever was.
**Cost: config only.** `scripts/run_e4.py`'s `--segment-regimes` already defaults to `R0 R2`.
**Invalidates:** nothing.

### C.3 — Chart-training data: **on-policy CEM rollouts toward real dataset goals under the shifted regime**, with the contact rejection filter removed

The failure this replaces is a proxy substitution: `scripts/run_e0.py:312–352`'s `scripted`/`hybrid`
generator aims the agent at `block_xy + U(−40, 40)` — a random point near the block's *initial*
position, with no goal anywhere in the loop — and accepts a trajectory iff `total_contacts > 0`. The
objective that was hard to satisfy (goal-directed pushing) was swapped for one that is cheap to
satisfy (touching the block), and the arms were then judged against the cheap one. The replacement
must satisfy both halves of the brief, and "goal-directed" has a stronger reading available than
"aims at the goal": **the training distribution should be the deployment distribution.** So the
collector is the deployment loop itself — a real `GC_Agent`/CEM planner, planning against the frozen
`c0` predictor, toward a real `sample_dataset_init_goal` pair, executed under the shifted regime, at
the *same* planner config as evaluation. Goal-directedness is then not a heuristic that approximates
the objective; it *is* the objective, because the planner's cost function is the goal. Regime
exercise is not rejection-sampled either; it follows structurally, because reducing the goal cost
requires pushing the block, and the shift is defined on exactly the contact and post-contact
dynamics that pushing produces. That lets the `total_contacts > 0` acceptance filter be **deleted**,
which removes the contact-conditioning train/deploy mismatch (`SUBMISSION_PLAN.md` A-viii) that
survived even in the `dataset` collector — a second instance of the same proxy pattern that CHECK 2
only caught in the `scripted` arm. The residual honest caveat, stated rather than papered over: this
data is on-policy for `c0` and off-policy for the trained chart, so a chart changes the distribution
it will later be scored on. That is inherent to any single-shot offline chart, it is the exact
objection E5's cross-policy matrix exists to measure, and it is what the online REFINE step in the
deployed loop is for. **Cost: a targeted edit, not a rewrite.** The `closed_loop` branch of
`load_regime_trajectories` (`run_e0.py:278–311`) already does this; it needs (i) its CEM config
raised from 100×10 to the canonical eval config, (ii) `--collect-num-act-stepped` made functional
(it is documented as a no-op at `run_e0.py:553–573`), (iii) the contact filter at line 351 removed
for this source, and it becomes the default. **Invalidates:** the `scripted` and `hybrid` chart
artifacts and every conclusion drawn from them (already flagged for retraction by CHECK 2). The
`dataset` (expert-replay) charts stay valid and are **retained as a named comparison arm**, because
"on-policy collection vs. expert replay" is now a clean one-variable contrast — which is what
`OPUS_REMAINING_TASKS.md` #11 said `hybrid` failed to be.

### C.4 — Thresholds: every value gets a stated derivation or an explicit "round default + sensitivity"; `m`'s normaliser is pinned; **pure argmin (`m = 0`) becomes the headline router**

CHECK 3 found 5 of 7 values with no stated reason, the pre-registered sensitivity sweep cut, and
`m = 0.05` doing active damage under a normaliser it was never chosen for. §6 derives each one. The
two decisions worth flagging here: first, **τ is defined as the 95th percentile of `UMF(c₀)` on
in-regime chunks**, measured in Phase 0 from data largely already on disk — which makes a strike mean
"worse than the frozen model is 95% of the time in its own regime", i.e. a 5% per-chunk false-strike
rate *by construction*. That in turn makes `q` derivable rather than assumed: the smallest `q` with
expected spurious probes < 1 over the stream is `q = 3` (§6.2), so the round-number default turns
out to be the principled choice, which is worth saying plainly. Second, `m`'s normaliser: both live
implementations are degenerate (spread-normalised is provably inert at K=2; incumbent-normalised is
over-sticky and sign-broken for `sdyn`, whose scores are ≤ 0), and the audit shows hysteresis is the
*sole* cause of the E2 routing reversal. Rather than pick a third normaliser and hope, the plan
**removes hysteresis from the headline result**: the primary router is pure argmin (`m = 0`), and
hysteresis is reported as a sensitivity row over `m ∈ {0, 0.05, 0.25}` under one pinned, non-degenerate
normaliser (a *fixed, pre-measured per-router score scale*, §6.3 — sign-safe, scale-safe, and never a
function of the current chunk). This deletes an unjustified load-bearing parameter from the paper's
central claim instead of defending it. **Cost:** τ, q, `n_probe`, `K_max` and the motion gate are
config values fed by Phase-0 measurements (no new code); the hysteresis normaliser is a ~10-line
change in `atlas/router.py` plus one calibration constant per router. **Invalidates:** every routing
number produced under either existing normaliser — which the audit already treats as unresolved, so
this settles the conflict by re-running rather than by adjudicating it.

### Reuse ledger

| Decision | Config-only? | If code: size | Existing `atlas_out/` still valid |
|---|---|---|---|
| C.1 `nas=2` | Yes (+1 constant in `run_e1.py`) | — | All `nas=6` results, re-scoped as "single open-loop plan" |
| C.2 regimes | Yes | — | All |
| C.3 collector | No | ~3 targeted edits in one existing branch of `run_e0.py` | `dataset` charts (kept as comparison arm); `scripted`/`hybrid` retracted |
| C.4 thresholds | Mostly | ~10 lines in `router.py` | Non-routing results unaffected; all routing numbers re-run |

Total new code: **no new module, no structural rewrite.** `atlas/score.py::umf`, `atlas/chart.py`,
`atlas/stats.py` and `atlas/loop.py`'s ordering are untouched, per the audit's finding that they are
correct.

---

## §1. The fixed frame (not redesigned — restated so this document is self-contained)

**Research questions.** RQ0 adapter capacity · RQ1 fitness routing vs. dynamics fingerprint · RQ2
dynamics vs. appearance · RQ3 verification-gated expansion vs. detect-and-spawn · RQ4 persistent-library
recall on A→B→A. Exactly as `ATLAS_proposal_v7.md` §5.

**Contributions.** C1 predictive-fitness routing · C2 verification-gated expansion · C3 UMF as a
normalised score · C4 the Deployment Stream protocol. Exactly as §2 of the proposal.

**The loop, in this order, every replan:**

```
1. SCORE   UMF(c; Q) for all c in the library, on the newest executed chunk Q
2. SELECT  c* = argmin UMF            (hysteresis is a sensitivity row, not the default — §6.3)
3. EXPAND  if UMF(c*) > τ for q consecutive informative checks:
              fit a candidate on the DEFICIT chunks;
              commit only if it beats both τ and c* on the NEXT unseen chunk
4. EXECUTE plan with c* (CEM unchanged)
5. REFINE  1 SGD step on c*, AdaJEPA's loss — strictly AFTER scoring
```

REFINE-after-SCORE is load-bearing for the "not cheating" argument and is what G2 tests.

**Inherited unchanged:** frozen DINOv2 encoder + ViT predictor (`dino_wm_pusht`, plain LN); L2 latent
distance as the prediction metric; the CEM planner *architecture*; adapters as the adaptation
surface; the expansion structure trigger → instantiate → prune. Specific hyperparameter *values* are
open and are set in §3–§6.

**Cross-arm structural rules (Part B — non-negotiable):** every arm runs the identical planner
configuration; charts are disjoint parameter sets; every chart is identity-initialised; no chart is
ever scored on data it just refined on.

---

## §2. The unit of currency

Every design choice below is priced in one unit: **a CEM replan**. From the `wall_time` and
`replans` fields of 280 episode records across `atlas_out/e0_planning_n100`, `e0_planning_nas2`,
`e0_v3_planning_dataset_baseline` and `e0_v4_planning_lora4` *(computed here, least-squares)*:

```
wall_time  =  2.4 s  +  146.5 s x (number of replans)          (n = 280)
```

| replans/ep | n | median wall | ← this is a clean linear law with ~zero intercept |
|---:|---:|---:|---|
| 1 | 253 | 148.5 s | |
| 2 | 3 | 289.6 s | |
| 3 | 24 | 440.4 s | |

So at `300 samples x 30 iterations x horizon 6` on an L4 ($0.80/GPU-h, `SUBMISSION_PLAN.md`):

> **1 replan = 146.5 s = $0.0326.** The entire ~$90 budget is **~2,760 replans** at `iterations=30`,
> **~5,500** at `iterations=15`, **~8,200** at `iterations=10` (cost is linear in `iterations`).

Two consequences the old plan missed. First, v2 §12's 21 GPU-h estimate for E4 assumed 30 s/episode;
the real figure is 146.5 s *per replan*, so v2's compute budget is wrong by roughly 5x and the
7-arm x 6-segment x 20-episode x 3-seed E4 as specified costs ~$190 at `iterations=30` — it was never
affordable. Second, **the number of replans per episode is the same quantity that decides whether
the mechanisms can run at all.** C.1 is therefore not a free choice; it is a purchase, and §3 prices
it.

---

## §3. C.1 — the planner budget, derived

### 3.1 What each constraint actually requires

| Constraint | Requirement | Binding on |
|---|---|---|
| Lookahead matches the task | goals are sampled `GOAL_TRAJ_LEN = 5*6+1 = 31` raw steps apart (`run_e0_planning.py:72`) | `horizon` |
| Baseline must be competent | frozen SR must stay well off 0 and off ceiling | `horizon`, `iterations`, `num_samples` |
| Loop must run within an episode | ≥ 3 replans: score chunk₁ → route → refine → score chunk₂ | `num_act_stepped` |
| UMF must not collapse into `e1` | scored chunk must have `T ≥ 2` model steps | `num_act_stepped` |
| Must fit ~2,760–8,200 replans | total replans across all experiments | everything |

The third and fourth rows are the ones v2 never separated from the second, which is why both of its
attempts failed.

### 3.2 The `T = 1` collapse — why `nas = 1` is excluded on scientific, not budget, grounds

`atlas/score.py:78` sets `T = actions.shape[0]`; the UMF denominator (`score.py:107`) is
`Σ_k ‖z_k − z_0‖²`, a function of the *observations only*. It is therefore identical across every
chart scored on a given chunk. `atlas/router.py::_e1_score:167` returns `‖ẑ_0 − z_1‖²`. Hence at
`T = 1`:

```
argmin_c UMF(c) = argmin_c ‖ẑ_1 − z_1‖² / ‖z_1 − z_0‖²  =  argmin_c ‖ẑ_1 − z_1‖²  =  argmin_c e1(c)
```

UMF and its own pre-registered ablation baseline become **the same router**. RQ1 asks whether a
multi-step normalised fitness score beats a one-step error; at `nas=1` that question has no content.
`nas ≥ 2` is a hard requirement, independent of cost. *(This generalises — see §7.1.)*

### 3.3 Options priced

| Option | replans/ep | chunk `T` | Baseline SR | Task changed? | Verdict |
|---|---:|---:|---|---|---|
| **A** AdaJEPA-literal: `H=25, nas=5, it=10, N=200` | ~6 | 5 | **0/1** (measured, v2 §7.0a) | no | Rejected. 125 raw steps of lookahead for a 30-step task |
| **B** Substrate Push-T: `H=6, nas=6, it=30, N=300` | **1** | 6 | 44/100 (`e0_planning_n100`) | no | Rejected for E1/E3/E4: loop cannot run. Retained as the "single open-loop plan" protocol for the N1 result |
| **C** v2's own suggestion: `nas=6`, `MAX_MPC_STEPS = 120` | 4 | 6 | untested | **yes** | Rejected — §3.4 |
| **D** `nas=1`, `MAX_MPC_STEPS = 30` | 6 | **1** | untested | no | Rejected — §3.2 (UMF ≡ `e1`), and 2.6x option E's cost |
| **E → chosen** `H=6, nas=2, N=300`, `MAX=30` | **3** | **2** | **8/20** (`e0_planning_nas2`) | no | Adopted |
| F | `nas=3` (Metaworld's H/2 ratio) | 2 | 3 | untested | no | Runner-up. 2 replans allows one routed decision but no chunk scored *under* the routed chart — the REFINE→SCORE half of the loop still never runs |

### 3.4 Why lengthening the episode (option C) is worse than it looks

`MAX_MPC_STEPS = 30` and `GOAL_TRAJ_LEN = 31` are matched by construction: the agent gets exactly as
many raw steps as the expert demonstration took to move the block between the sampled init and goal.
Raising `MAX_MPC_STEPS` alone breaks that match in the direction that *inflates* success for every
arm — a ceiling effect that compresses precisely the range in which arms must differ, and it makes
every E0 number on disk incomparable. Raising `GOAL_TRAJ_LEN` in step restores the match but defines
a different, longer-horizon pushing task that the substrate never validated, that no result on disk
speaks to, and that costs the same per replan. And in both variants each replan still commits six
model-steps open-loop — the compounding-error regime JEPA-WM explicitly warns about — so the extra
replans buy less closed-loop correction per unit of compute than option E does.

### 3.5 `iterations` — a pre-registered calibration gate rather than an assumed value

Phase 5 of `SUBMISSION_PLAN.md` proposed `iterations=10` to afford the replan count, and the audit
correctly flagged this as re-introducing exactly the §7.0a class of unvalidated deviation. But it is
not an unvalidated axis in the substrate: `jepa-wms`'s own Metaworld/DINO-WM config uses
`iterations=15`. The plan therefore neither assumes 30 nor assumes 10 — it **measures**:

> **GATE P0-C (pre-registered).** Run the frozen baseline at `nas=2, horizon=6, N=300` and
> `iterations ∈ {30, 15, 10}`, n = 20 paired episodes each on identical seeds, under R2. Adopt the
> **smallest** `iterations` whose success rate lies within the paired bootstrap CI of `iterations=30`.
> Log all three rows in the paper's appendix regardless of outcome. If none of 15 or 10 qualifies,
> `iterations=30` stands and §11's experiment sizes are cut instead (in the order given in §11.3) —
> **the planner budget is never traded for statistical power without this measurement.**

Cost of the gate itself: 3 x 20 x 2.3 ≈ 138 replans ≈ **$2.30** at worst. This is the check §7.0a was
written to enforce and that Phase 5 skipped.

### 3.6 The adopted configuration

```
frameskip        = 5          (substrate)
horizon          = 6          (substrate; matched to GOAL_TRAJ_LEN = 31 raw steps)
num_samples      = 300        (substrate)
num_elites       = 10         (substrate)
var_scale        = 1.0        (substrate)
num_act_stepped  = 2          (v3 — §3.2, §3.3)
iterations       = set by GATE P0-C, default 30
MAX_MPC_STEPS    = 30         (unchanged; keeps GOAL_TRAJ_LEN match)
GOAL_TRAJ_LEN    = 31         (unchanged)
=> 3 replans/episode max, 2.2-2.35 mean (early success terminates), T = 2 model-steps per scored chunk
=> E1's N_WARMUP_REPLANS: 2 -> 1  (replan 1 has no prior chunk to score; routing begins at replan 2)
```

**Identical for every arm**, per Part B. This is one `--num-act-stepped 2` flag and one constant in
`run_e1.py`.

---

## §4. C.2 — regimes

| Regime | Parameter | Value | Mechanism | Measured effect |
|---|---|---|---|---|
| **R0** | — | shipped (`friction=0`, `elasticity=0`, `damping=0` everywhere) | — | reference |
| **R1** | `shape.friction`, agent **and** block shapes | **2.0** | tangential contact impulse; changes the rotation/translation split of a push | UMF ratio 1.67x; frozen SR −20pp |
| **R2** | `space.damping` | **0.5** | post-contact glide; the block keeps travelling after the pusher leaves | UMF ratio 2.47x; frozen SR −40pp; systematic goal **overshoot** |

Sources: `atlas/regimes.py:55–66`; calibration table `E0_RECOVERY_PLAN.md` §0.4–§0.5
— **that file was deleted at commit f65f95a; recover with `git show f65f95a^:E0_RECOVERY_PLAN.md`**
(FABLE5_VALIDATION.md §1.3). The regime-calibration numbers it holds (frozen R0 86.7% vs
R2 46.7%, n=15, 2026-08-25 protocol; contact 38.5→13.3/episode) are cited from that git object.

**Three dead axes, do not revisit.** `body.mass`/`body.moment` — cancels exactly against a kinematic
pusher at any scale (`REGIME_DESIGN_REVIEW.md`, four independent confirmations). `shape.elasticity` —
`pusht_env.py:687` hardcodes `space.damping = 0`, which annihilates restitution velocity before it
displaces anything. Block **geometry** — works, but violates matched appearance, which is the entire
reason dynamics shifts were chosen over visual ones.

**Assignment.** S2 stream (E3/E4) and E2's decisive cell B use **R0 ↔ R2**. R1 is E0's second
capacity regime and E2's cell-B replicate. Rationale in §0/C.2: R2 has the larger effect, a
directional and nameable failure mode (overshoot — the kind of deficit a small adapter can plausibly
absorb), and the only dose-response ladder on disk. Retaining R1 preserves the proposal's
two-mechanism capacity argument, and the two axes are now genuinely different mechanisms rather than
two flavours of the same one.

**Does this serve RQ2?** RQ2 contrasts a dynamics shift against an appearance shift. Both R1 and R2
are appearance-neutral by construction (no pixel of the render changes), so the contrast is clean on
the dynamics side; the appearance side remains AdaJEPA's ported visual corruptions (`regimes.py`,
E2 only). No better parameter was found: the two remaining live candidates from
`REGIME_DESIGN_REVIEW.md` — agent PD gains `k_p`/`k_v`, and `action_scale` — are unverified, and
`action_scale` additionally confounds "the world model mispredicts" with "the planner's action space
moved", which is a different phenomenon than the one RQ2 is about.

**Open, deliberately not closed here (§15-2):** the contact-count collapse under damping 0.5 (38.5 →
13.3 per episode) is real but confounded, because it was measured under open-loop expert replay
(`E0_RECOVERY_PLAN.md` §0.4 — deleted, `git show f65f95a^:E0_RECOVERY_PLAN.md`). The C.3 collector
re-measures it on-policy for free, and that measurement is the check — if the planner's contact count
under R2 stays near R0's while replay's collapsed, the collapse was a collector artifact. If it does
not, `damping=0.1` (the milder rung, already characterised) is the pre-registered fallback.

**B-2 DECISION, 2026-08-31 (FABLE5 Day 1.8 — recorded, not deferred further):** the on-policy P0-G
collector *did* re-measure contact under R2 = **4.5 contacts/trajectory** (`phase0_v3/p0g_onpolicy/`
guard / `chunks_R2.jsonl`), well below R0's on-policy rate and *below* the confounded replay figure
that triggered §15-2's concern — i.e. the collapse is **not** a pure collector artifact; it is real
on-policy. Per the pre-registered rule the fallback `damping=0.1` should have been triggered. **It
was not, and will not be** — rationale: (a) FABLE5 retired the routing/stream experiments that
`damping` severity was being tuned *for*; (b) the C-2 result and the 1.1-R settle-check are both
*about* R2's overshoot/glide behaviour, which `damping=0.5` produces and `0.1` would mute; (c)
changing the regime now would orphan every C-1/C-2/settle number on disk. **The paper's limitations
section must disclose:** R2 `damping=0.5` is a global extrapolation shift (all 18,685 training demos
at `damping=0`), the on-policy contact rate under it is ~4.5/traj (low), the pre-registered
contact-collapse fallback was measured as triggered and deliberately not applied, and the chart
trained on that low-contact data plausibly feeds the compression/inaction behaviour reported.

---

## §5. C.3 — the chart-training data objective

### 5.1 The requirement, stated before the design

A chart-training trajectory set must be (a) **goal-directed** — the generating objective is reaching
the actual target, not a proxy for it; and (b) **regime-exercising** — the shifted parameter must
materially shape the transitions, reliably, without that reliability being purchased by a filter.
"High contact rate" satisfies neither on its own: it is the proxy that was substituted in, and it is
what `total_contacts > 0` rejection-sampling manufactures.

### 5.2 The design

> **Collector `onpolicy` (new default).** For each training trajectory: draw a real
> `sample_dataset_init_goal` pair (same sampler, same filters as evaluation); reset a
> `PhysicsRegime(base_env, R)`-wrapped env to the init state; run the real `GC_Agent`/CEM planner
> against the **frozen `c₀` predictor** toward that goal, at the **canonical evaluation planner
> config** (§3.6); record the executed `(observation, action, proprio)` chunks. **No acceptance
> filter.**

- **(a) Goal-directed by identity, not by approximation.** The generating objective is the planner's
  own L2-latent goal cost. There is no substitute objective to drift away from.
- **(b) Regime-exercising structurally.** Reducing that cost requires displacing the block, which
  requires contact and post-contact motion — precisely the dynamics R1 and R2 modify. Contact is a
  *consequence* of the objective rather than a *filter* on the output, so the training distribution
  is not conditioned on it.
- **(c) It is the deployment distribution.** The chunks a chart is trained on have the same
  provenance as the chunks it will be scored and refined on at deployment, which removes the
  train/deploy mismatch that `SUBMISSION_PLAN.md` A-viii and CHECK 2 both flag (CEM's candidate
  distribution is ~80% contact; the old training distribution was 100% by filter).

**Retained comparison arm.** The existing `dataset` collector (expert-demo action replay) stays, as
a named one-variable contrast: *on-policy planner rollouts vs. expert replay*, same goals, same
regime, same budget. This is the controlled version of the test `hybrid` was supposed to be and
wasn't (`OPUS_REMAINING_TASKS.md` #11). `scripted` and `hybrid` are **retired**, and any conclusion
resting on them is retracted in the write-up.

**The honest caveat, stated in the paper, not buried.** `onpolicy` data is on-policy for `c₀` and
off-policy for the chart trained on it — a chart shifts the distribution it will later be scored on.
This is inherent to any offline-fitted adapter, it is exactly what E5's cross-policy matrix measures,
and it is what the online REFINE step exists to correct. It is a limitation to report, not a defect
to hide.

**Implementation.** Three targeted edits inside the existing `closed_loop` branch of
`run_e0.py::load_regime_trajectories` (lines 278–311): raise its CEM config from 100x10 to §3.6's;
make `--collect-num-act-stepped` functional (currently documented as a no-op, lines 553–573); drop
the `total_contacts > 0` acceptance at line 351 for this source. Rename to `onpolicy` and make it the
default. **No new module.**

**Cost.** 100 trajectories x ~2.3 replans x $0.0326 ≈ **$7.50 per regime** at `iterations=30`, less
under GATE P0-C. Collected once per regime and shared across all chart kinds.

---

## §6. C.4 — thresholds, each with a derivation or an explicit default

Everything below is measured in **Phase 0** (§11.1) before any experiment spends budget. Every
measurement is forward-only (no planner) except where noted, and several can be read off chunk
scores already on disk in `atlas_out/e2_charts` and the E0 sweep directories.

| Symbol | Value | Status | Derivation |
|---|---|---|---|
| `τ` | **measured** (≈0.5 expected) | **DERIVED** | §6.1 |
| `q` | **3** | **DERIVED** (from τ) | §6.2 |
| `m` | **0** headline; `{0, 0.05, 0.25}` sensitivity | **DEFAULT + PRE-REGISTERED SENSITIVITY**; normaliser now pinned | §6.3 |
| `n_probe` | **20** | **DERIVED** | §6.4 |
| `K_max` | **5** | **DEFAULT**, explicitly a safety cap, not a mechanism | §6.5 |
| motion gate | **measured** | **DERIVED** (replaces "10th percentile") | §6.6 |
| success threshold | 20 px, π/9 | **INHERITED**, cited | §6.7 |
| `min_block_pos_diff` | 40 px | **DERIVED** (= 2 x success tolerance) | §6.7 |
| `max_agent_block_dist` | 160 px | **DERIVED**, empirically | §6.7 |
| chart lr | 5e-4 | **CITED** (AdaJEPA) | — |
| refine steps | 1 | **CITED** (AdaJEPA) | — |

### 6.1 `τ` — the adequacy threshold

*Principle:* a strike should mean "the library predicts this chunk worse than the frozen model
predicts chunks in the regime it was built for."

> **τ = P95 of `UMF(c₀; Q)` over informative chunks collected under R0 by the `onpolicy` collector.**

This fixes the per-chunk **false-strike rate at 5% by construction** under no shift. Report the
measured value; if it lands near 0.5, say so — that is a validation of the old default, not a
coincidence to hide. Measurement is forward-only over existing or newly collected R0 chunks (~$0).

### 6.2 `q` — strikes to arm the probe

*Principle:* choose the smallest `q` such that the expected number of **spurious** probes across the
entire stream is below 1.

With τ's construction giving per-chunk false-strike rate `p = 0.05`, and `N_chunks ≈ 1,100` scored
chunks across a full E4 stream (§11.2):

```
p^q * N_chunks < 1   ->   0.05^q * 1100 < 1   ->   q >= ln(1100)/ln(20) = 2.34   ->   q = 3
```

Under a genuine shift, if the per-chunk strike rate is ≥ 0.8 (measured in GATE P0-D), the probe arms
within ~4 chunks ≈ 2 episodes — fast enough for a 20-episode segment. **This resolves the CHECK 3
finding that `q=3` "almost never fires":** it never fired because it was paired with a τ whose
matched-regime strike rate was 15.7%, not 5%. Tying `q` to τ's false-alarm rate fixes both at once.
If P0-D measures a shifted-regime strike rate below 0.5, `q` is recomputed from the *measured* rates
and the recomputation is reported — not silently retained.

### 6.3 `m` and its normaliser — the load-bearing fix

Both live normalisers are degenerate. `(current − best)/spread` is identically 1.0 at K=2 (hysteresis
never holds — FIXLOG A1). `(current − best)/current` is sign-broken for `sdyn`, whose scores are ≤ 0,
and over-sticky in general; the audit attributes the entire E2 routing reversal to it. Any normaliser
computed *from the current chunk's own scores* has this failure mode, because at small K the score
set carries no scale information.

> **Pinned normaliser: a fixed, pre-measured, per-router score scale.** Switch from incumbent `i` to
> challenger `j` iff `s_i − s_j > m · σ_r`, where `σ_r` is the interquartile range of router `r`'s
> scores over the Phase-0 calibration chunk set — a constant, logged once per router, never a
> function of the current chunk.

Sign-safe (works for negative-valued `sdyn`), scale-safe (each router carries its own units),
non-degenerate at any K. **And `m`'s value is removed from the headline:** the primary router is pure
argmin, `m = 0`. `m ∈ {0, 0.05, 0.25}` is a pre-registered three-row sensitivity table in the
supplement. Rationale: `m` has no derivation available that isn't circular, it is the single
parameter the audit shows flipping the project's one positive result, and a paper is stronger for
reporting the parameter-free version as the result and the parameter as a robustness check. **This
also settles the `CLAUDE.md` §1.7 ⚠ open item** — a normaliser is now pinned, and every routing number
is re-run under it.

### 6.4 `n_probe` — probe fitting steps

*Principle:* the candidate must see the deficit buffer enough times to fit a real deficit, and few
enough to not memorise a 4-chunk set.

Deficit buffer = `q` strike chunks + the arming chunk = 4 chunks, batch size 1. **Five passes over
the buffer → `n_probe = 4 x 5 = 20`.** The existing value, now with a stated rule that scales
automatically if `q` changes.

### 6.5 `K_max` — library cap

*Principle:* a runaway-growth guard, not a mechanism. **`K_max = 2 x (regimes the stream presents) + 1
= 5`** for S2. Lowered from 10 so that it would actually engage if detect-only over-committed — at
10 it could never bind (max observed library size across the project is 3). Reported honestly as a
safety cap that never fired, alongside the max observed size. Note there is no `evict()` in
`atlas/library.py` (`SUBMISSION_PLAN.md` A-xiii): growth halts at the cap. Stated as a limitation;
not implemented, since nothing in this plan approaches the cap.

### 6.6 The motion gate — replacing "10th percentile of training displacement"

The old rule has no justification for "10th", and B3 showed it was additionally calibrated on a
different chunk size than it was applied to, gating out 100% of chunks. B3 fixed the chunk-size bug;
the percentile is still arbitrary.

*Principle:* gate a chunk when its observed latent displacement is not distinguishable from the
displacement produced by the **agent moving while the block does not** — i.e. when the chunk carries
no dynamics information about the thing the regime modifies.

> **Motion gate = P95 of `‖z_T − z_0‖_F` over chunks in which the block's pixel displacement is
> < 1 px.** Measured at exactly the chunk size it will be applied to (`T = num_act_stepped = 2`),
> over ≥ 30 trajectories per regime.

Forward-only, ~$0. Report both this value and the old 10th-percentile value for continuity. **Encode
B3's lesson as an invariant, not a fix:** the calibration call must derive its trajectory length from
`frameskip * num_act_stepped` rather than a literal, so the gate can never again be calibrated at a
granularity it is not applied at. Assert it in G6.

### 6.7 Task thresholds

- **Success: `block_pos_diff < 20 px` and `angle_diff < π/9`.** Inherited from the substrate's own
  `eval_state`, with the documented correction that the agent-position term is dropped because our
  goals are sampled independently of the agent (`run_e0_planning.py:210–228`). Cite the substrate;
  claim nothing.
- **`min_block_pos_diff = 40 px` = 2 x the success tolerance.** A pair whose block has moved less than
  twice the tolerance could be satisfied without a real push; at 2x it cannot. This derives the
  existing value.
- **`max_agent_block_dist = 160 px`.** Derived empirically in `run_e0_planning.py:150–154`: the
  largest reachable init distance observed was 150.48 px, the confirmed-unreachable one 184.37 px;
  160 sits in the gap. Keep the derivation in the paper's appendix — it is a real one.

---

## §7. Generalising from the two named examples

The brief's rule: each named example is one instance of a class. Both classes recur.

### 7.1 Class: *a parameter chosen for one purpose silently renders a different mechanism inert*

`nas=6` making the prequential loop unrunnable is one instance. Others found in this codebase:

| Instance | Mechanism made inert | Status |
|---|---|---|
| `nas=6` + `MAX_MPC_STEPS=30` | the whole SCORE/SELECT/REFINE loop (1 replan/ep) | **fixed by C.1** |
| `nas=1` (the proposed fix!) | RQ1's UMF-vs-`e1` contrast (§3.2 — `T=1` collapse) | **avoided by C.1** |
| `m=0.05` + spread normaliser | hysteresis identically inert at K=2 (FIXLOG A1) | **fixed by C.4** |
| `m=0.05` + incumbent normaliser | routing itself — everything sticks to the incumbent | **fixed by C.4** |
| motion gate at 2-step calibration, 1-step application | UMF, router and Expander all (100% gated) | fixed by B3; made structural in §6.6 |
| `q=3` at a 15.7% matched-regime strike rate | expansion (~0.4% fire probability) | **fixed by C.4** (τ/q now coupled) |
| `K_max=10` vs. max observed library size 3 | the cap (never binds) | **acknowledged**, lowered to 5 |
| `--collect-num-act-stepped` | documented no-op | **fixed by C.3** |

The pattern is general enough to deserve a standing check rather than a case-by-case fix:

> **GATE G7 — mechanism liveness (new).** For every mechanism gated by a threshold — the motion gate,
> the router's switch rule, the strike counter, the probe, the library cap — measure its **firing
> rate on real data at the production config** before spending budget on the full run, and require
> that rate to lie inside a stated non-degenerate band (not ~0%, not ~100%). A mechanism whose firing
> rate is degenerate is reported as such and is **not** described in the paper as having run.

This is the gate whose absence let four separate mechanisms sit dead while producing plausible tables.

### 7.2 Class: *an objective that is hard to satisfy is replaced by a proxy, and the work is then validated against the proxy*

The `scripted`/`hybrid` collector is one instance. Others:

| Instance | Real objective | Substituted proxy | Status |
|---|---|---|---|
| `scripted`/`hybrid` collector | push the block to the goal | touch the block (`total_contacts > 0`) | **fixed by C.3** |
| Acceptance filter in **all** collectors, incl. `dataset` | represent the deployment distribution | contact-conditioned distribution | **fixed by C.3** (filter deleted) |
| E2's routing "correctness" | select the chart that plans better here | select the chart matching the **regime label** | **fixed by §8.3** |
| G4 "regimes are real" | distributional difference in trajectories | `1e-6` mean-pixel difference (CHECK 4.1) | **fixed by §9** |
| G6 "denominator guard" | exercise the zero-displacement guard | the motion gate returns first, guard never reached (CHECK 4.2) | **fixed by §9** |
| E1 closed analytically | routing recovers a real competence gap | a gap that does not exist in the current library | **addressed by §8.1 → §8.2 gating** |

The E2 one deserves emphasis because it is the paper's title. In a paper called *"Measure Fitness,
Don't Infer the Regime,"* scoring the router against the regime label measures regime inference. §8.3
adds a competence-based ground truth so the claim and the measurement agree.

---

## §8. Experiments

All at §3.6's configuration, all arms identical, all paired on identical seeds.

### 8.1 E0′ — adapter capacity under replanning (RQ0). **Runs first; gates E1.**

The existing E0 answer ("a 10.7k-param adapter monotonically lowers UMF but buys nothing in planning
success", 43/100 vs 44/100) was measured under **one open-loop plan per episode** — a protocol that
structurally cannot express a model improvement, since the model's better prediction is never used to
correct anything. `atlas_out/e0_planning_nas2` already hints the direction flips (+10pp, n=20,
CI [−10, +30]). E0′ is that test at power.

- **Arms:** frozen `c₀` vs. `ln_act` chart, on **`R2` only**, N = 100 paired episodes each.
  **[R1 DROPPED — explicit human sign-off 2026-08-29 ("you can drop R1 … if time
  permits then we will run the experiments").** Pre-registration change per
  `CLAUDE.md` §1.8. Basis: P0-F/G4 measured R1 as a *prediction-level* shift only
  (Δpose +8–9 px, inside the ±13 px null band, flat 40→200 steps) → an R1
  planning arm is likely uninformative a priori; and R1 collection shares 50/100
  tasks with the eval set (`seed_base["R1"]=0`, re-verified model-free
  2026-08-29). Consequence: the decision rule below is read over `{R2}`; E2's
  R0/R1 cell-B replicate (§8.3) and the K≥3 stream option (§D) are foreclosed,
  not deferred. R1 may be revisited as future work if budget permits.]
- **Charts:** trained on `onpolicy` data (C.3); the `dataset`-collector chart is run as the
  one-variable comparison arm on R2 only.
- **Capacity sweep:** `{ln_act, lora4, full}` on R2 only, at a **matched trajectory budget** — the
  confound `OPUS_REMAINING_TASKS.md` #12 flags in the existing capacity numbers.
- **Reported:** Δ success with paired bootstrap CI and McNemar; UMF on a **disjoint** test split
  (`seed_offset = 20_000`) so no optimistic bias; the UMF-vs-success scatter within arm.
- **Decision rule (pre-registered, replacing the inapplicable "≥90% of full" rule, which became
  undefined once `full`'s gain went negative):** report the smallest kind whose paired Δ success CI
  **excludes zero** in R2 (was "at least one regime" pre-2026-08-29, when R1 was
  in scope — see Arms). If no kind qualifies, that is the RQ0 answer and it is
  reported as such — *"under receding-horizon replanning, at N=100 paired, no adapter class in
  {ln_act, lora4, full} produces a detectable planning gain"* — which is a materially stronger and
  more citable negative result than the single-plan version, because it removes the one protocol
  objection that undermines it.

**Cost:** ~460 replans per (arm-pair x regime); ~$30 at `iterations=30`, ~$15 at 15.

### 8.2 E1 — fitness routing (RQ1). **Gated on E0′.**

E1's pre-registered rule reports normalised recovery only when `SR_oracle − SR_random ≥ 10 pp`.
`HANDOFF.md` §7.1 measured that spread at 2.5–3.3 pp under the single-plan protocol and closed E1
analytically — correctly, on that data. Under §3.6 the spread must be **re-measured**, and E0′
measures it: the oracle−random spread is a direct function of the per-regime chart advantage E0′
estimates.

> **GATE E1-GO.** Run E1 iff E0′ establishes a per-regime chart advantage implying
> `SR_oracle − SR_random ≥ 10 pp`. Otherwise report E1 as **unrunnable by its own pre-registered
> rule**, with the measured spread, and state that no routing algorithm can recover a competence
> difference the library does not contain. Per `CLAUDE.md` §1.8 that is a result, not a failure to
> paper over — and it is a *sharper* result than a low recovery number would be.

If it runs: routers `umf · e1 · sdyn · random · oracle_id`, pure argmin (`m=0`), 60 episodes x 1 seed,
identical seeds across routers, `N_WARMUP_REPLANS = 1`. T1 as in the proposal, plus the `m`
sensitivity rows.
**Cost:** 5 x 60 x 2.3 ≈ 690 replans ≈ $22 at `it=30`, $11 at 15.

### 8.3 E2 — dynamics vs. appearance (RQ2), with a competence-based ground truth

E2 needs no planner for its scoring half, so its cost is ~$1 and its existing structure is sound.
Two changes:

1. **Re-run every routing number** under the pinned normaliser at `m = 0` (§6.3). This is also the
   deciding experiment `COHERENCE_AUDIT_2.md` §3.5-1 names for the unresolved 60.3%/36.5% conflict,
   run on `--charts-dir atlas_out/e2_charts` (the post-rollout-fix charts) on HEAD, with `sdyn`
   evaluated pure-argmin. It settles the conflict either way, and the "reversal" banners in
   `FIXLOG.md` / `ATLAS_SUMMARY.md` §4.5 / `E2_RESULTS.md` are corrected to match the outcome.
2. **Add a second ground truth.** Alongside regime-label accuracy, report **competence-matched
   accuracy**: on a subset of episodes, run *every* chart to completion and define the correct chart
   as the one that actually achieves the lower final goal distance. Regime-label accuracy stays (it
   is the field's convention and the comparison to S-dyn needs it), but the headline claim is stated
   against competence, so that the measurement matches the title. Cost of the subset: ~20 situations
   x K charts x 2.3 replans ≈ 140 replans ≈ **$4.50**.

Cells: A (R0/R0), **B decisive** (R0/R2; ~~plus R0/R1 as replicate~~ — R0/R1
replicate dropped with R1, human sign-off 2026-08-29, see §8.1 Arms), **C over-expansion**
(R0 vs R0+colour, charts committed must be 0), D realistic.

### 8.4 E3 + E4 — expansion ladder and the continual stream (RQ3, RQ4)

**This is the experiment the project has never run, and the venue's core theme.** S2 = `A,B,A,B,A,B`
with A = R0, B = R2; paired seeding (episode `i` of segment `s` uses `hash(s,i)` for every arm); the
seven-arm ladder from the proposal, unchanged. Arms 4→5→6 are E3; arms 1→2→3→6 are the adaptation
ladder; one run yields both tables.

Prerequisites, all now satisfied or scheduled: B3's motion-gate recalibration (landed 2026-08-28,
gate 317.77 → 117.62, gated fraction 100% → 33%); B4/C6's silent-no-refine fix (landed); G7 liveness
(§9); GATE P0-C for `iterations`.

Sizing is set in §11.2 by whatever GATE P0-C buys. The design invariant: **seeds and
episodes-per-segment are cut before arms are cut**, because the seven-arm ladder is the attribution
argument and a ladder with a missing rung attributes nothing. If the budget cannot support the full
ladder, the pre-registered cut order is in §11.3.

### 8.5 E5 — cross-policy diagnostic (supplementary)

`M[i,j]` = chart `i`'s UMF on chunks generated by chart `j`'s plans; per-column argmin accuracy and a
column-normalised heatmap. Under C.3 this acquires a second, load-bearing purpose: it is the
measurement of the on-policy/off-policy caveat that §5.2 states. Report it as such, not only as a
reply to the "charts are judged on the selected chart's own data" objection.
**Cost:** K² x ~10 chunk-generating episodes; ~$8 at K=3.

---

### 8.6 ADDENDUM 2026-08-30 — FABLE5 six-day plan (supersedes the E1/E2/E3+E4 launches above)

V3-18/V3-19 (see `FIXLOG.md`) closed E0′'s RQ0 question in the negative and worse:
the on-policy `ln_act`×R2 chart improves every mean-based statistic (held-out UMF
−22.5%, CEM ρ 0.001→0.28, elite-set 124→57 px) while collapsing paired closed-loop
threshold success (1/20 vs 10/20, McNemar p=0.0039). `research_audit/FABLE5_VALIDATION.md`
(2026-08-30) is the blind-spot pass; `research_audit/FABLE5_SIX_DAY_PLAN.md` is the
resulting Aug 30 → Sep 5 plan. **E1, E2, E3+E4, E5 as specified in §8.1–8.5 do NOT
run** (rationale: `FABLE5_VALIDATION.md` §6 — every experiment presupposing UMF-selected
charts carry competence is now uninterpretable). The Phase-0 freeze pipeline (§11.1
τ/q/m/n_probe/K_max/σ_r/G7-asserting) also does **not** run (`FABLE5_SIX_DAY_PLAN.md` §1).

**Day 1 experiments — pre-registered here before launch (2026-08-30):**

- **1.1 Settle-check (criterion validity).** New additive `--settle-steps` flag on
  `run_e0_planning.py` (FALSIFICATION in `FIXLOG.md`). On a pass-through success, hold
  position N=15 raw steps with a zero action and re-check `block_success`. Runs: frozen
  R2 it=10/nas=2, and both nas=6 arms (baseline + ln_act), 20 episodes, `--settle-steps 15`.
  **Decision rule:** report both pass-through SR and settled SR. If the frozen arm's
  settled SR is ≥2 episodes below its pass-through SR, criterion validity leads the
  evaluation section and the abstract uses settled numbers (pass-through footnoted);
  otherwise pass-through stands and this is a robustness confirmation.
- **1.7 Headline at N=50 paired, fresh tasks (replication).** Episodes 20–49 (disjoint
  from every existing cell), frozen + ln_act chart, R2 damping=0.5, it=10/nas=2 AND nas=6.
  **Decision rule:** the 20–49 set must reproduce the *direction* (chart SR below frozen,
  sd-ratio < 1) at both cadences. If it does not, **STOP and report** — that is a
  seed-set-fragility finding and the paper pivots to it. Do NOT merge 20–49 into 0–19
  and average it away.
- **1.2 R0 anchor.** (a) frozen baseline R0 (no regime-config), it=10/nas=2, 20 eps;
  (b) ln_act (the R2 chart) on R0, same protocol — cost of a false-positive route.
  Purely descriptive anchors, no decision rule.
- **1.3 Repeat the headline chart cell.** ln_act R2 damping=0.5, it=10/nas=2, 20 eps,
  fresh launch. **Decision rule:** if it differs from the archived 1/20 by >2 successes,
  single-launch variance becomes a stated limitation and the N=50 result is the headline.
- **1.4 R0 UMF crosscheck (B-6).** Re-download `trajs_R0.pt`, score the P0-G chart and
  frozen c₀ on R0's T=2 windows (mirror of `c2_widened_offline_umf.json`). Descriptive.

Every run `--detach`, archived under `phase0_v3/`, ledgered in `EVIDENCE_LEDGER.md`
the same day. Experiments freeze end of Day 4.

**1.10 damping=0.1 milder-shift check — pre-registered 2026-08-31 (external review + user).**
`DAY1_CADENCE_METRIC_ANALYSIS.md` §3: the R2 contact collapse (frozen 15.1/ep on R0 → 6.0 on R2,
−60%; training data −74%) is dominated by the *regime*, not cadence or chart. §15-2's pre-registered
contact-collapse fallback is `damping=0.1`, measured-triggered, never applied (B-2). Test whether a
milder shift restores planner engagement and whether the prediction/control dissociation survives it.
- **Runs:** frozen `c₀` at `--regime R2 --regime-config '{"damping": 0.1}'`, it=10, N=300, H=6,
  `--settle-steps 40`, seeds 0–19, at **nas=2 and nas=6**. (`c2_settle2_dmp01_baseline_nas{2,6}`.)
  Chart arm deferred — only run if the frozen arm shows restored engagement.
- **Metrics reported:** contacts/episode, pass-through SR, **settled SR**, settled block-distance —
  each vs the `damping=0.5` frozen arm (1.1-R) and the R0 frozen arm (1.2).
- **Decision rule (pre-registered):**
  (a) if frozen contacts at 0.1 are **≥ ~10/ep** (near R0's 15) AND settled SR **> 0** → engagement
      is restored; `damping=0.1` becomes the regime the chart should be trained/evaluated in, and
      the R2 (`0.5`) numbers are reported as "an over-severe shift" limitation.
  (b) if frozen contacts stay **< ~7/ep** and settled SR **= 0** → the collapse is not a
      severity artifact of `0.5` specifically; R2 as a family disengages the planner. Report
      `damping=0.5` as-is with the collapse disclosed, and the milder-shift check as a negative
      control that rules out the "we just picked too hard a number" objection.
  (c) intermediate → report both, no regime change without explicit sign-off (§15).
- **Cost:** ~$0.5 (2 × 20 episodes, L4).

---

**1.9 R0-chart control — pre-registered 2026-08-31 (user proposal; supported by external review).**
The C-2 dissociation and 1.1-R "less destructive" reading are stated *for R2*. A reviewer will ask
whether the on-policy-collect + L2-fine-tune recipe produces the same signature in *any* regime, or
specifically under the R2 shift. Control: train `ln_act` on the P0-G **R0** on-policy trajectories
(`phase0_v3/p0g_onpolicy_r0/trajs_R0.pt`, volume copy — same recipe/params as the R2 chart:
N=300, it=10, nas=2, 100 trajs, steps 2000 + early stopping), evaluate held-out UMF and (conditionally)
closed-loop control on R0 vs frozen c₀.
- **Stage 1 (precursor, ~$1, launched):** `p0g_finetune --regime R0`, then offline widened-UMF on
  R0 chunks (`scripts/c2_widen_offline_umf.py`, forward-only).
- **Pre-registered gate:** if the R0-chart's held-out UMF improvement over c₀ is **< 5%**, the
  control is **declared void** (c₀ already models R0 — its UMF there is 0.254 vs 0.627 on R2, so
  little headroom exists) and **no closed-loop screen runs**. The 1.4 interpretation then stays
  "specialisation vs no-headroom: undetermined" and the corresponding paper sentence is not written.
- **Stage 2 (only if Stage-1 gate passes, ~$1):** R0-chart vs c₀, 20 paired episodes on R0,
  it=10, nas=2 **and** nas=6 (mirror the settle protocol). Report settled block-distance Δ + the
  neither-succeeded subset, same as 1.1-R. **Power note:** baseline R0 SR ≈ 95% (19/20), so SR is
  ceiling-limited for detecting *benefit* — a null SR with a real UMF gain is the informative
  outcome (it would show the recipe's compression signature is regime-general).

---

**1.1-R settle-check RE-RUN — pre-registered 2026-08-31 (after the V3-20 floor-effect
correction; FIXLOG V3-20/V3-21).** The first settle-check found settled SR = 0/20 in
all 4 arms and wrongly concluded "no dissociation" — a floor effect at the 20 px
threshold. On settled *distance* the dissociation reverses (nas=6 chart 64.0 vs frozen
101.5 px, p=0.011; neither-succeeded subset p=0.0078). This re-run makes the
measurement clean.
- **Code:** `--settle-steps` now applies to every episode + records `settled_trace`
  (checkpoints 1/5/15/30/45/N). `--settle-steps 40`.
- **Runs:** 5 arms, 20 episodes each — `c2_settle2_{baseline,ln_act}_{nas2,nas6}` (R2
  `damping=0.5`, it=10) + `c2_settle2_R0_baseline_nas6` (R0 control, no regime-config).
- **Primary metric:** paired Δ of the settled block-distance (Wilcoxon + bootstrap CI),
  reported per cadence, with the neither-succeeded subset as the confound-free cut.
  **Not** settled SR (floors). nas=6 is the headline cadence; nas=2 is secondary
  (cross-launch reproducibility limitation, see 1.1-R non-determinism note).
- **R0 control decision rule:** under R0, settled SR should ≈ pass-through SR (real
  successes survive). If R0 settled SR also collapses, the settle mechanism itself is
  suspect and the whole check is reconsidered before any settled number is published.
- **Falsification built in:** `c2_settle2_baseline_nas6` must reproduce the archived
  `c2_settle_baseline_nas6` 11 successful episodes' `settled_block_pos_diff` exactly
  (nas=6 is deterministic).
- **Cost:** ≈ $1.9 (L4).

---

### 8.7 ADDENDUM 2026-08-31 — FINAL FIVE-DAY PLAN Day 1 (supersedes the FABLE5 six-day experiment list)

`research_audit/FINAL_FIVE_DAY_PLAN.md` is now the operative experiment plan (Aug 31 → Sep 5
AoE). It retains every governance rule of §8.6. Day 1 launches, pre-registered here **before
launch** (2026-08-31), all via `modal/modal_e0_planning.py::main`, `--detach`, on Modal
profile `aiden-dsouza-201323`, archived under `phase0_v3/`, standing config `it=10, N=300,
H=6, --settle-steps 40`, charts `--charts-root phase0_v3 --charts-subdir p0g_onpolicy`:

- **1.B — R2 adapter screened at `damping=0.1` (cheap fallback for 1.A).** The existing
  `p0g_onpolicy` `ln_act` R2 chart vs frozen `c₀` at `--regime R2 --regime-config
  '{"damping":0.1}'`, nas=2 and nas=6, seeds 0–19. Frozen arm already on disk
  (`c2_settle2_dmp01_baseline_nas{2,6}`, it=10/N=300/settle-40 — verified this session:
  pass-through SR 0.55/0.70, settled 0.10/0.20). New: the chart arm only.
  Artifact `phase0_v3/dmp01_transfer_ln_act_nas{2,6}/`.
  **Decision rule:** report settled block-distance paired Δ (Wilcoxon + bootstrap CI) and
  settled SR, chart vs frozen, per cadence. Caveat travels with any null: chart trained at
  damping 0.5, tested at 0.1 → a null is ambiguous between "effect is severity-specific" and
  "adapter is off-distribution." If 1.A lands, 1.A supersedes this; 1.B becomes a
  transfer-robustness footnote.
- **1.C — damping dose ladder (frozen `c₀` only).** `damping ∈ {0.05, 0.2, 0.3}` × nas ∈
  {2,6}, seeds 0–19, settle-40. 6 new cells. Artifact
  `phase0_v3/ladder_dmp{005,02,03}_baseline_nas{2,6}/`.
  **Decision rule (H1):** the claim is **monotone divergence** — pass-through SR falls more
  slowly than settled SR as damping rises. If the two fall together at some intermediate
  damping, report it verbatim ("criterion valid in a band, fails outside it"). **Do not drop
  points.** Also recompute coast (residual momentum) at each new damping for H6.
- **1.D — N=50 replication on disjoint tasks.** `--episode-start 20 --episodes 50` (tasks
  20–49, disjoint from every existing cell), frozen + `ln_act` chart, R2 `damping=0.5`, nas=2
  and nas=6, settle-40. Artifact `phase0_v3/n50_{baseline,ln_act}_nas{2,6}_ep20-49/`.
  **Pre-registered replication target:** settled block-distance at nas=2 (chart < frozen).
  **Decision rule:** if the direction replicates on 20–49, merge and report paired n=50,
  disclosing the two-launch structure. If not, **report both sets separately** and §4's claim
  weakens to "significant in one of two task sets." **Do not average it away.** §3 (H1–H3) is
  frozen-baseline-only and unaffected.
- **1.A — `damping=0.1` on-policy chart collection (longest pole).** `modal_phase0.py` P0-G
  collector at `damping=0.1`, same recipe as the R2 collection (100 train / 8 val / 8 test,
  CEM 300×10 nas=2, `_determinism.py` active, `total_contacts>0` filter OFF).
  Artifact `phase0_v3/p0g_onpolicy_dmp01/`. **Hard abort:** if the chart is not trained AND
  screened by EOD Day 2, drop it — 1.B is the guaranteed fallback.

**1.E (`--no-early-stop` flag) is CUT** — not deprioritised. Reasons recorded in the plan §1.E:
redundant with 1.G.2's matched-compute subset, measures a protocol nobody deploys, and early
termination is the phenomenon (1.G.5), not a confound. No code change to a protected file
(`atlas/score.py::umf`, `atlas/stats.py` existing fns, the planning loop) is in scope.

**THE ONE RULE:** experiments freeze at EOD Day 2 (Sept 1). Days 3–5 are writing and checking.

---

### 8.8 ADDENDUM 2026-09-01 — FINAL FIVE-DAY REVISION-2 launches (pre-registered before launch)

Per `FINAL_FIVE_DAY_PLAN.md` REVISION 2 §R2.2. All on Modal profile
`aiden-dsouza-201323` (`MODAL_PROFILE` pinned per command), `--detach`, archived under
`phase0_v3/`, standing config `it=10, N=300, H=6, --settle-steps 40`, charts
`--charts-root phase0_v3 --charts-subdir p0g_onpolicy --out-root phase0_v3`.

- **A — controller-family rank inversion.** 8 new planning cells, R2 `damping=0.5`, N=300,
  H=6, **nas=2**, `--settle-steps 40`, seeds 0–19, `--kind {baseline,ln_act}`:
  `--iterations ∈ {1,3,30}` (6 cells) + `--objective-alpha 0` (2 cells, `--iterations 10`).
  With `it=10/nas=2` and `it=10/nas=6` already on disk (`c2_settle2_*`) this makes **12
  controllers**. Artifacts `phase0_v3/fam_it{1,3,30}_{baseline,ln_act}_nas2/`,
  `phase0_v3/fam_alpha0_{baseline,ln_act}_nas2/`. **No code change** — `--iterations`,
  `--objective-alpha`, `--num-act-stepped` are all live flags (Modal wrapper params exist).
  **Pre-registered decision rule:** rank all 12 controllers by (i) pass-through SR and (ii)
  mean settled block-distance; report **Kendall τ between the two orderings with a
  permutation CI**, and the per-pair inversions. τ near 0/negative → the two criteria induce
  systematically different orderings over a controller family (§4 upgrades from anecdote to
  measurement). τ near 1 → the criteria largely agree and the inversions are confined to
  pairs differing in outcome variance (makes H7 the precise scope condition). **Both branches
  are reported as-is.**
- **B — n=100 on the headline.** `--kind {baseline,ln_act} --regime R2 --regime-config
  '{"damping":0.5}' --episode-start 50 --episodes 100` (⇒ 50 episodes, tasks 50–99, disjoint
  from 0–19 and 20–49), **nas=2 only**. Artifact `phase0_v3/n100_{baseline,ln_act}_nas2_ep50-99/`.
  **Pre-registered decision rule:** report the three task sets (0–19, 20–49, 50–99)
  **separately and merged (paired n=100)**, whatever they show. Primary metric = paired settled
  block-distance (Wilcoxon + bootstrap CI), and the neither-**pass-through**-succeeded clean
  subset (R2.0-a convention). If 50–99 does **not** replicate the direction (chart settled
  distance < frozen), §4 weakens to "significant in two of three task sets" and task-set
  variance becomes a stated limitation — **do not average it away**. §3 (H1–H3) is
  frozen-baseline-only and unaffected either way.

---

## §9. Gates

| Gate | Checks | Change from v2 |
|---|---|---|
| **G1** identity | `{c₀}` only ⇒ trajectory identical to frozen; a refined chart must change the output *before* the restore check | unchanged (rewritten 2026-08-26, now genuinely rigorous) |
| **G2** prequential | over-refine X on `W`, score on `W′`; X must not auto-win; asserts the leak signature | unchanged |
| **G3a/G3b** probe fires / discriminates | unchanged — but the paper must keep the scope caveat: these test the `Expander` **primitive**, never planning competence | unchanged |
| **G4** regimes real | **rewritten.** Must run against a real `env_factory` (currently hard-skipped in headless mode and therefore never run), and must be a genuine distributional test — two-sample KS or paired bootstrap on a **trajectory statistic** (block displacement, final block pose) per regime, ≥ 20 rollouts each — not `1e-6` on a mean pixel value (CHECK 4.1) | **rewritten** |
| **G5** pairing | unchanged (demonstrated to fail on mismatched seeds) | unchanged |
| **G6** denominator | **extended.** Add a sub-check with `motion_gate=None` on a static chunk, so the zero-displacement guard is actually reached (CHECK 4.2). Add an assertion that the gate was calibrated at `frameskip x num_act_stepped` (§6.6's invariant) | **extended** |
| **G7** mechanism liveness | **new.** §7.1 — firing rate of every thresholded mechanism, on real data, at the production config, inside a stated non-degenerate band | **new** |

**Rule, unchanged and non-negotiable:** a gate that has not been run has not passed, and a gate whose
rewrite has not been demonstrated to **fail on a deliberately broken input** is not evidence. This is
what made G1, G2 and G5 worthless before Phase 2 and what made G4 worthless still.

---

## §10. Statistics

Unchanged from v2 §8 — `atlas/stats.py` was independently verified correct by three separate audit
passes and is not touched. Every table reports Δ with a CI, never two bare means; `paired_bootstrap`
and `mcnemar_paired` as specified; `normalised_recovery` returns `None` below a 0.10 spread and that
behaviour is preserved.

One addition. v2's power argument (360 paired episodes/arm) assumed a compute budget that does not
exist (§2). Whatever E4 is finally sized to, **the achieved power is stated in the paper** — the
number of paired episodes per arm and the minimum detectable paired difference at that N — rather
than the design being quietly reduced and reported as if it were the original. A 120-pair McNemar
detects roughly a 15 pp paired difference; that is a limitation to disclose, not to hide.

---

## §11. Budget and phasing

### 11.1 Phase 0 — measure, then commit (~$6, mostly forward-only)

| # | Measurement | Feeds | Cost |
|---|---|---|---|
| P0-A | `UMF(c₀)` distribution on R0 `onpolicy` chunks → **τ** | §6.1 | ~$0 (mostly on disk) |
| P0-B | latent displacement on block-static chunks → **motion gate** | §6.6 | ~$0 |
| P0-C | **`iterations` gate**: frozen SR at `it ∈ {30,15,10}`, n=20 paired | §3.5 | $2.30 |
| P0-D | per-chunk strike rate under matched and shifted regimes → confirms **q** | §6.2 | ~$0 |
| P0-E | per-router score IQR on the calibration chunk set → **σ_r** | §6.3 | ~$0 |
| P0-F | G4 rewrite + G7 liveness sweep at the production config | §9 | ~$1 |
| P0-G | `onpolicy` collection, 100 trajectories x 2 regimes | §5 | ~$15 (≤$8 if P0-C buys `it=15`) |

**Nothing downstream launches until P0-C and P0-F/G7 have both passed and been read directly.**

### 11.1-R Phase-0 results (updated as gates land)

*Raw artifacts live at `phase0_v3/` in the repo tree (downloaded, not temp) and the
`atlas-data` Modal volume. The arithmetic record is `research_audit/EVIDENCE_LEDGER.md`
§5 — that file wins on any disagreement. This subsection is the plan-side pointer.*

| Gate | Status | Measured | Raw file | Modal app (aiden) |
|---|---|---|---|---|
| P0-A τ | provisional (dataset-replay proxy) | **0.262** (P95 UMF(c₀), n=137 R0 informative) | `phase0_v3/phase0_summary.json` · `phase0_chunks.jsonl` | `ap-5yW7XOAH0wQWAtmDEWMiHF` |
| P0-B motion gate | provisional | **242.7** (P95 latent ‖z_T−z_0‖_F, n=389 block-static; gates 71/68/52% of R0/R1/R2) | same | same |
| P0-D strike rate | provisional | R0 **0.051** / R1 0.340 / R2 0.675 | same | same |
| P0-D q | provisional | **3** (derived: 0.0511²·1100 ≥ 1 > 0.0511³·1100) — equals the default | same | same |
| P0-E σ_r | provisional | umf **0.0462** · e1 **2133.5** · sdyn **0.0409** (IQR over R0 informative) | same | same |
| P0-C iterations | **COMPLETE** — it=30 40% (8/20) · it=15 45% (9/20) · it=10 50% (10/20), n=20 paired. Both cuts inside the it=30 CI (McNemar p=1.000 / 0.625). **Decision rule adopts `iterations = 10`.** Caveat: n=20, CIs ~±25pp — a ≤10pp real degradation would not be caught; §14 fallback (revert to it=30, cut sizes) stands if E0′/E4 look degraded. | 40 / 45 / 50 % | `phase0_v3/p0c/p0c_it{10,15}_baseline_R2.jsonl` | aiden, `p0c_it{10,15}` |
| P0-G onpolicy collection | not started — **needs a funded Modal account** | — | — | — |
| P0-F G4 | **DONE** — R2 a real trajectory shift (Δpose +33 px); **R1 prediction-level only** (Δpose within noise, flat 40→200 steps). Rescope: R2 for all trajectory/SR claims, R1 for prediction/RQ2 only. | — | `phase0_v3/g4_*` | local, $0 |
| P0-F G6 | **DONE** — "UMF blow-up near zero motion" checked over 1440 chunks: **does not occur** post-rollout-fix (max UMF 2.05). Drop the explosion framing; G6 = informativeness-filter test only. | — | EVIDENCE_LEDGER §5 | local, $0 |
| P0-F G7 (calibration stage) | **DONE (Groups A + B).** Motion gate / strike counter / router switch / probe all alive & non-degenerate at **τ≈0.26, gate P50, m=0, K_max 5–6**. `τ=0.5` kills expansion (5+ confirmations). **New finding: single-chunk accept gate → ~33–50% one-hit-wonder commits; `n_verify≈3` fixes it** (additive `verify_chunks` param in `atlas/expand.py`). Expansion helps net: R2 UMF 0.415→0.393→0.380 (frozen/refine/full), and `full` forgets R0 less than `refine`. | — | `phase0_v3/g7_*`, EVIDENCE_LEDGER §5 | local, $0 |
| P0-F G7 (asserting) | not started — re-runs one frozen config on P0-G on-policy charts + ≥576-chunk stream | — | — | — |

**Provisional-vs-final:** P0-A/B/D/E chunks are **dataset-replay** (real demos replayed
under each regime, forward-only), a proxy for §6's `onpolicy` source. They are re-derived
against P0-G's `onpolicy` chunks before Phase 0 closes; per §6.6 both values are reported.

**Open decision surfaced by P0-A:** measured τ ≈ 0.26 vs. the §1.7-pinned 0.5. Under §6.1's
own logic this validates the *method* (5% R0 false-strike by construction), but adopting it
edits a §1.7 non-negotiable and needs explicit human approval (§15-5). Both values carried
until then.

### 11.2 Indicative allocation (~$90)

| Item | Replans | at `it=30` | at `it=15` |
|---|---:|---:|---:|
| Phase 0 (incl. collection) | ~650 | $21 | $11 |
| E0′ (RQ0, 2 regimes + capacity sweep) | ~900 | $29 | $15 |
| E2 (re-run + competence subset) | ~150 | $5 | $3 |
| E1 (**if** GATE E1-GO passes) | ~690 | $22 | $11 |
| E5 cross-policy | ~250 | $8 | $4 |
| **E3+E4 stream** | remainder | — | **~$45** |
| | | *over budget* | **~$89** |

At `it=15`, E4's ~$45 buys ~2,760 replans ≈ **1,200 episodes** — e.g. 7 arms x 6 segments x 14
episodes x 2 seeds (168 paired episodes per arm), or x 20 episodes x 1 seed. At `it=30` E4 is not
affordable alongside E1 and the cut ladder engages. **This is why GATE P0-C is the first thing that
runs.**

### 11.3 Pre-registered cut ladder (applied uniformly across arms, and reported)

1. E5 cross-policy → supplementary, drop if needed
2. E2 cell D (realistic combined shift)
3. E4 seeds 2 → 1
4. E4 episodes/segment 20 → 14 → 10
5. E1 (only if GATE E1-GO already failed — otherwise E1 outranks E4 episode count, since RQ1 is the
   project's declared gate)
6. **Last:** drop ladder arm 4 (ATLAS-fixed-library)

Arms are never cut before episodes; the ladder is the attribution argument.

---

## §12. What on disk survives

| Artifact | Under v3 | Why |
|---|---|---|
| `e0_planning_n100`, `e0_v3_planning_*`, `e0_v4_planning_*` (`nas=6`) | **Valid**, re-scoped | A correct measurement of the single-open-loop-plan protocol; that is exactly how N1 must be reported |
| `e0_planning_nas2` | **Valid**, and now the pilot for E0′ | Same protocol as v3, n=20 |
| `cost_ranking_*` (N3, incl. the dose ladder) | **Valid** | No chart, router or planner-config dependence; must gain the R0 control row (`COHERENCE_AUDIT_2.md` §3.5-5) |
| `e2_charts/`, `e0_v3_dataset` charts | **Valid as artifacts**; retained as the expert-replay comparison arm | Trained post-rollout-fix |
| Every routing number (`e2*`, `e1*`) | **Superseded** — re-run under §6.3 | Both existing normalisers degenerate; this settles the unresolved conflict rather than adjudicating it |
| `e0_v3_hybrid`, `scripted` charts and conclusions | **Retracted** | Objective substitution (CHECK 2) |
| `e0/` (pre-rollout-fix) | **Invalid**, as already banner-marked | — |
| E4 / S2 | Nothing exists | Never run |

---

## §13. Literature check (Part E), performed this session

**Novelty, C1.** Searched for post-cutoff work on prediction-error routing among adaptation modules
on frozen visual world models. Nothing new beyond what `LITERATURE_AUDIT.md` already covers: HERA
(2608.05523, a single always-on memory adapter on frozen V-JEPA-2 — not a fitness-routed library),
Continual Model Routing in Evolving Model Hubs (2605.28577, routes between whole pretrained models in
a hub), and adapter-composition routing for LLMs. **C1 stands.** One caveat carried forward
unchanged: a search is not a proof of absence.

**Novelty, C2.** One candidate not in the existing survey: **DIMoE-Adapters** (arXiv:2605.07494,
"Dynamic Expert Evolution for Continual Learning in Vision-Language Models", Self-Calibrated Expert
Evolution). Its expansion trigger is a performance-degradation threshold computed on the **current
task's own training data**, and new experts are created and trained immediately on the triggering
data — same pattern as all six papers already checked. `[SECONDARY]` — read via an automated
full-text summary, not by me directly; a future pass should confirm the expansion criterion by
reading §3 of the PDF (saved locally by the fetch). It does not scoop C2 but **should be cited** as a
2026 concurrent instance of detect-then-commit expansion. **C2 stands.**

**The planner-budget precedent — this is the substantive finding, and it changes §3.**

1. **The substrate itself ships multi-replan MPC configs with a short per-replan action budget, for
   this same DINO-WM backbone.** `vendor/jepa-wms/configs/.../mw/dino-wm/reach_L2_cem_sourcexp_H6_nas3_...yaml:202–207`
   is `iterations=15, num_samples=300, num_elites=10, horizon=6, num_act_stepped=3`, with
   `max_episode_steps=100` and `frameskip=5` — i.e. `nas < horizon` **and** `iterations < 30`, both
   of the deviations §7.0a treated as departures from the substrate. RoboCasa's config is `H3_nas1`.
   Push-T's `H6_nas6` is simply the substrate's open-loop variant, not a universal substrate rule.
   `[VERIFIED — read directly from the vendored config files.]`
2. **DINO-WM's own results table separates the two protocols and prefers replanning.** Its Table 8
   defines CEM as "optimize a sequence and execute without correction or replan", GD as open-loop
   gradient descent, and MPC as "allowing replan and receding horizon with CEM"; the Push-T numbers
   are **MPC 0.90 > CEM 0.86 > GD 0.28**. `[SECONDARY — from a search-result extract of the paper's
   table; direct PDF extraction failed. Confirm before citing a specific number in the paper.]`
   If confirmed, this also corrects a documentation error: `ATLAS_implementation_plan_v2.md` §7.0a
   and `CLAUDE.md` attribute "~90% Push-T SR" to the `nas=6` config, but 0.90 is the **MPC** row and
   `nas=6` is the **CEM** row (0.86). Worth fixing before submission — it is the same class of small,
   checkable citation error as the MBCD AAMAS/ICML venue slip.
3. No paper was found that quantifies "open-loop single-plan CEM suppresses measurable
   model-improvement benefit" for JEPA-class world models specifically. `LITERATURE_AUDIT.md` §9
   point 3 reached the same conclusion. The paper should therefore cite standard receding-horizon
   control (Kwon & Han) for the feedback-robustness rationale and state plainly that E0′'s
   single-plan-vs-replanning contrast is **its own contribution to that question**, not a restatement
   of an established empirical result. Under v3 this is no longer a hedge to apologise for — E0′
   *measures* it, at N=100 paired, which makes it a small positive contribution rather than a
   limitation.

---

## §14. Risks, with pre-registered responses

| Risk | Detected by | Response |
|---|---|---|
| `iterations` cut degrades the baseline | GATE P0-C | Keep `it=30`; cut experiment size per §11.3 instead. Never trade planner budget for power silently |
| E0′ finds no chart advantage under replanning | E0′ | The RQ0 answer, reported. E1 becomes unrunnable by its own rule (GATE E1-GO) and is reported as such. The paper leads with N3 + a now-protocol-hardened N1 |
| E2 re-run kills the routing result | E2 | Report it. The reversal banners are corrected in the direction the data lands, and "measured fitness does not identify the matched adapter in a frozen visual latent space" is a citable answer to RQ1 |
| Probe never fires / always fires | GATE G7 | τ/q are recomputed from the *measured* strike rates (§6.2) and the recomputation is reported. Not silently retuned |
| Contact collapse under R2 is real, not a collector artifact | P0-G contact counts | Fall back to `damping = 0.1` (already characterised) and report the change |
| E4 still unaffordable after P0-C | §11.2 arithmetic | §11.3 cut ladder, in order, applied uniformly and reported. Arms are cut last |
| The stream runs but every mechanism is inert | GATE G7 | Do **not** publish a ladder table whose arms are secretly identical. Report the liveness rates and scope the claim to what ran |

---

## §15. Deliberately left open for a human

1. **`iterations`.** Set by GATE P0-C, not by this document. Everything downstream is sized off it,
   so this is the one measurement that must be read by a person before budget is committed.
2. **R2's contact collapse** (§4). Confounded; the C.3 collector re-measures it for free. The
   fallback is pre-registered but the call is a person's.
3. **Whether to run E1 at all** if GATE E1-GO fails. The plan's answer is "report it unrunnable,
   which is a result"; a reviewer-facing judgement about whether the paper is stronger with a
   low-power E1 anyway belongs to a human.
4. **The `.tex` vs `PAPER_DRAFT.md` divergence.** Out of scope for this plan; `COHERENCE_AUDIT_2.md`
   §3.3 is the live analysis. Nothing in v3 changes which draft should be submitted.
5. **`CLAUDE.md` §1.7.** τ, q, m's normaliser, `n_probe`, `K_max` and the motion gate all change
   under §6. §1.7 is a non-negotiables section and this document does not edit it — the values above
   are a **proposal** requiring explicit approval, per `CLAUDE.md` §8.

6. **C2 accept criterion — single verification chunk → `n_verify` held-out chunks.**
   Added 2026-08-29 from G7 Group B. The current commit gate (`atlas/expand.py::maybe_expand`)
   accepts a candidate iff it beats the incumbent's UMF on **one** held-out chunk. G7 measured
   ~33–50 % of such commits to be one-hit wonders (win that chunk, lose the next same-regime
   chunks) — a statistically underpowered decision rule. An additive `verify_chunks` param
   (default off; E2/N9 and G3a/G3b unaffected) requires a *majority* win over N chunks with
   mean UMF < τ. **Clean measurement — same commit set classified with both follow-up
   windows in one run.** (Two methodology corrections were made along the way and logged:
   the follow-up window overlapped the accept window → circular, fixed; and the candidate
   fit is Dropout-nondeterministic, ±2 commits/6 seeds, so *cross-run* comparisons are
   confounded.) Generalisation rate **42 % (N=1) → 75 % (N=3) → 80 % (N=5)**, fewer commits
   at higher N. The window bug's real effect was only at N=5 (fake 100 % → real 80 %).
   **Working proposal `n_verify = 3`** — 42→75 % is the real gain; N=5 costs 3/8 commits for
   +5 pp. Final value awaits the P0-G re-run: `torch.manual_seed` + Dropout off for
   reproducibility, fixed seeds, ≥576-chunk streams, joint `n_verify` × `n_probe` sweep on
   on-policy charts. §1.7-adjacent — needs explicit approval like τ/q/m.

7. **`R3` candidate — agent PD gains `k_p` / `k_v` (a "stiff vs sluggish actuator" regime).**
   Added 2026-08-28 after P0-F/G4 showed R1 (friction 2.0) is a *prediction-level* shift only —
   a fixed open-loop push ends within sampling noise of R0, flat across 40→200 steps
   (`phase0_v3/g4_duration_sweep.txt`), and friction saturates at 2.0 in `regimes.py` so the
   axis has no headroom. Per `REGIME_DESIGN_REVIEW.md` Task 2, **the only appearance-matched
   dynamics lever left unused** is the pusher's PD tracking gains (`pusht_env.py:384`,
   `self.k_p, self.k_v = 100, 20`). Changing them changes the *kinematic driver's own motion*
   → changes `v_rel` at contact → not subject to the mass-cancellation, and unlike friction it
   should produce **both** a prediction-level and a trajectory-level effect (a harder/softer push
   moves the block further/less). Mild residual caveat (weaker than `action_scale`'s): a planner
   validated at `k_p=100` may plan slightly worse at another value — but the action *space* is
   unchanged, only the tracking dynamics, so it reads as a dynamics shift, not an action-space
   shift. **Not built.** Would need: a `_get_agent()` PD-gain setter in `regimes.py` (~15 lines,
   same pattern as `_set_shape_property`), forward-only UMF calibration to match R2's severity, a
   full G4 pass, then downstream budget. It is the right "second real regime" for future work and
   should be named as such in the paper. If a K≥3 continual stream is wanted *this* cycle without
   new code, use two damping severities (`R0 / damping-0.25 / damping-0.5`) instead — already
   characterised in `cost_ranking_dose_*`, one validated mechanism, tests library granularity.

---

*End of IMPLEMENTATION_PLAN_V3.md. No code was run and no experiment code was modified in the
production of this document.*
