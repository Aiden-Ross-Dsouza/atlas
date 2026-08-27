# ATLAS — PROPOSAL_CODE_ALIGNMENT

**Last updated: 2026-08-27, pass 2 (proposal-code-auditor) — pass 1 traced `ATLAS_proposal_v7.md` §6/§7 and `ATLAS_implementation_plan_v2.md` §6.1a/§7.0a/§7.1/§7.4/§7.6/§7.7 against `atlas/*.py`, `scripts/*.py`, `modal/*.py` and `configs/`, resolving eleven checklist items and finding four previously-unrecorded structural drifts in the E4 ladder. Pass 2 closes all six leftover sub-items pass 1's own table flagged as unresolved (see new section L below) — most consequentially, resolves the E4 regime-persistence hazard as a NON-ISSUE (direct source read of `pusht_env.py::_setup()`), upgrades the `--arms atlas` crash and `CEMPlanner.plan()` clamp findings from inference to direct-read, and finds one new previously-unrecorded drift (`E0_RECOVERY_PLAN.md`'s prescribed E1-harness fix was never applied to code, though it does not contaminate any currently-reported number). No verdict from pass 1 changed. No source code was modified in either pass.**

---

## How to read this file

This is a **documentation pass, not a fix pass.** Every finding below was
verified by reading the code at the cited `file:line`. Nothing was changed.
Several findings are latent bugs; they are recorded, not repaired, because
silently fixing them would destroy the provenance of the numbers already on
disk in `atlas_out/`.

Repository root: `D:/Shubham/DeepLearning/Atlas/atlas/`.
Claim identifiers (C1, C2, N3, L-1, P-1, RQ3, S-4, G-1, …) refer to rows in
`research_audit/CLAIMS_MATRIX.md`.

Verdict vocabulary, used exactly:

- **MATCHES** — the code does what the design document specifies.
- **DRIFTS** — the code does something materially different. Every DRIFT
  names a `file:line` and the claim ID it threatens.
- **NOT YET IMPLEMENTED** — the specified behaviour is absent from the code.
- **COULDN'T VERIFY** — could not be settled by reading; states what would
  settle it.

Four kinds of statement are kept separate throughout, and labelled:
**[CODE]** facts read directly out of the source; **[DOC]** claims made by
this repository's own markdown, which are unverified assertions about the
code until checked; **[INFERENCE]** the auditor's reasoning from the two;
**[OPEN]** questions this pass could not close.

---

## Summary table

| Item | Subject | Verdict |
|---|---|---|
| A | Method order of operations (`atlas_step` / `atlas_refine`) | **DRIFTS** (expansion unreachable; detect-only skips candidate fitting) |
| B | Fixed hyperparameters (τ, q, m, n_probe, K_max, lr, refine steps, motion gate) | **MATCHES** on literal values; **DRIFTS** on the semantics of `m` |
| C | The 7-arm ladder | **DRIFTS** — three adjacent pairs fail the one-mechanism rule; one adjacent pair is behaviourally identical |
| D | Disjoint parameter sets / `chart.restore_()` | **MATCHES** for P-1 as stated; `restore_()` allegation **CONFIRMED** |
| E | Paired seeding | **MATCHES** |
| F | Single shared substrate / frozen backbone | **MATCHES** on the encoder; **DRIFTS** — ATLAS arms' ability to adapt depends on arm execution order |
| G | Planner configuration | **DRIFTS** — experiment-dependent `num_act_stepped`; a `steps_left` unit error in two files |
| H | Chart training protocol | **DRIFTS** — 2-way split reported as held-out; unmatched budgets; `full`×R1 never trained |
| I | Training vs deployment action distribution | **DRIFTS** — collection is open-loop replay, evaluation is CEM; collection budget ≠ eval budget |
| J | Regimes | **DRIFTS** from plan §6.1a's stated values; wrapper bypass is a latent hazard |
| K | Documentation claims | **DRIFTS** — nine specific stale or unverifiable statements listed |

---

## A. Method order of operations

**Specification** (`ATLAS_proposal_v7.md` §6, lines 178–190; `CLAUDE.md` §1.6):

```
1. SCORE   UMF(c; Q) for all c        — skip Q if uninformative
2. SELECT  c* = argmin UMF, hysteresis margin m
3. EXPAND  if UMF(c*) > τ for q consecutive informative checks:
             fit a candidate on the DEFICIT chunks;
             commit only if it beats both τ and c* on the NEXT unseen chunk
4. EXECUTE plan with c*
5. REFINE  1 SGD step on c* — strictly AFTER scoring
```

### A.1 Steps 1, 2, 4, 5 — MATCHES

**[CODE]** `atlas/loop.py:110-123` calls `route()` (score+select) before
anything else. `atlas/loop.py:171-177` returns without refining;
`atlas_refine()` is a separate function (`atlas/loop.py:179`) that the caller
invokes after execution. In the one production caller,
`atlas/harness_e4.py:213-219` runs `atlas_step` at the top of the replan
loop on `prev_chunk` — the chunk executed by the *previous* replan — and
`atlas/harness_e4.py:305-315` calls `atlas_refine` at the bottom, after
`base_env.step` has executed the plan. Scoring is `@torch.no_grad()`
(`atlas/score.py:29`). The order is correct and the chunk scored is genuinely
one the selected chart has not trained on.

Claim **P-2** (strict prequential order) is therefore supported at the level
of code structure. It has still never executed in production (claim **G-1**).

### A.2 Step 3 in `expansion_mode="atlas"` is UNREACHABLE from production — DRIFTS

**[CODE]** `atlas/loop.py:140-150`:

```python
if cfg.expansion_mode == "atlas":
    expander.record(best_umf, encoder_output, actions, proprio_ctxt)
    if (
        next_encoder_output is not None
        and next_actions is not None
        and expander._strikes >= cfg.q
    ):
        probe_outcome = expander.maybe_expand(...)
```

**[CODE]** The only production call site, `atlas/harness_e4.py:217`, passes:

```python
next_encoder_output=None, next_actions=None,
```

and never passes anything else — `grep` over the whole repository finds
`next_encoder_output` assigned a non-`None` value only in
`scripts/smoke_gates.py:310` and `scripts/smoke_gates.py:365`, which call
`Expander.maybe_expand()` directly on synthetic latents, bypassing
`atlas_step()` entirely.

**[INFERENCE]** Therefore **arm 6 (`atlas`) can never commit a chart.** The
guard is unsatisfiable in the E4 runner. `Expander.record()` still increments
strikes and accumulates deficit chunks, so the arm looks alive in the logs
(`strikes` is written into every episode record at
`atlas/harness_e4.py:339`), but `maybe_expand()` is never entered, `charts_committed_cumulative` stays 0, and `probe_outcome` stays `"not_ready"` forever.

**[CODE]** `atlas/harness_e4.py:198-202` carries a comment claiming the
opposite:

```
# Two-deep chunk buffer: chunk k is deficit data for the ATLAS strike
# counter, chunk k+1 is the NEXT unseen chunk maybe_expand() verifies on
```

There is no two-deep buffer. `prev_chunk` (`atlas/harness_e4.py:202`, assigned
at `:318`) holds exactly one chunk.

**[DOC]** `E3_E4_IMPLEMENTATION_PLAN.md:374` specifies exactly this code with
the annotation `# filled on the NEXT iteration`, and `:380-382` says "Keep a
two-deep chunk buffer." The "filled on the NEXT iteration" half was never
implemented.

**Threatens:** **C2** (verification-gated expansion), **RQ3** (ATLAS commits
≈2 vs detect-only >2 — ATLAS's count is 0 by construction, so the comparison
is not a measurement of verification, it is a measurement of a disabled code
path), **L-1** (arms 5→6 do not differ by "verifies"; they differ by "expands
at all"), **C2-probation**, and **N9** in part (N9's live demonstration came
from `scripts/run_e2.py:331`, which calls `maybe_expand()` directly and does
work — see A.4).

### A.3 `detect_only` commits an unfitted duplicate — DRIFTS

**[CODE]** `atlas/loop.py:152-170`:

```python
elif cfg.expansion_mode == "detect_only":
    expander.record(best_umf, encoder_output, actions, proprio_ctxt)
    if expander._strikes >= cfg.q and not library.is_full():
        best_idx = selected_idx
        new_chart = library.clone_from(best_idx)
        library.add(new_chart)
        ...
```

`library.clone_from()` (`atlas/library.py:76-85`) returns `chart.clone()`, a
deep copy with **identical weights** (`atlas/chart.py:131-137`). No
`_fit_candidate()` call, no gradient step. The committed chart is byte-identical
to its parent until a later `atlas_refine` moves it.

**[INFERENCE]** Two consequences. (1) The proposal's detect-only arm is
described as "persistent deficit → immediately commit a chart (the field's
convention)" (`ATLAS_proposal_v7.md` §7 E3 table) — the field's convention is
to *spawn and train* a module, not to duplicate an existing one. (2) Because
the duplicate scores identically to its parent, UMF ties, and `route()`'s
`min()` (`atlas/router.py:83`) resolves the tie to the lowest index, so the
new chart is never selected and never refined — it is inert library mass.
The E3 metric "charts committed" would count them, but they do nothing.

**Threatens:** **RQ3**, **L-1**, **C2**.

### A.4 The expansion path that *does* work is E2's, not ATLAS's

**[CODE]** `scripts/run_e2.py:331` calls `expander.maybe_expand(...)` directly
with real next-chunk data. That is the path that produced the three committed
charts recorded as **N9** (`atlas_out/e2_R2_cellB_q1`). It bypasses
`atlas_step()` entirely.

**[INFERENCE]** N9's claim that "the verification-gated expansion mechanism
has been demonstrated to fire correctly" is true of `Expander.maybe_expand()`
and false of the ATLAS controller that the paper describes. Those are
different claims and the distinction is load-bearing: the demonstrated path
is never the one E4 would run.

**Verdict for item A: DRIFTS.**

---

## B. Hyperparameters

**Specification:** implementation plan §7.7, `CLAUDE.md` §1.7.

| Symbol | Spec | Where the literal lives in code | Verdict |
|---|---:|---|---|
| `τ` | 0.5 | `atlas/loop.py:38`, `atlas/expand.py:37`, `scripts/run_e4.py:203`, `scripts/smoke_e4.py:123`, `configs/atlas/default.yaml:5`, `modal/modal_e0_planning.py:387` | MATCHES |
| `q` | 3 | `atlas/loop.py:39`, `atlas/expand.py:38`, `scripts/run_e4.py:203`, `scripts/smoke_e4.py:123`, `configs/atlas/default.yaml:6`, `modal/modal_e0_planning.py:438` | MATCHES as default |
| `m` | 0.05 | `atlas/loop.py:40`, `atlas/router.py:36`, `scripts/run_e1.py:90`, `scripts/run_e4.py:82`, `scripts/smoke_e4.py:123`, `scripts/make_e2_figure.py:31`, `configs/atlas/default.yaml:7` | value MATCHES, **semantics DRIFT** — see B.1 |
| `n_probe` | 20 | `atlas/loop.py:42`, `atlas/expand.py:39`, `scripts/run_e4.py:203`, `configs/atlas/default.yaml:8` | MATCHES; **`scripts/smoke_e4.py:124` uses `n_probe=3`** (smoke only, disclosed in its own docstring) |
| `K_max` | 10 | `atlas/loop.py:44`, `atlas/library.py:26`, `scripts/run_e4.py:204`, `configs/atlas/default.yaml:9` | MATCHES |
| chart lr | 5e-4 | `atlas/loop.py:41`, `atlas/loop.py:181`, `atlas/expand.py:40`, `atlas/adajepa.py:62`, `scripts/run_e0.py:509`, `scripts/run_e4.py:203`, `configs/atlas/default.yaml:10/14` | MATCHES everywhere |
| refine steps | 1 | `atlas/loop.py:220-244` (one `optimizer.step()`); `atlas/adajepa.py:132-140` (one `step()` after summing per-buffer-item backwards) | MATCHES |
| min-motion gate | 10th pct of training displacement | `atlas/score.py:237-251` (`percentile: float = 10.0`); wired at `scripts/run_e0.py:675-680`, `scripts/run_e1.py:275`, `scripts/run_e4.py:182-190`, `scripts/smoke_e4.py:104-112` | MATCHES |

### B.1 The hysteresis margin `m` no longer means what the proposal says — DRIFTS

**[CODE]** `atlas/router.py:94-101`:

```python
current_score = valid.get(current_idx)
if current_score is not None:
    spread = max(valid.values()) - min(valid.values())
    relative_gap = (current_score - best_score) / spread if spread > 0 else 0.0
    if relative_gap < hysteresis:
        return current_idx, {"scores": scores, "gated": False}
```

The proposal (§6, `SELECT c* = argmin UMF, hysteresis margin m`) specifies an
**absolute** margin on the score. The code applies `m` to the **fraction of
this replan's own chart-to-chart spread**. The in-code comment
(`atlas/router.py:86-93`) discloses this and justifies it: an absolute 0.05 is
a no-op for the `e1` router (scores ~1e4) and dominant for `sdyn` (scores in
[-1,1]).

**[INFERENCE]** The justification is sound as engineering and is a genuine
scope change to the method as described. Worse, the normalisation makes `m`
**mathematically inert in a two-chart library**: with exactly two valid
scores, if the incumbent is the worse of the two then
`current_score == max` and `best_score == min`, so
`relative_gap = (max−min)/(max−min) = 1.0 ≥ 0.05` always. The router
therefore switches on *any* improvement, i.e. it is pure argmin with no
hysteresis at all.

**[CODE]** E2's main 2×2 library is exactly two charts —
`scripts/run_e2.py:66-76` (`build_library`: `{c0, chart_R}`). Only the
3-chart confusion-matrix diagnostic (`scripts/run_e2.py:80-96`) has a
library where hysteresis can bind.

**Threatens:** **N7** directly. N7 reports "post-hysteresis-fix Cell B: UMF
0.833 vs S-dyn 0.570, down from the pre-fix +55.6pp" — i.e. the reported
change is attributed to a hysteresis fix that, in that cell's 2-chart
library, cannot have any effect through the margin. The number changed for
some other reason (most plausibly the *sequential `current_idx` carry-forward*
also introduced at `scripts/run_e2.py:287-291`, which is a separate change
bundled under the same name). Also threatens **C1** and **N6** to the extent
they are described as using the pre-registered `m`.

**Verdict for item B: MATCHES on every literal value; DRIFTS on the semantics
of `m`, in a way that has already propagated into a reported result (N7).**

### B.2 Documented diagnostic overrides (disclosed, not silent)

**[CODE]** `scripts/run_e2.py:142-152` exposes `--probe-q` and `--probe-tau`
with help text explicitly labelling them "DIAGNOSTIC OVERRIDE of CLAUDE.md
1.7's fixed q=3 / tau=0.5", and `scripts/run_e2.py:200` prints a
`[DIAGNOSTIC RUN]` banner when they deviate. `atlas_out/e2_R2_cellB_q1` and
`atlas_out/e2_R2_cellC_q1` are `q=1` runs. This is correctly disclosed
machinery, not a silent goalpost move; but any citation of N9 must carry
"at q=1, not the pre-registered q=3."

---

## C. The 7-arm ladder — the highest-value check

**Specification:** implementation plan §7.4 (arm table), §7.6 (baseline
configurations: "ATLAS-*: same loss, lr, optimiser, **buffer size** as
AdaJEPA — only the library/routing/expansion differ"), proposal §7 E4 ("each
rung adds exactly one mechanism, all sharing the same adaptation surface,
loss and optimiser").

Implementation: `scripts/run_e4.py`, `atlas/harness_e4.py::build_arm_state`
(`:65-130`) and `::run_e4_episode` (`:140-353`), `atlas/adajepa.py`.

### Adjacent-pair analysis

| Pair | Spec says the only difference is | What the code actually differs by | Verdict |
|---|---|---|---|
| 1→2 | adapts | adapts (`AdaJEPA` constructed, `refine()` called) | **MATCHES** |
| 2→3 | persists (per-episode re-init) | **nothing** | **DRIFTS — C.1** |
| 3→4 | library + routing | library + routing **and** buffer size 5→1 **and** c0-never-refined **and** a pre-trained E0 chart | **DRIFTS — C.2** |
| 4→5 | expands | expands (unfitted duplicate — see A.3) | DRIFTS (A.3) |
| 5→6 | verifies | arm 5 expands, arm 6 **cannot expand at all** (see A.2) | **DRIFTS — A.2** |

### C.1 Arms 2 and 3 are behaviourally identical — DRIFTS

**[CODE]** `atlas/adajepa.py:94-104`:

```python
def reset(self) -> None:
    if self.variant == "adajepa":
        self.predictor.load_state_dict(self.pretrained_state, strict=False)
        self._buffer.clear()
        self._optimizer = optim.Adam(self._params, lr=self.lr)
```

`variant` is the *only* thing that differs between arm 2 and arm 3
(`atlas/harness_e4.py:100-105`), and it is consulted *only* inside `reset()`.

**[CODE]** `grep -rn "adapter.reset\|\.reset()"` over `scripts/run_e4.py`,
`atlas/harness_e4.py` and `modal/modal_e4.py` returns **nothing**. The only
call site in the repository is `scripts/smoke_e4.py:142`, in the smoke
harness — which calls it from the driver loop, not from
`run_e4_episode()`.

**[INFERENCE]** In a production `run_e4.py` run, `AdaJEPA.reset()` is never
called. Arm 2 (`adajepa`) therefore never re-initialises, which makes it
**bit-identically the same method as arm 3 (`adajepa_persist`)**. Given
paired seeding, the two arms should produce identical episode records. The
rung labelled "*(ours)*" in the proposal's central table — the one whose
whole contribution is persistence — is compared against a copy of itself.

**[DOC]** `E3_E4_IMPLEMENTATION_PLAN.md:292-299`'s per-arm table specifies
`adapter.reset()` in the "reset per episode" column for arm 2 and "none" for
arm 3. The implementation dropped the arm-2 half.

**[INFERENCE]** This is also why the smoke test did not catch it:
`scripts/smoke_e4.py:141-149` calls `state.adapter.reset()` itself and then
asserts the predictor matches `pretrained_state`. The assertion passes
because the *smoke driver* performed the reset — it validates
`AdaJEPA.reset()`'s implementation, not that anything calls it.

**Threatens:** **L-1** (the ladder's 2→3 rung measures zero), **RQ4** (the
"paired Δ < 0 expected for Persistent-AdaJEPA vs > 0 for ATLAS" prediction in
plan §7.4's T2 table needs a genuine non-persistent baseline), **C4**,
**P-3**.

### C.2 Arms 3 and 4 differ by at least four things — DRIFTS

**[CODE]** Arm 3's refinement, `atlas/adajepa.py:119-141`, loops over a
`deque(maxlen=5)` buffer (`atlas/adajepa.py:55, 88`) and backwards each item
before one `optimizer.step()`. Arm 4's refinement, `atlas/loop.py:220-244`
via `atlas/harness_e4.py:312-315`, sees **exactly one chunk** — the
just-executed one. There is no buffer.

Implementation plan §7.6 states the requirement verbatim: "ATLAS-*: same
loss, lr, optimiser, **buffer size** as AdaJEPA — only the library/routing/
expansion differ." The buffer size is 5 for arms 2/3 and 1 for arms 4/5/6.

**[CODE]** Second difference: `atlas/harness_e4.py:305` guards refinement
with `and state.current_idx != 0`. When the router selects `c0`, arms 4/5/6
perform **no adaptation at all** for that replan. Arms 2/3 always adapt. So
"adapts" is not held constant across the 3→4 rung; it is conditional on a
routing decision.

**[CODE]** Third difference: `atlas/harness_e4.py:125-127` with
`expansion_start_library="full"` (the default, `scripts/run_e4.py:136`) loads
`chart_{kind}_{regime_b}.pt` — an E0 chart trained offline for up to 2000
Adam steps on 20–100 regime-B trajectories — into arms 4/5/6's library. Arms
2/3 start from pretrained weights with no such prior. The rung labelled
"library+routing" also silently adds a large quantity of offline supervised
adaptation to regime B that the lower rungs never receive.

**[INFERENCE]** This third one is arguably *specified* (plan §7.4 defines arm
4 as "ATLAS-fixed-library … charts from E0 only"), so it is a design-level
confound rather than a code drift — but the paper cannot describe the 3→4
delta as "the effect of adding a library and routing" when it is also the
effect of adding offline pre-training on the target regime. `E3_E4_IMPLEMENTATION_PLAN.md:305-320`
(§2b) flags exactly this and says "ask the user which to report before the
real run"; that decision has not been made.

**Threatens:** **L-1** (this is the pair the "attributes gain to a mechanism"
argument most needs), **RQ4**, **C4**.

**Verdict for item C: DRIFTS. Three of the five adjacent pairs fail the
one-mechanism rule and one pair (2→3) is behaviourally a null comparison.
As the code stands, the central table attributes nothing.**

---

## D. Disjoint parameter sets (claim P-1)

**Specification:** proposal §1 and §6 — "Charts are disjoint parameter sets,
so updating one **cannot** alter another's parameters."

### D.1 P-1 as stated — MATCHES, with a scope condition

**[CODE]** Every chart owns cloned tensors, never shared references:
`atlas/chart.py:64-66` (`p.data.clone()` at construction),
`atlas/chart.py:136` (`{k: v.clone()}` in `clone()`),
`atlas/chart.py:166` (`torch.load` result in `load()`).
`atlas/library.py:76-85` (`clone_from` → `chart.clone()`).

**[CODE]** `apply_` (`atlas/chart.py:102-105`) does
`state[name].data.copy_(value)` over the chart's full `_param_names` set. For
`ln_act` and `full`, all charts of the same kind select the *same* name set
(`atlas/chart.py:198-238` is a deterministic function of the predictor's
architecture, not of the chart), so applying chart B fully overwrites
whatever chart A left behind.

**[CODE]** Refinement writes only into the chart's own tensors:
`atlas/chart.py:172-190` (`update_from_predictor_`) copies out of the
predictor into `self._params` only for names in `self._param_names`.

**[INFERENCE]** P-1 holds **provided every chart in a library is the same
kind.** No production path builds a mixed-kind library
(`scripts/run_e4.py:135` and `scripts/run_e2.py:66-96` construct libraries
from a single `--kind`). If one ever did, `apply_` would not overwrite the
previous chart's parameters and P-1 would fail silently. Nothing in the code
enforces or asserts the same-kind invariant.

### D.2 The `HANDOFF.md` §4 allegation — CONFIRMED

The allegation, verbatim from `HANDOFF.md` §4: "`chart.restore_()` does NOT
restore the pretrained weights — for every kind except `lora4` it is
literally `self.apply_(predictor)` (`atlas/chart.py:107-127`), i.e. it
re-applies *that same chart*."

**[CODE]** `atlas/chart.py:107-127`:

```python
def restore_(self, predictor: nn.Module) -> None:
    if self.kind == "lora4":
        ... parametrize.remove_parametrizations(mod, attr, leave_parametrized=False)
    else:
        self.apply_(predictor)
```

**The allegation is correct.** For `ln_act` and `full`, `restore_()` is
functionally a no-op after `apply_()`.

**[CODE]** The call sites, enumerated by grep over `atlas/` and `scripts/`
(15 total, of which these are the production ones):
`atlas/score.py:99`, `atlas/router.py:159`, `atlas/router.py:186`,
`atlas/harness.py:241`, `atlas/harness.py:371`, `atlas/harness_e4.py:252`,
`atlas/loop.py:247`, `atlas/expand.py:230`, `scripts/run_e0.py:438`,
`scripts/diagnose_umf_locality.py:189`.

**Consequence, stated precisely:**

1. **Not a numerical error today.** Every one of those sites is followed by
   another `apply_()` of a same-kind chart before the predictor is used, so
   no stale weights survive into a computation. Verified by reading each
   site: `score.umf` and both router scorers are inside a per-chart loop that
   re-applies on the next iteration; `harness_e4.py:252` is followed by the
   next replan's `apply_` at `:245`; `run_e0.py:438` is followed by
   `wm.predictor.load_state_dict(pristine_predictor_state)` at `:689`.

2. **`atlas/loop.py:247` carries a false comment**:
   `chart.restore_(predictor)  # restore predictor to chart's baseline weights`.
   It does not. After `atlas_refine`, the predictor holds the *refined*
   weights. Any future code that assumes pristine state after refinement
   would be silently wrong.

3. **Gate G1 does not test this.** `scripts/smoke_gates.py::gate_g1`
   (`:82-146`) builds a **freshly-constructed** chart (`Chart(predictor, kind)`,
   `:96`), applies it, restores it, and asserts the state dict is
   bit-identical to pristine. A fresh chart's `_params` *are* the pretrained
   values, so `restore_() == apply_()` is trivially bit-identical. **G1 passes
   for a reason unrelated to what it claims to check.** It also never tests
   `kind="full"` (`:96` iterates `("ln_act", "lora4")` only).

**[DOC]** `EXPERIMENT_STATUS.md` §4 records G1 as "claimed passing since
2026-08-26." That claim is true and its scope is narrower than the gate's
stated purpose ("any error in the chart apply/restore path").

### D.3 A real breakage for `lora4` — DRIFTS

**[CODE]** After `apply_()` registers a LoRA parametrization
(`atlas/chart.py:96`), PyTorch's `parametrize` moves the base weight to
`...parametrizations.weight.original` and the base name disappears from
`predictor.named_parameters()`.

Both refinement paths select parameters by base name:

- `atlas/loop.py:223`: `params = [p for n, p in predictor.named_parameters() if n in chart._param_names]`
- `atlas/expand.py:201`: identical construction
- `atlas/harness_e4.py:308-309`: identical construction

**[INFERENCE]** For `kind="lora4"` these lists are **empty**, and
`torch.optim.Adam([])` raises `ValueError: optimizer got an empty parameter
list` (verified against this environment's torch). `atlas/harness.py:112-126`
is the only refinement path that handles this correctly — it selects by
`p.requires_grad` after explicitly enabling `lora_A`/`lora_B`
(`atlas/harness.py:118-125`). So E0 offline training works for `lora4` and
online ATLAS refinement does not.

**Threatens:** **RQ0** only indirectly (E0's `lora4` path is fine), but it
means the E0 winner-kind mechanism is not actually usable downstream if the
winner were `lora4`. `scripts/run_e4.py:135` and `configs/atlas/e4.yaml`
default to `ln_act`, so the live default avoids it.

**Verdict for item D: P-1 MATCHES as stated (same-kind libraries only). The
`HANDOFF.md` §4 allegation is CONFIRMED and G1 does not catch it. `lora4`
online refinement DRIFTS (raises).**

---

## E. Paired seeding (claim P-5) — MATCHES

**Specification:** implementation plan §7.4 — "episode `i` of segment `s`
uses seed `hash(s,i)` for **every** arm."

**[CODE]** `atlas/streams.py:39-51`:

```python
def paired_seed(segment_idx: int, episode_idx: int, arm: str = "") -> int:
    key = f"seg={segment_idx}:ep={episode_idx}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big")
```

The `arm` argument is accepted and deliberately unused. Confirmed by reading:
`arm` appears nowhere in the key.

**[CODE]** `atlas/streams.py:86-87` builds seeds as
`paired_seed(seg_idx + stream_seed_offset*1000, ep_idx + seed_run*10_000)`.
`get_stream` (`:98-119`) never passes a nonzero `stream_seed_offset`, so seed
independence across the 3 seed-runs comes from the `ep_idx + seed_run*10_000`
term. Distinct and deterministic. `scripts/run_e4.py:213` iterates
`streams[seed_run]` per arm — the same `EpisodeSpec` list object per
seed-run, so every arm sees the identical seed sequence.

**The seed determines the environment state, not merely an action sampler.**
**[CODE]** `atlas/harness_e4.py:176-184`:

```python
rs = np.random.RandomState(spec.seed)
init_state, goal_state = sample_dataset_init_goal(dataset_states, dataset_seq_lengths, rs)
...
goal_obs, _ = prepare_with_visual(base_env, regime, spec.seed, goal_state)
agent.set_goal(...)
obs, _ = prepare_with_visual(base_env, regime, spec.seed, init_state)
```

`sample_dataset_init_goal` (`scripts/run_e0_planning.py:152-215`) draws both
the init state and the goal state from the single `RandomState(spec.seed)`.
`prepare_with_visual` (`scripts/run_e0_planning.py:113-124`) does
`base_env.seed(seed); base_env.reset_to_state = state; return regime.reset()`
— the env is forcibly reset **to that exact state**, so the initial state and
the goal are fully determined by the seed, for every arm. **P-5 verified.**

**[CODE]** The router RNG is also paired:
`scripts/run_e4.py:251` derives `router_rng_seed` from `spec.seed` only.

**One gap, not a drift in the code:** plan §8 requires 20 ep × 6 segments × 3
seeds = **360 paired episodes per arm**. `scripts/run_e4.py:125,127` defaults
to `--episodes 20 --seeds 3` (= 360, MATCHES), but
`configs/atlas/e4.yaml:8-9` sets `episodes_per_segment: 10, seeds: 1` (= 60).
**[CODE]** Nothing in the repository loads any file under `configs/` —
`grep -rn "configs/atlas\|OmegaConf.load\|yaml.safe_load\|hydra"` over
`scripts/`, `atlas/` and `modal/` returns **zero hits**. The YAMLs are
documentation only. So the effective defaults are the script's (360), but the
YAML that a reader would consult as the config of record disagrees with them
by 6×. See K.9.

**Verdict for item E: MATCHES.**

---

## F. Single shared substrate and frozen backbone (claim P-3)

### F.1 Frozen DINOv2 encoder — MATCHES

**[CODE]** Every entry point freezes the encoder:
`scripts/run_e0.py:573-574`, `scripts/run_e0_planning.py:449-450`,
`scripts/run_e1.py:259-260`, `scripts/run_e2.py:174-175`,
`scripts/run_e4.py:165-166`, `scripts/run_e5.py:37-38`,
`scripts/smoke_e1.py:76-77`, `scripts/smoke_e4.py:97-98`,
`scripts/profile_episode.py:69-70`, `scripts/diagnose_cem_costs.py:160`,
`scripts/diagnose_umf_locality.py:125`.

**[CODE]** No optimizer anywhere is constructed over encoder parameters. Every
optimizer selects from `predictor.named_parameters()`:
`atlas/harness.py:125`, `atlas/loop.py:223`, `atlas/expand.py:201`,
`atlas/adajepa.py:89`, `atlas/harness_e4.py:308`. Backprop cannot reach the
ViT encoder. **Claim P-3's frozen-backbone half is verified.**

**[CODE]** `atlas/harness.py:117-123` additionally sets `requires_grad_(False)`
on every non-chart predictor parameter for E0 training, so `ln_act` and
`lora4` fine-tunes touch only their own surface. For `kind="full"` all
20,800,884 predictor parameters are trained — by design (`atlas/chart.py:13`),
though note `CLAUDE.md` §1.5's "~10k floats swapped" description does not
survive the `full` kind.

### F.2 Same substrate for the AdaJEPA arms — MATCHES

**[CODE]** `atlas/harness_e4.py:102-103`:

```python
param_names = Chart(predictor, kind)._param_names
adapter = AdaJEPA(world_model, param_names, variant=variant, lr=cfg.lr)
```

Arms 2/3 adapt **exactly** the parameter set arms 4/5/6's charts cover, on the
same `world_model` object, with the same loss
(`atlas/adajepa.py:135-137` vs `atlas/loop.py:238-240` — both
`(z_preds - enc_out[1:]).pow(2).mean(dim=-1).mean()` over
`_open_loop_rollout`), same optimiser (Adam), same lr. Same `GC_Agent`
instance (`scripts/run_e4.py:193`, constructed once outside the arm loop).
**Substrate parity verified.**

### F.3 Whether an ATLAS arm can adapt at all depends on which arms ran before it — DRIFTS

**[CODE]** `scripts/run_e4.py:165-168` freezes **both** encoder and predictor:

```python
for p in wm.encoder.parameters():
    p.requires_grad_(False)
for p in wm.predictor.parameters():
    p.requires_grad_(False)
```

(`scripts/smoke_e4.py:97-100` does the same.)

**[CODE]** `atlas/adajepa.py:89-92` re-enables gradients for its own surface:

```python
self._params = [p for n, p in self.predictor.named_parameters() if n in self.param_names]
for p in self._params:
    p.requires_grad_(True)
self._optimizer = optim.Adam(self._params, lr=lr)
```

**[CODE]** `atlas/loop.py::atlas_refine` (`:220-231`) and
`atlas/expand.py::_fit_candidate` (`:199-202`) and
`atlas/harness_e4.py:307-311` do **not** re-enable `requires_grad`. They build
an Adam over the frozen tensors and call `loss.backward()`.

**[CODE]** Verified against this environment's torch: Adam silently accepts
parameters with `requires_grad=False`; the loss then has
`requires_grad=False` and `loss.backward()` raises
`RuntimeError: element 0 of tensors does not require grad and does not have a
grad_fn`.

**[CODE]** `wm.predictor.load_state_dict(pristine_predictor_state)`
(`scripts/run_e4.py:236`) restores *values*, not `requires_grad` flags.

**[INFERENCE]** Consequences:

1. In a default `--arms` run, `ALL_ARMS` order (`scripts/run_e4.py:65-68`)
   puts `adajepa` at index 1, so by the time `atlas_fixed` runs at index 3
   the `ln_act` parameters have been permanently flipped to
   `requires_grad=True` by arm 2's `AdaJEPA.__init__`. Arms 4/5/6 then work —
   **only as a side effect of a different arm having run earlier in the same
   process.**
2. Running an ATLAS arm alone (`--arms atlas`) raises at the first refine.
3. **`--profile` mode is broken.** `scripts/run_e4.py:209-211` sets
   `args.arms = ["atlas"]`, so the budget-calibration run the implementation
   plan §7.0 instructs you to do before spending GPU-hours would crash as
   soon as the router selects a non-`c0` chart. It has evidently never been
   run — `atlas_out/` contains no `e4` directory of any kind.
4. `scripts/smoke_e4.py:132` iterates `RUN_E4_ARMS` in the same order, so the
   smoke test masks the bug for the same reason.

**Threatens:** **P-3** (arms are not independently reproducible — an arm's
behaviour depends on execution order through shared mutable module state),
**L-1**, **RQ3**, **RQ4**, and the credibility of any future E4 result run
with a non-default `--arms` subset or in per-arm Modal containers
(`configs/atlas/e4.yaml:34` explicitly proposes "one Modal container per
(arm, seed_run)" — which would put every ATLAS arm in a process with no
AdaJEPA arm in it, i.e. would crash).

**Verdict for item F: MATCHES on the frozen backbone and substrate parity;
DRIFTS on arm isolation.**

---

## G. Planner configuration

**Specification:** plan §7.0 originally `CEM 200×10, horizon 25, 5 executed
actions/replan, ≤30 MPC steps`; superseded by §7.0a (2026-08-24, T6) to
`num_samples=300, iterations=30, num_elites=10, horizon=6,
num_act_stepped=6, var_scale=1.0, frameskip=5` → 30 raw steps/episode, one
replan.

### G.1 Exhaustive inventory of what is actually in the code

| File | `num_samples` | `iterations` | `horizon` | `num_act_stepped` | `max_mpc_steps` | `num_elites` | `var_scale` | `frameskip` |
|---|---:|---:|---:|---:|---:|---|---|---:|
| `scripts/run_e0_planning.py` (argparse `:387-397`) | 300 | 30 | 6 | **6** | 30 (`--max-steps`) | `min(10, N)` `:101` | 1.0 `:103` | 5 `:65` |
| `scripts/run_e1.py` (`:84-89`) | 300 | 30 | 6 | **6** | 30 | `min(10, N)` `:103` | 1.0 `:105` | 5 `:83` |
| `scripts/run_e2.py` | — | — | — | — | — | — | — | — (no planner at all; `:388`, `:484` record `"planner": "none — collected trajectories"`) |
| `scripts/run_e4.py` (`:70-81`) | 300 | **30** | 6 | **1** | 30 | `min(10, N)` `:94` | 1.0 `:96` | 5 `:70` |
| `modal/modal_e0_planning.py` (`:80, :168, :244, :284`) | (passes through) | (passes through) | (passes through) | **6** | — | — | — | — |
| `modal/modal_e4.py` (`:86-87, :146`) | (passes through) | (passes through) | (passes through) | **1** | 30 | — | — | — |
| `configs/atlas/default.yaml` (`:27-31`) | 300 | 30 | 6 | **1** | 30 | — | — | — |
| `configs/atlas/e4.yaml` (`:39-43`) | 300 | **10** | 6 | **1** | 30 | — | — | — |
| `scripts/smoke_e4.py:115` | 2 | 2 | 2 | 1 | 10 | — | — | 5 |
| `scripts/run_e0.py` `closed_loop` collector (`:614-618`) | **100** (`--collect-num-samples`, `:532`) | **10** (`--collect-iterations`, `:539`) | 6 | **1** | — | — | — | 5 |

### G.2 Inconsistencies across scripts — DRIFTS

1. **`num_act_stepped` is experiment-dependent: 6 for E0-planning and E1, 1
   for E4.** `scripts/run_e4.py:74-80` documents the deviation and its reason
   (at nas=6 there is one replan per episode, which cannot exercise routing,
   refinement, or next-chunk verification). The reasoning is correct. The
   consequence is that **E4's planner is not the same planner E0's numbers
   were produced under**, so no E0 number can be used as a baseline for an
   E4 number, and the "1 replan per episode" open-loop protocol that produced
   the null result N1 is not the protocol under which the method would be
   evaluated. This is a documented deviation in plan §7.0a's style; it is not
   *arm*-dependent (all seven E4 arms share one `GC_Agent`,
   `scripts/run_e4.py:193`), so §1's single-substrate non-negotiable is not
   violated. It **is** experiment-dependent, which is a comparability
   problem, not a validity problem within one experiment.

2. **`iterations` disagrees between the script and its own config file.**
   `scripts/run_e4.py:72` = 30; `configs/atlas/e4.yaml:40` = 10 ("cut from 30
   — validate against frozen@R0 first"). Since nothing loads the YAML (see
   E), the effective value is 30 and the documented budget calibration in
   `configs/atlas/e4.yaml:20-38` (which computes ~35 GPU-h at iterations=10)
   does not describe what `run_e4.py` would actually do (~3× that).

3. **`configs/atlas/default.yaml:30` records `executed_actions: 1`** as the
   project-wide default, which is E4's deviation, not E0/E1's value. A reader
   consulting the config for "the planner config" gets E4's.

### G.3 A `steps_left` unit error in the multi-replan path — DRIFTS

**[CODE]** `scripts/run_e4.py:154`:
`n_replans_target = max(args.max_mpc_steps // args.num_act_stepped, 1)`
→ `30 // 1 = 30`.

But at `num_act_stepped=1` each replan executes `1 × frameskip = 5` raw
steps, and the episode is capped at `max_raw_steps = 30`
(`atlas/harness_e4.py:206`), so **only 6 replans can ever occur.**
`n_replans_target` should be `max_raw_steps // (num_act_stepped * frameskip)`.

**[CODE]** The stale value is then fed to the planner:
`atlas/harness_e4.py:248`
`steps_left_model = (n_replans_target - replan_idx) * num_act_stepped`
→ `30` at replan 0, where the true remaining model-chunk budget is `6`.
`CEMPlanner.plan()` computes `plan_length = min(self.horizon, steps_left)`,
so the planner is told it has 5× more budget than it has. The same
construction is at `scripts/run_e0_planning.py:265,286` and
`scripts/run_e1.py:238`, where at `nas=6` it happens to be right
(`30 // 6 = 5` replans × 6 = 30 model steps... which is itself 150 raw steps
against a 30-raw-step cap — the same 5× overstatement, masked because
`horizon=6` clamps it).

**[INFERENCE]** With `horizon=6`, `min(6, 30)` and `min(6, 6)` are both 6, so
this error is **currently inert** at the first replan and only bites for the
last replan of an episode, where the planner should be shortening its plan
and does not. Low severity, but it is a real unit error in a line the E3/E4
plan explicitly warned about (`E3_E4_IMPLEMENTATION_PLAN.md:384-386`:
"`steps_left` is in model-chunk units — do not multiply by frameskip"; the
warning addressed the wrong half of the conversion).

### G.4 Scientific-validity observation, stated plainly

At `num_act_stepped=6` — the configuration under which **every planning number
currently on disk** was produced (`atlas_out/e0_planning*`, `e0_v3_planning_*`,
`e0_v4_planning_*`, `e0_v5_planning_*`) — one CEM search covers
`6 × frameskip(5) = 30` raw steps, which is the entire episode. There is
**exactly one plan per episode and no replanning.** The agent commits to a
30-step open-loop action sequence chosen before it has observed a single
consequence of its own actions, and executes it to completion regardless of
what happens.

For an experiment about **adaptive world models**, this is structurally
disabling in three specific ways:

1. **A better world model cannot express itself as better behaviour.** The
   only channel by which improved dynamics prediction can improve outcome is
   a better one-shot plan. There is no feedback loop for the model to correct
   an error it made, which is precisely the capability a regime-adapted model
   should buy.
2. **It makes N1's null uninterpretable as evidence about the method.** A
   −1.0pp difference between chart and baseline at N=100 (claim **N1**) is a
   well-powered null *about one-shot open-loop planning*, not about adaptive
   world modelling. Claim **N5** (closed-loop, `nas=2`, 3 replans, +10.0pp,
   N=20, p=0.625) points the other way and is the only measurement taken
   under a protocol where adaptation could matter — and it is underpowered by
   a factor of five.
3. **It is causally upstream of N3.** Under R2 the planner's own cost ranking
   is near-zero-correlated with true outcome and its converged plan lands
   *farther* from the goal than the episode started (**N3**, **N3b**). Under
   a one-shot protocol, that ranking failure is the entire outcome. Under a
   replanning protocol it would be partially recoverable. N3 and the
   `nas=6` protocol are not independent findings.

The project's own code already agrees with this reading:
`scripts/run_e4.py:74-79` says nas=6 "structurally cannot exercise routing,
refinement, or Expander's next-chunk verification," and
`scripts/run_e0.py:294-300` says the `closed_loop` collector "MUST be built
with num_act_stepped=1: at the eval config's nas=6 a single plan covers the
whole trajectory open-loop." Both statements are correct and both apply with
equal force to the evaluation protocol that produced N1 and N4.

**Threatens:** **N1**, **N4**, **N5**, **RQ0**, **G-1**, and the framing of
the entire negative result.

**Verdict for item G: DRIFTS (experiment-dependent `num_act_stepped`;
script-vs-config `iterations` disagreement; `n_replans_target` unit error).
The one-replan protocol is a design decision, not a bug, and is recorded here
as a scientific-validity finding.**

---

## H. Chart training protocol

**Specification:** plan §7.1 — `{ln_act, lora4, full} × {R1, R2}` = 6 offline
fine-tunes, ~2000 steps each, Adam, predictor lr 5e-4.

### H.1 What the code actually does

**[CODE]** `scripts/run_e0.py` argparse defaults:

| Knob | Default | Line |
|---|---:|---|
| `--steps` | 2000 (a **maximum**, early stopping can cut it) | `:471-473` |
| `--num-train-trajs` | 20 | `:474` |
| `--train-traj-len` | 25 raw steps (= 5 model steps) | `:482` |
| `--num-val-trajs` | 8 | `:489` |
| `--eval-traj-len` | 50 raw steps (= 10 model steps) | `:494` |
| `--eval-every` | 25 | `:501` |
| `--patience` | 5 | `:504` |
| `--lr` | 5e-4 | `:509` |
| `--data-source` | `dataset` | `:511` |

**[CODE]** Early stopping is real: `atlas/harness.py:151-156` tracks
`best_val_loss` / `best_params` / `checks_since_improvement`; the returned
chart is the best-validation snapshot, not the final step's weights
(`atlas/harness.py:100-102` docstring). So "2000 SGD steps" is a ceiling. The
actual step count per chart is not recorded in `results.json`
(`atlas_out/*/results.json` carry only `train_loss`, `eval_loss`, `eval_umf`,
`params`, `status`).

**[CODE]** Training is not minibatched: `atlas/harness.py:158+` loops over
**all** trajectories every step. With `--num-train-trajs 20 --train-traj-len 25`,
"2000 steps" means 2000 full passes over 20 trajectories of 5 model
transitions each — i.e. 100 model transitions, seen 2000 times.

### H.2 The train/validation split is 2-way and reported as if 3-way — DRIFTS

**[CODE]** `scripts/run_e0.py:489-493`, the `--num-val-trajs` help text, states
this outright:

```
"Number of held-out validation trajectories, used both for early
 stopping during training and (necessarily reusing the same set --
 this is not a 3-way train/val/test split) for the final reported
 eval_loss/eval_umf."
```

**[CODE]** Confirmed in the flow: `val_trajectories`
(`scripts/run_e0.py:645-648`) is passed both to `run_e0_finetune`'s
`_val_loss()` early-stopping check (`atlas/harness.py:131-150`) *and* to
`evaluate_e0_chart(...)` (`scripts/run_e0.py:406`, `:435-438`) whose output
is the reported `eval_umf`.

**[INFERENCE]** Every reported `eval_umf` in `atlas_out/*/results.json` is
measured on the set that selected the checkpoint. With `--patience 5
--eval-every 25`, model selection consults that set up to 80 times over a
2000-step run. The reported UMF is therefore **optimistically biased** by an
unknown amount. This affects the ΔUMF column of the RQ0 capacity table and,
critically, claim **N4** — "more training data monotonically improves UMF
(0.336 → 0.302 → 0.268 at 20/60/100 trajectories)". `--num-val-trajs`
stayed at 8 across that sweep (`atlas_out/e0_train_sweep_60/e0_seed_manifest.json`
and `.../e0_train_sweep_100/...` both show `n_eval: 8`), so the selection
set is fixed at 8 trajectories while training data grows 5×; a monotone
improvement measured this way is partly a monotone improvement in
selection-set fit.

**Threatens:** **N4** directly, **RQ0**, **C3**.

### H.3 Adapter kinds were NOT trained at matched budgets — DRIFTS

**[CODE]** What exists on disk, per `atlas_out/`:

| Chart | Directory | `eval_umf` | Manifest present? |
|---|---|---:|---|
| `ln_act` × R2 | `e0_v3_dataset` | 0.3357 | **no** |
| `lora4` × R2 | `e0_v4_lora4` | 0.3286 | **no** |
| `full` × R2 | `e0_v4_full` | 0.7280 | **no** |
| `ln_act` × R2 (closed-loop data) | `e0_v5_closed_loop` | 0.4233 | yes (20 train / 8 eval, `closed_loop`) |
| `ln_act` × R1 | `e0_v6_R1` | 0.2845 | yes (20 train / 8 eval, `dataset`, friction 2.0) |
| `lora4` × R1 | `e0_v6_R1` | 0.2876 | yes (same manifest) |
| **`full` × R1** | — | — | **DOES NOT EXIST** |

**[CODE]** `e0_v3_dataset`, `e0_v4_lora4` and `e0_v4_full` contain **no
`e0_seed_manifest.json`**, so their training budgets cannot be recovered from
disk.

**[DOC]** `CLAUDE.md` §0.1 states that `lora4`×R1 was "confounded by an
OOM-forced smaller training budget, not yet re-run at parity."
`atlas/harness.py:163-170` documents the P2a per-trajectory-backward fix that
removed the OOM and explicitly names this: "this is why `lora4` OOM'd at the
same 20×25 budget `ln_act` used and had to be retrained at a smaller,
confounding 10×15." `e0_v6_R1` was produced after that fix and its manifest
shows both R1 kinds at 20/8, so the R1 pair is matched. **The R2 trio
(`e0_v3_dataset` / `e0_v4_lora4` / `e0_v4_full`) cannot be confirmed matched.**

**[INFERENCE]** Two conclusions, stated explicitly as the checklist requires:

1. **The pre-registered `{ln_act, lora4, full} × {R1, R2}` matrix is
   incomplete — `full` × R1 was never trained.** Plan §7.1's decision rule
   ("smallest kind reaching ≥90% of `full` on **both** metrics in **both**
   regimes") is therefore not merely inapplicable because `full`'s gain went
   negative (the reason the project gives); it is **undefined for R1 because
   the denominator was never computed.** This is a stronger and simpler
   statement than the one currently in the project's documents.
2. **Any capacity conclusion drawn from the R2 trio is confounded with
   training budget**, because the budgets are unrecorded and at least one
   kind is documented as having been trained at a reduced budget at some
   point. `full`'s conspicuously bad `eval_umf` (0.728, vs 0.329–0.336 for
   the other two) is exactly the shape a budget confound would take, and is
   also consistent with overfitting a 20.8M-parameter model on ~100 model
   transitions — a hypothesis the 2-way split (H.2) cannot distinguish from
   genuine capacity failure.

**Threatens:** **RQ0**, **N4**, and claim **S-4** ("E0 is closed as a negative
result").

**Verdict for item H: DRIFTS.**

---

## I. Training vs deployment action distribution

**[CODE]** `scripts/run_e0.py` offers four collectors
(`load_regime_trajectories`, `:74-395`; `--data-source`, `:511-528`):

| Source | What generates the actions | Reactive to the shifted physics? | Line |
|---|---|---|---|
| `dataset` (**default**) | Real recorded Push-T demo actions replayed step by step under the shifted regime | **No — open-loop.** The recording was made under R0. | `:261-275` |
| `hybrid` | Real init state + scripted aimed-walk driven by the live agent position | Yes, but corrections come from a scripted policy | `:317-357` |
| `scripted` | Synthetic aimed-walk from a random reset, `ACTION_GAIN=0.25` | Yes, scripted | `:317-357` |
| `closed_loop` | CEM planner replanning every model chunk against the live shifted state | **Yes — on-policy** | `:276-316` |

**[CODE]** Evaluation-time actions come from `GC_Agent`/`CEMPlanner` at
`num_act_stepped=6` (`scripts/run_e0_planning.py:265-300`) — one open-loop
30-raw-step plan.

### I.1 The default collector's distribution does not match deployment — DRIFTS

**[INFERENCE]** The chart that produced the headline null (N1) is
`ln_act`×R2 from `atlas_out/e0_v3_dataset`, trained on `--data-source dataset`
= replayed *expert human demonstration* actions. The evaluation distribution
is *CEM-planned* actions from a frozen DINOv2/ViT world model. These are
different action distributions on two axes at once: expert-vs-planner, and
"the actions that were optimal under R0" vs "the actions this planner
believes are optimal under R2."

**[CODE]** The code itself already reaches this conclusion:
`scripts/run_e0.py:520-528` says `closed_loop` is "The only source whose
trajectories contain the model's own overshoot AND its attempted correction;
'hybrid' is reactive but its corrections come from a scripted policy, not
from the model being adapted."

**[DOC]** `CLAUDE.md` §0.1 flags it as an "Open, unresolved concern:
real-data-replay training trajectories are open-loop (don't react to the
regime-shifted physics as they unfold)."

**Threatens:** **N1**, **N2**, **N4**, **RQ0**. A chart fit to explain expert
open-loop demonstrations under R2 is not obviously the chart that would help a
CEM planner rank its own candidate action sequences under R2 — which is
exactly the failure mode **N3** documents.

### I.2 Collection-time planner budget ≠ evaluation-time planner budget — DRIFTS

**[CODE]** `scripts/run_e0.py:614-618` builds the `closed_loop` collector's
agent with `build_cfg(args.collect_num_samples, args.collect_iterations,
horizon=6, num_act_stepped=1)` where the defaults are
`--collect-num-samples 100` (`:532`) and `--collect-iterations 10` (`:539`).
Evaluation uses 300 × 30 at `nas=6`.

So the on-policy collector collects data from a **100×10, replanning-every-
chunk** planner and the chart is then evaluated against a **300×30, one-shot**
planner. That is a 9× search-budget difference *and* an open-loop/closed-loop
difference, in the one collector that was built specifically to close the
train/deploy gap.

**[CODE]** The trade-off is disclosed at `scripts/run_e0.py:532-538`
("Deviation from the validated planner config — record it with any result").

**[INFERENCE]** Claim **S-5** (`OPUS_REMAINING_TASKS.md` item 10: the
`closed_loop` result was "tested with a broken instrument — four stacked
train/deploy mismatches") is consistent with what the code shows. This audit
independently confirms at least three of them: action-distribution mismatch
(I.1), search-budget mismatch (I.2), and open-vs-closed-loop mismatch (G.4).

**Verdict for item I: DRIFTS.**

---

## J. Regimes (claim P-4)

### J.1 Which values are in the code — DRIFTS from plan §6.1a

**[CODE]** `atlas/regimes.py:55-66` is the single live source of truth:

```python
REGIME_CONFIGS: dict[str, dict] = {
    "R0": {},
    "R1": {"friction": 2.0},   # friction saturates here (5.0 measured identical); was 0.8
    "R2": {"damping": 0.5},    # was elasticity 0.9 -- proven mechanically DEAD
}
```

**Resolution of the documented conflict:** plan §6.1a's table says R1 =
friction `0.0 → 0.8` and R2 = elasticity `0.0 → 0.9`. **Neither is what the
code does.** The code is R1 = friction **2.0** and R2 = **damping 0.5**, and
the in-code comments record both corrections. `configs/regimes/pusht.yaml`
agrees with the code (`:28`, `:39`) and states at `:3-5` that nothing loads it
— it is documentation only. `CLAIMS_MATRIX.md` P-4's "R1 = friction 2.0,
R2 = damping 0.5" is **correct**; plan §6.1a is the stale document.

**[CODE]** `atlas/regimes.py:69-73` exposes `set_regime_config()` for
calibration sweeps, threaded through `scripts/run_e0.py:460-470` and
`scripts/run_e0_planning.py:377-383` as `--regime-config`, and the resolved
values are logged into `e0_seed_manifest.json` (`scripts/run_e0.py:650`) and
the planning summary. Attribution is preserved. Confirmed present in the two
manifests that exist: `{'friction': 2.0}` for R1, `{'damping': 0.5}` for R2.

### J.2 Both shapes are set, as pymunk requires — MATCHES

**[CODE]** `atlas/regimes.py:126-135`:

```python
def _set_shape_property(self, name: str, value: float) -> None:
    for body in (self._get_agent(), self._get_block()):
        for shape in body.shapes:
            setattr(shape, name, value)
```

Both the agent's and the block's shapes are set, which is the non-obvious
requirement plan §6.1a calls out. **Correct.**

**[CODE]** For R2 the mechanism is `self.env.space.damping = 0.5`
(`atlas/regimes.py:124`) — a space-level property, so the agent/block issue
does not arise. Note the pusher is `KINEMATIC` and its velocity is set
explicitly each sim step, so damping affects the **block only** — the code
comment at `:120-123` says so.

### J.3 Survives `reset()` — MATCHES

**[CODE]** `atlas/regimes.py:99-102`:

```python
def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    self._apply_physics()
    return obs, info
```

Physics is re-applied after every reset, which is essential because
`PushTEnv._setup()` rebuilds the pymunk space and hardcodes `damping = 0`.
Every production path resets **through the wrapper**:
`scripts/run_e0_planning.py:113-124` (`prepare_with_visual` → `regime.reset()`),
`atlas/harness.py:279-281` (`_prepare_env`), and both are the functions
`atlas/harness_e4.py:164-166` imports and uses. `scripts/run_e0.py:255`
calls `env.reset()` on the wrapper.

**One asymmetry worth recording:** `_apply_physics` returns early for R0
(`atlas/regimes.py:106-107`). Since `scripts/run_e4.py:196-198` builds **one**
`base_env` and wraps it in two `PhysicsRegime` objects sharing it, R0's reset
relies on `PushTEnv._setup()` restoring `damping = 0` and shape friction to
the shipped `0.0`. **[INFERENCE]** That is what the code comment asserts and
is almost certainly true, but it is an unasserted invariant: if a future
`PushTEnv` did not rebuild shapes on reset, R2's damping or R1's friction
would leak into the next R0 segment of the S2 stream and silently destroy the
regime contrast. Gate **G4** — which is the gate designed to catch exactly
this — is the only gate never run (`EXPERIMENT_STATUS.md` §4).

### J.4 The planning loops step the base environment, bypassing the wrapper — latent hazard

**[CODE]** All three planning loops call `base_env.step(a)`, not the
wrapper's `step`:
`scripts/run_e0_planning.py:310`, `atlas/harness.py:381`,
`atlas/harness_e4.py:263`.

**[INFERENCE]** For `PhysicsRegime` this is currently benign: the wrapper has
no per-step logic; its whole effect is applied at reset and lives on the
pymunk space and shapes. **But it silently disables any wrapper-level
per-step behaviour**, `VisualCorruption` included. If E2 were ever run with a
planner, or E4 with an appearance shift (cell D's realistic condition), the
corruption would be applied to the reset observation and to nothing else —
every stepped observation would be uncorrupted, and the experiment would
silently measure the wrong thing.

**[CODE]** E2 is currently safe because it does **not** use the planning
loops: `scripts/run_e2.py` collects via
`scripts/run_e0.py::load_regime_trajectories`, which steps `env` — the
wrapper stack including `VisualCorruption` (`scripts/run_e0.py:216-224`,
`:262`, `:307`, `:355`). Verified.

**Threatens (prospectively):** **RQ2** cells C and D, and **P-4**, for any
future run that puts a corruption in a planning loop.

**Verdict for item J: DRIFTS (values differ from plan §6.1a; the code is
authoritative and internally consistent). Wrapper mechanics MATCH. The
`base_env.step` bypass is a latent hazard, recorded not fixed.**

---

## K. Documentation claims

`CLAUDE.md` was updated on 2026-08-27 by another agent in this audit with an
appended AUDIT NOTE listing seven drifts. Those are not repeated here except
where this pass **independently confirmed or extended** them. What follows is
this pass's own list, restricted to statements that can be checked against
code.

### K.1 How stale is `CLAUDE.md` §0.1 overall?

**Materially stale — it is two rounds of results and one full experiment
behind, and it misdescribes the implementation state of three scripts.** It
carries its own honest stamp ("Last checked: 2026-08-25") but is auto-loaded
into every session as current fact. Specific statements verified false or
unverifiable:

1. **"`run_e2.py`, `run_e4.py`, `run_e5.py` still `raise NotImplementedError`."**
   **[CODE] FALSE for two of three.** `scripts/run_e2.py` is 500 lines with a
   full `main()` (`:107+`) and has produced nine output directories.
   `scripts/run_e4.py` is 323 lines with `main()` at `:121` and a resume path
   at `:106-118`. Only `scripts/run_e5.py:45` still raises. (Independently
   confirms AUDIT NOTE item 1.)

2. **"`atlas.loop.atlas_step()`/`atlas_refine()` … are no longer known-broken
   if they were."** **[CODE] Not supportable.** This pass found three
   previously-unrecorded defects in exactly those functions: the unreachable
   expansion branch (A.2, `atlas/loop.py:140-150`), the unfitted detect-only
   clone (A.3, `atlas/loop.py:158-166`), and the missing `requires_grad`
   re-enable (F.3, `atlas/loop.py:220-231`). "Not known-broken" was accurate
   as a statement about what was known; it should not be read as a positive
   claim.

3. **"All available gates (G2, G3a, G3b, G5, G6) pass post-fix."**
   **[CODE] COULDN'T VERIFY** — this pass did not execute `smoke_gates.py`
   (running it would consume GPU and is outside a documentation pass). What
   *can* be said from reading: **G1's scope is narrower than its stated
   purpose** (see D.2) — it tests only unrefined charts and only
   `ln_act`/`lora4`, so "chart apply/restore bugs" as a class is not covered.
   **G3a/G3b exercise `Expander` directly** (`scripts/smoke_gates.py:311`,
   `:366`), never through `atlas_step()`, which is why the unreachable-branch
   bug (A.2) survived them. **G5's E4 analogue** (`scripts/smoke_e4.py:167-172`)
   checks `init_block_pos_diff` equality across arms — a valid pairing check.

4. **"`gate_g1` has the identical API bug, NOT yet fixed."**
   **[CODE] Stale.** `scripts/smoke_gates.py:82-146` is the rewritten
   headless version; its own docstring records the rewrite. (Confirms AUDIT
   NOTE item 3.)

5. **"Section 3's code layout"** omits `atlas/harness_e4.py`, which owns
   `build_arm_state` and `run_e4_episode` — i.e. the entire 7-arm ladder.
   (Confirms AUDIT NOTE item 4.)

### K.2 Claims in other documents that this pass could not confirm from code

6. **`E3_E4_IMPLEMENTATION_PLAN.md:380-382`** — "Keep a two-deep chunk buffer
   — chunk `k` is deficit data, chunk `k+1` is the held-out verification
   set." **[CODE] NOT IMPLEMENTED.** `atlas/harness_e4.py:202,318` keeps one
   chunk. See A.2.

7. **`E3_E4_IMPLEMENTATION_PLAN.md:292-299`** — the per-arm table requires
   `adapter.reset()` for arm 2. **[CODE] NOT IMPLEMENTED** in
   `scripts/run_e4.py` / `atlas/harness_e4.py`. See C.1.

8. **`atlas/harness_e4.py:198-202`'s own comment** describes the two-deep
   buffer as though it exists. **[CODE] FALSE.** A comment that describes a
   design rather than the code beneath it is the most dangerous kind of stale
   documentation, because it is read by whoever next edits that function.

9. **`scripts/smoke_e4.py:16-18`** claims among its assertions
   "atlas_detect / atlas can commit (mechanism reachable), not asserted to
   fire within this tiny budget." **[CODE] Half false.** `atlas_detect`'s
   mechanism is reachable; `atlas`'s is not (A.2). The smoke test's own
   docstring is what made the unreachable branch look tested.

10. **`configs/atlas/*.yaml` are not loaded by anything.**
    **[CODE]** `grep -rn "configs/atlas\|OmegaConf.load\|yaml.safe_load\|hydra"`
    over `scripts/`, `atlas/` and `modal/` → **zero hits**. Only
    `configs/regimes/pusht.yaml:3-5` states this about itself. The four
    `configs/atlas/*.yaml` files present themselves as configuration and are
    documentation, and two of them (`default.yaml:30`, `e4.yaml:40`) disagree
    with the scripts' actual defaults. See E and G.2.

11. **`README.md:181`** — "The experiment scripts raise `NotImplementedError`
    at the point where the [planner integration goes]." **[CODE] Stale** for
    E0/E1/E2/E4 for the same reason as K.1.1.

**Verdict for item K: DRIFTS. Eleven specific statements listed; `CLAUDE.md`
§0.1 is materially stale and is auto-loaded into every session.**

---

## What this pass verified vs. what it did not

**Verified from code (L2–L3 evidence about the implementation, not about any
result):** items A–K above, each at a cited `file:line`.

**Verified by execution:** one property only — that PyTorch raises on
`loss.backward()` when every parameter in the graph has
`requires_grad=False`, and that `Adam([])` raises `ValueError`. Run as a
four-line scratch snippet against this environment's torch, not against the
repository.

**NOT run:** `scripts/smoke_gates.py` (any gate), `scripts/smoke_e4.py`,
`scripts/run_e4.py`, and every other script. No GPU work was performed. Every
statement above about runtime behaviour that was not covered by the snippet
is marked **[INFERENCE]** and rests on reading the code.

**NOT audited here** (owned by the other audit agents): whether any number in
`atlas_out/` is correctly computed (results-auditor), whether the negative
result is novel (literature), and general code-correctness bugs outside the
proposal-alignment question (bughunter).

---

## Corrections to `research_audit/EXPERIMENT_STATUS.md`

The checklist asked for these to be stated here rather than edited in place.

1. **§3's table is too generous.** It marks all seven arms "Implemented: yes."
   Accurate rows would be: arm 2 **implemented but its distinguishing
   mechanism is never invoked** (C.1); arm 5 **implemented but commits
   unfitted duplicates** (A.3); arm 6 **implemented but its distinguishing
   mechanism is unreachable** (A.2). "Implemented" at L2 means "code exists
   and is reachable" — for arm 6's verification path, it is not reachable.

2. **§2's EXPAND row** says the mechanism "Fired for real once, in E2's q=1
   diagnostic — 3 charts committed through `Expander.record()` →
   `library.clone_from()` → `_fit_candidate()` → `library.add()`." Correct,
   and it should add: **that path was entered from `scripts/run_e2.py:331`,
   which calls `Expander.maybe_expand()` directly. The route through
   `atlas_step()` — the one E4 would use — has never fired and cannot.**

3. **§4's G1 row** should record that the rewritten G1 tests only
   *unrefined* charts and only `ln_act`/`lora4`, so it does not cover the
   `restore_()` behaviour `HANDOFF.md` §4 flags (D.2).

4. **§1's E0 row** should record that `full` × R1 was never trained, so plan
   §7.1's pre-registered rule is undefined for R1 independently of the
   negative-gain problem (H.3).

---

## Open questions this pass could not resolve

1. **What training budget produced `atlas_out/e0_v3_dataset`,
   `e0_v4_lora4` and `e0_v4_full`?** No `e0_seed_manifest.json` exists in
   those directories and `results.json` records no step count or trajectory
   count. **What would settle it:** the Modal run logs, or re-running with
   `--num-train-trajs`/`--train-traj-len` recorded. Until then the R2 capacity
   comparison is unfalsifiable as to budget matching.

2. **How many SGD steps did each chart actually take before early stopping?**
   `atlas/harness.py` tracks `stopped_early_at` internally (`:157`) but
   `results.json` does not carry it. **What would settle it:**
   `val_loss_{kind}_{regime}.json`, which exists for `e0_v6_R1` and the two
   train-sweep directories but not for the v3/v4 charts.

3. **Would arms 2 and 3 in fact produce byte-identical episode records?**
   This is a directly testable prediction of C.1 and would settle it in one
   short run. **What would settle it:** `python scripts/run_e4.py --arms
   adajepa adajepa_persist --episodes 2 --seeds 1` and a diff of the two
   arms' JSONL lines. This pass did not run it (no GPU work permitted).

4. **Does `PushTEnv._setup()` reliably restore `shape.friction` to 0.0 on
   reset?** J.3's R0-leak concern depends on it. **What would settle it:**
   gate **G4**, which is the only gate never run, or a five-line read of
   `hub/hub/facebookresearch_jepa-wms_main/evals/simu_env_planning/envs/pusht_env/pusht_env.py::_setup`.

5. **Which `expansion_start_library` mode is E3 to be reported under?**
   `E3_E4_IMPLEMENTATION_PLAN.md:305-320` says to ask the user before the
   real run. The default (`full`) makes correct ATLAS behaviour **0 commits**,
   which is indistinguishable from the A.2 bug's output. If E4 is ever run
   under the default without fixing A.2 first, the resulting "ATLAS committed
   0 charts" would look like a *success* and would be an artifact.

---

## The five drifts that most endanger the paper

Ordered by consequence. Each is a code-level fact with a locator, not an
impression.

### 1. The ATLAS arm cannot commit a chart. `atlas/loop.py:140-150` + `atlas/harness_e4.py:217`

`atlas_step()`'s verification-gated expansion is guarded on
`next_encoder_output is not None`, and the only production caller passes
`None` unconditionally. Arm 6 therefore records strikes forever and never
fires a probe. **Consequence:** RQ3's headline comparison — ATLAS commits ≈2
where detect-only commits >2 — cannot be measured; ATLAS's count is 0 by
construction. C2, the paper's secondary contribution, has no path to
empirical support. And because the intended default
(`--expansion-start-library full`) makes *correct* ATLAS behaviour also 0
commits, a run under the default would produce a plausible-looking result
that is entirely an artifact of the disabled code path. This is the single
most dangerous finding in this audit, because it fails silently and in the
direction of the hypothesis. Threatens **C2**, **RQ3**, **L-1**, **N9**'s
scope, **C2-probation**.

### 2. Arms 2 and 3 are the same method. `atlas/adajepa.py:94-104` + absence of any `reset()` call in `scripts/run_e4.py` / `atlas/harness_e4.py`

`AdaJEPA.reset()` — the sole implementation of the per-episode re-init that
*defines* the difference between AdaJEPA and Persistent-AdaJEPA — is called
from exactly one place in the repository, `scripts/smoke_e4.py:142`, which is
a test harness. In production, arm 2 never re-initialises. Given paired
seeding, arms 2 and 3 should produce identical episodes. **Consequence:** the
rung of the ladder labelled "*(ours)*" is compared against a copy of itself,
so the "persists" mechanism contributes a measured effect of exactly zero by
construction, and RQ4's central prediction (paired Δ < 0 for
Persistent-AdaJEPA, > 0 for ATLAS) loses its comparator. Threatens **L-1**,
**RQ4**, **C4**, **P-3**.

### 3. Arms 3 and 4 differ by four things, not one. `atlas/adajepa.py:55,88` (buffer 5) vs `atlas/loop.py:220-244` (buffer 1); `atlas/harness_e4.py:305` (`current_idx != 0`); `atlas/harness_e4.py:125-127` (pre-trained E0 chart)

Implementation plan §7.6 states the requirement in so many words — "same
loss, lr, optimiser, **buffer size** as AdaJEPA — only the library/routing/
expansion differ." The buffer size is 5 for arms 2/3 and 1 for arms 4/5/6;
arms 4/5/6 additionally skip refinement entirely whenever the router picks
`c0`; and arms 4/5/6 start with an offline-trained regime-B chart the lower
rungs never see. **Consequence:** the 3→4 delta — the pair the "attributes
gain to a mechanism" argument most needs — confounds routing with a 5× change
in adaptation data, a conditional adaptation rule, and a large quantity of
offline supervision on the target regime. Threatens **L-1**, **RQ4**, **C4**.

### 4. The evaluation protocol has one plan per episode and no feedback. `scripts/run_e0_planning.py:392` (`--num-act-stepped` default 6), `scripts/run_e1.py:87`

At `nas=6`, one CEM search covers all 30 raw steps. Every planning number on
disk was produced this way. The agent commits to a 30-step open-loop sequence
before observing any consequence of its own actions. **Consequence:** a better
world model has exactly one channel through which to become better behaviour,
and no channel at all for correcting its own error — which is the capability
adaptation is supposed to buy. N1's "well-powered null" is well-powered about
one-shot open-loop planning, not about adaptive world modelling; N5 (the only
closed-loop measurement, +10.0pp) points the other way at N=20; and N3's
cost-ranking degeneracy is not independent of the protocol, it is the whole
outcome under it. This is a design decision rather than a bug, and it is the
finding most likely to be raised by a reviewer of a *continual world models*
paper. Threatens **N1**, **N4**, **N5**, **RQ0**, **G-1**.

### 5. The reported E0 UMF is measured on the set that selected the checkpoint. `scripts/run_e0.py:489-493`, `atlas/harness.py:131-156`, `scripts/run_e0.py:406,435-438`

`--num-val-trajs` (8 trajectories) serves both the early-stopping decision —
consulted up to 80 times per run at `--eval-every 25 --patience 5` — and the
final reported `eval_umf`. The help text says so. **Consequence:** every ΔUMF
in the RQ0 capacity table is optimistically biased by an unmeasured amount,
and claim **N4** ("more training data monotonically improves UMF: 0.336 →
0.302 → 0.268") is measured against a *fixed 8-trajectory* selection set
while training data grows 5×, so part of the monotone trend is monotone
selection-set fit. Compounding it: `full` × R1 was never trained
(`atlas_out/` has no such chart), so plan §7.1's pre-registered decision rule
is undefined for R1 for a second, simpler reason than the one the project
gives. Threatens **RQ0**, **N4**, **C3**, **S-4**.

---

## What I did not get to — per checklist item, for whoever resumes

All eleven checklist items (A–K) were reached and carry a verdict. What
remains unfinished within each is listed here so a future session does not
re-derive it.

| Item | Coverage | What remains |
|---|---|---|
| A order of operations | complete | Nothing. Steps 1/2/4/5 read and verified; step 3 traced to an unreachable guard. |
| B hyperparameters | complete | Nothing for the literals. The `m`-semantics drift's effect on the *already-reported* E2 numbers (N7) needs the results-auditor to recompute. |
| C 7-arm ladder | complete | Pairs 1→2, 2→3, 3→4, 4→5, 5→6 all analysed. **Not run:** the direct test that arms 2 and 3 produce identical episodes (Open question 3). |
| D disjoint params | complete | `restore_()` allegation confirmed; P-1 verified for same-kind libraries. **Not done:** no assertion exists in code enforcing the same-kind invariant; nobody has checked whether a mixed-kind library is reachable via `--kind` combinations. |
| E paired seeding | complete | Nothing. Seed→init-state→goal chain traced end to end. |
| F substrate / frozen backbone | complete | Encoder freeze verified at all 11 entry points. **Not run:** the `--arms atlas` crash (F.3) was reasoned from a torch snippet, not reproduced against the real checkpoint. |
| G planner config | complete | Full cross-script table built. **Not read:** `CEMPlanner.plan()`'s own source in the hub cache — the `steps_left` clamp behaviour in G.3 is inferred from the plan's own comments, not from reading the planner. |
| H training protocol | complete | Defaults, early stopping and split all read. **Blocked:** actual step/trajectory counts for `e0_v3_dataset`, `e0_v4_lora4`, `e0_v4_full` are unrecoverable from disk (Open question 1). |
| I train/deploy distribution | complete | All four collectors read; both mismatches located. Nothing outstanding. |
| J regimes | complete | Values resolved against code; both-shapes and reset-survival verified. **Not verified:** whether `PushTEnv._setup()` restores `shape.friction` on reset (Open question 4) — needs a 5-line read of the hub-cached `pusht_env.py`, or gate G4. |
| K documentation | complete for code-checkable claims | **Not read in full:** `E0_RESULTS.md` beyond its top ~90 lines (84 KB, newest-first), `E0_RECOVERY_PLAN.md`, `ATLAS_SUMMARY.md`, `code-review.md`, `ACTION_SAMPLING_REVIEW.md`, `REGIME_DESIGN_REVIEW.md`. Claims in those files that are *about results* rather than *about code* are the results-auditor's scope; claims about code that this pass did not reach may remain unflagged. |

**Not attempted at all, deliberately:** executing any gate, smoke test or
experiment. No GPU work was performed. The single exception is the four-line
torch snippet described under "What this pass verified".

---

# PASS 2 — Closure of pass-1's leftover sub-items

*Run 2026-08-27 at the coordinating session's request, after pass 1's own
"What I did not get to" table named six specific open sub-items. This pass
closes all six. Read-only; no source file modified; no GPU work performed.*

## L. Closure of pass-1's leftover sub-items

### L.1 Item C — arms 2 vs 3, cross-checked against `CODE_AUDIT.md` §2.2

`CODE_AUDIT.md:239-271` independently reaches the identical conclusion as this
file's C.1: `AdaJEPA.reset()` (`atlas/adajepa.py:94-104`) is called from
exactly one place in the repository, `scripts/smoke_e4.py:142`, and nowhere in
`scripts/run_e4.py` or `atlas/harness_e4.py`. Both audits found this by the
same grep and the same read. `CODE_AUDIT.md` adds one detail this file's C.1
did not state: the 5-transition buffer (`atlas/adajepa.py:88`,
`deque(maxlen=5)`) is also never cleared for arm 2, so in addition to never
re-initializing weights, its adaptation window spans episode **and regime**
boundaries — i.e. arm 2's buffer at episode 50 of a stream can still contain
transitions from episode 45's different regime. This strengthens rather than
weakens C.1's verdict: arms 2 and 3 are not merely weight-identical but
state-identical in every respect the code touches. **Reconciled: no new
finding, two independent reads confirm the same DRIFT.** The direct empirical
test (`python scripts/run_e4.py --arms adajepa adajepa_persist --episodes 2
--seeds 1` and diff the JSONL) was still not run — no GPU work permitted — but
confidence is now higher (two independent static traces agree) rather than
untested.

### L.2 Item D — mixed-kind library reachability via CLI

Checked all three scripts' `--kind`/`--kinds` argument definitions directly:
- `scripts/run_e0.py:457`: `--kinds` is `nargs="+"`, default `["ln_act",
  "lora4", "full"]` — trains multiple kinds in one invocation, but each kind's
  chart is written to its own file (`chart_{kind}_{regime}.pt`);
  `run_e0.py` never constructs an `atlas.library.Library` object at all, so
  this cannot itself produce a mixed-kind library.
- `scripts/run_e1.py:217`: `--kind` is a single `choices=["ln_act","lora4","full"]`
  argument (not `nargs`). `load_library_from_e0()` (`scripts/run_e1.py:115-140`)
  builds `c0 = Chart(predictor, kind)` and loads `chart_{kind}_{regime}.pt` for
  every regime using that **same single `kind` value** — every chart in the
  resulting library is necessarily the same kind.
- `scripts/run_e4.py:135`: same single-`choices` pattern, used once at `:238`
  to build one arm's library.

**Verdict: NOT reachable via any `--kind` CLI combination in any of the three
scripts.** One residual, file-tampering-only path is worth recording rather
than dismissing: `Chart.load()` (`atlas/chart.py:145-167`) reads the chart's
kind from the file's own saved `data["kind"]` field, **not** from the caller's
`kind` argument or the filename — so a manually renamed/copied `.pt` file
could load into a library the caller believes is single-kind. And
`atlas/library.py:58-74` (`Library.add()`) performs no kind check whatsoever —
a bare list append with a capacity check, nothing else. So the same-kind
invariant is enforced today only by CLI argument plumbing being consistent,
not by any code-level assertion. Given the checklist's specific question
("reachable via `--kind` CLI combination"), the answer is no; given the
broader question CLAUDE.md's non-negotiable §1.5 implies ("is this invariant
enforced"), the answer remains no — this confirms rather than weakens pass-1's
D verdict.

### L.3 Item F — `--arms atlas` crash, re-derived from the actual call chain

Traced the real call chain rather than relying on a hypothetical torch
snippet:
1. `scripts/run_e4.py:165-168` calls `p.requires_grad_(False)` on every
   parameter of `wm.encoder` and `wm.predictor`, unconditionally, before any
   arm runs.
2. `atlas/adajepa.py:89-92` (`AdaJEPA.__init__`) is the **only** code path in
   the repository that ever calls `p.requires_grad_(True)` on a predictor
   parameter after that freeze — only for arms 2/3, only once constructed
   (`atlas/harness_e4.py:100-105`, gated on `arm in ("adajepa",
   "adajepa_persist")`).
3. `atlas/loop.py::atlas_refine` (`:220-231`, called by arms 4/5/6 via
   `atlas/harness_e4.py:307-315`) and `atlas/expand.py::_fit_candidate`
   (`:199-214`) both build an `Adam` optimizer directly from
   `predictor.named_parameters()` filtered by chart param names — **neither
   ever calls `requires_grad_(True)` on anything.**
4. `scripts/run_e4.py:65-68` (`ALL_ARMS`) orders the default `--arms` sweep as
   `frozen, adajepa, adajepa_persist, atlas_fixed, atlas_detect, atlas, ...` —
   so in a default full-ladder run, arm 2's `AdaJEPA.__init__` flips `ln_act`'s
   `requires_grad` to `True` on the **shared** `wm.predictor` object before
   arm 4 ever calls `atlas_refine`. `load_state_dict` (used to reset predictor
   values between arms, `scripts/run_e4.py:236`) restores tensor *values* but
   not the `requires_grad` flag, so once flipped it stays flipped.
5. `scripts/run_e4.py:209-211` sets `args.arms = ["atlas"]` when `--profile`
   is passed — i.e. the budget-calibration run the implementation plan
   instructs be done first runs *only* arm 6, with no prior `AdaJEPA`
   construction in that process to perform the flip.

If arm 6 (or any ATLAS arm) is the first or only arm executed in a process,
the first `atlas_refine` call builds `Adam` over `requires_grad=False`
parameters whose loss (computed under `torch.no_grad()` rollout helpers) has
no `grad_fn` — `loss.backward()` raises `RuntimeError: element 0 of tensors
does not require grad and does not have a grad_fn`. Independently confirmed by
`CODE_AUDIT.md:325-373` (§2.4) via an identical file:line trail.

**Confidence: high**, re-derived from the real production call chain and
corroborated by an independent audit pass. Not executed on GPU, but the
crash mechanism is standard, version-independent PyTorch behavior.

### L.4 Item G — `CEMPlanner.plan()` read directly

`atlas/hub/hub/facebookresearch_jepa-wms_main/evals/simu_env_planning/planning/planning/planner.py:211-344`,
read directly rather than inferred from comments. Confirms exactly as pass 1
inferred:
- Line 272-275: `plan_length = self.horizon if steps_left is None else
  min(self.horizon, steps_left)` — the literal `steps_left` clamp, now
  [CODE]-verified.
- Line 333: `a = mean[: self.num_act_stepped]` — actions returned are
  truncated to `num_act_stepped` regardless of `plan_length`; `num_act_stepped`
  and `steps_left`/`horizon` are two independent knobs, exactly as G's table
  documented.
- No reference to `frameskip` anywhere in `planner.py` — it is purely a
  caller-side concept. Confirms G.3's diagnosis: the unit error is entirely in
  the callers' arithmetic (`scripts/run_e4.py:154`, `atlas/harness_e4.py:248`),
  not in `CEMPlanner` itself.

**Verdict: G.3's clamp-behavior claim upgraded from L3-by-inference to L2
direct-code-read.** No new drift; previously reported unit error stands.

### L.5 Item J — does `PushTEnv._setup()` restore friction/damping on every reset?

`CODE_AUDIT.md:1304-1317` (Priority 9) flagged this as unresolved and
explicitly deferred to a direct read of `pusht_env.py`, warning that if
unresolved it would be "a serious, silent, regime-mixing bug." Read
`hub/hub/facebookresearch_jepa-wms_main/evals/simu_env_planning/envs/pusht_env/pusht_env.py`
directly (the hub-cache copy — confirmed by `CLAUDE.md` to be what actually
loads at runtime via `torch.hub`, not the `vendor/jepa-wms` copy):
- `PushTEnv.reset()` (`:432-474`) calls `self._setup()` unconditionally as its
  literal first line (`:433`).
- `_setup()` (`:684-716`) does `self.space = pymunk.Space()` (`:685`, a
  **brand-new** space, not a mutation), explicitly sets `self.space.damping =
  0` (`:687`), and rebuilds the agent (`:701`) and block (`:703`) as brand-new
  `pymunk.Body`/`pymunk.Shape` objects. New shapes default `friction=0.0`
  unless explicitly set, and `_setup()` never sets it — it only sets
  `body.friction = 1` on the agent body (`:726`), which
  `REGIME_DESIGN_REVIEW.md:109` independently and correctly notes is dead
  code (`pymunk.Body` has no `friction` attribute the solver reads).

**Verdict: `_setup()` genuinely, fully rebuilds the pymunk space and every
shape/body object on every `reset()` call, and explicitly re-zeroes
`space.damping`.** A prior regime's `shape.friction=2.0` (R1) or
`space.damping=0.5` (R2) **cannot** survive into a later `reset()` on the same
`base_env` object — that reset discards the very objects those values were
set on. **This closes J.3 and `CODE_AUDIT.md` Priority 9 in the reassuring
direction: the regime-mixing/contamination hazard both audits worried about
does not exist.** `PhysicsRegime.reset()`'s pattern (reset the wrapped env,
then re-call `_apply_physics()`) is necessary (because `_setup()` wipes the
physics config) but also sufficient (because `_setup()` wipes it completely,
leaving nothing stale to leak). E4's alternating-regime stream is not at risk
of this specific contamination mode.

### L.6 Item K — additional doc code-claims spot-checked; one new drift found

`E0_RECOVERY_PLAN.md:952-967` ("P5 — Fix E1's protocol... do before any E1
run") lists three prescribed fixes to `atlas/harness.py::run_e1_episode`:
use E0's filtered dataset init/goal pairs (not random), use
`run_e0_planning.py::block_success()` (not `goal_utils.eval_state()`, which
the doc says wrongly includes agent position), and set `num_act_stepped=1`.
Checked whether these landed:
- **Fix 1 not applied.** `atlas/harness.py:328` still calls
  `goal_utils.sample_random_init_goal_states(episode_seed)`.
- **Fix 2 not applied.** `atlas/harness.py:386` still calls
  `goal_utils.eval_state(...)["success"]`.
- **Fix 3 not applied.** `scripts/run_e1.py:84-89` still defaults
  `num_act_stepped=6` (already noted in pass 1's item G.1 table).

**Previously-unrecorded finding: `E0_RECOVERY_PLAN.md`'s own prescribed "do
before any E1 run" fix was never applied to code.** This does **not**
retroactively undermine `HANDOFF.md` §7.1's E1-closure argument (N8): that
computation is built from `e0_v3_planning_dataset_baseline`/`_ln_act` —
E0's own planning records, produced by `scripts/run_e0_planning.py` (which
already uses the correct dataset/success-criterion/nas) — not from any run of
`atlas/harness.py::run_e1_episode`. The E1-closure argument bypasses the
buggy function entirely, so N8 is not contaminated by this drift. What it does
mean: if `run_e1_episode` (E1's actual stream-style router-evaluation code)
were ever run today as-is, it would measure a different task than E0 in three
ways `E0_RECOVERY_PLAN.md` already diagnosed but never fixed. Threatens
**P-5** only if E1's stream is ever revived without applying this fix first;
no threat to any currently-reported number.

Other spot checks (`REGIME_DESIGN_REVIEW.md:109`'s dead-code claim,
`code-review.md:408-414`'s Bug #7 `PlanEvaluator` claims) found consistent
with prior framing; no further code-claim discrepancies found in the time
available. `ACTION_SAMPLING_REVIEW.md` and most of `E0_RESULTS.md`/
`ATLAS_SUMMARY.md` beyond what pass 1 covered were not exhaustively
line-by-line read (results-narrative scope, not this file's).

### Updated "What I did not get to" table (supersedes pass 1's)

| Item | Status after pass 2 |
|---|---|
| C (arm 2 vs 3) | Cross-checked against `CODE_AUDIT.md` §2.2 — independent confirmation, plus a buffer-persists-across-regimes detail pass 1 had not recorded. Still not run empirically (no GPU permitted); confidence raised from single-source to two-independent-static-traces. |
| D (mixed-kind library) | **Closed.** Not reachable via `--kind` CLI in any script. Residual file-tampering path noted but not CLI-reachable. |
| F (`--arms atlas` crash) | **Closed.** Re-derived from the real call chain, corroborated by `CODE_AUDIT.md` §2.4. High confidence; not executed on GPU. |
| G (`CEMPlanner.plan()`) | **Closed.** Read directly; confirms prior inference exactly, evidence upgraded L3→L2. |
| J (regime persistence across reset) | **Closed, reassuring direction.** No regime-persistence contamination is possible — confirmed by direct source read, not inference. |
| K (remaining docs) | Spot-checked `E0_RECOVERY_PLAN.md`, `REGIME_DESIGN_REVIEW.md`, `code-review.md`, partial `E0_RESULTS.md`. One new drift found (E1 harness's un-applied P5 fixes, scoped, does not threaten N8). `ACTION_SAMPLING_REVIEW.md` and the bulk of results-narrative docs not exhaustively read (out of this file's scope). |

**No verdict from pass 1 changes.** The most consequential update is J: the
regime-persistence hazard is now resolved as a non-issue by direct source
read. The arm-2-vs-arm-3 verdict (DRIFTS, behaviourally identical) is
unchanged but doubly-confirmed. One new, previously unrecorded drift was
found (E1 harness's un-applied P5 fixes) and is scoped to not threaten any
currently-reported number.

---

*End of PROPOSAL_CODE_ALIGNMENT.md. No source file was modified during either
pass. Every drift above is documented and left in place, per the audit's
provenance rule.*
