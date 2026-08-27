# ATLAS — Code Audit (silent-corruption bug hunt)

**Last updated: 2026-08-27, pass 2 — resumed exactly where pass 1's "What I did not get to" left off (priorities 8, 9, 6-remainder, and a partial pass 4). Pass 2 findings are in PART 9 below; pass 1 (Parts 1-8) is unmodified except for the "What I did not get to" section at the bottom, which now reflects what pass 2 resolved.**

---

## What this file is

A pre-submission audit of the ATLAS codebase hunting for bugs that **do not
crash** — bugs that quietly corrupt results into something plausible-looking.
Written for a future session with zero memory of the conversation that
produced it.

Claim IDs (C1–C4, RQ0–RQ4, L-1, P-1..P-5, N1..N9, S-1..S-8, G-1) refer to
rows in `research_audit/CLAIMS_MATRIX.md`.

**No bug found here was fixed.** Fixing silently would destroy the provenance
of every number already on disk. Every finding below records whether it
affects results already on disk or is still-unrun code (free to fix).

Evidence discipline used throughout: findings are separated into
**FACT (verified from code / from data on disk)**, **DOC CLAIM (asserted by a
project document, not independently checked)**, **INFERENCE**, and
**OPEN QUESTION**.

---

# PART 1 — THE CLOSED-LOOP / MULTI-REPLAN PATH (audit item 7)

This was the finding the user most wanted. **Verdict: the multi-replan path is
CORRECT on the specific point that was doubted.** There is no stale-context
bug. Two secondary asymmetries are documented below, neither of which
invalidates N5.

## 1.1 How many CEM searches actually occur per episode — FACT

`scripts/run_e0_planning.py:265`

```python
n_replans_target = max(max_steps // num_act_stepped, 1)
```

and the inner execution loop at `scripts/run_e0_planning.py:305-320` executes
`num_act_stepped * frameskip` raw env steps per replan, breaking on
`elapsed >= max_steps`.

With `max_steps=30`, `frameskip=5`:

| config | `n_replans_target` (loop bound) | raw steps executed per replan | **actual CEM searches per episode** |
|---|---|---|---|
| `num_act_stepped=6` (default) | 5 | 6×5 = 30 | **1** (fully open-loop) |
| `num_act_stepped=2` (`e0_planning_nas2`) | 15 | 2×5 = 10 | **3** |
| `num_act_stepped=1` (E4's default) | 30 | 1×5 = 5 | **6** |

**Verified against data on disk, not just from the code.** Reading
`atlas_out/e0_planning_n100/{baseline,ln_act}_R2.jsonl` and
`atlas_out/e0_planning_nas2/{baseline,ln_act}_R2.jsonl`:

- `e0_planning_n100`: `replans == 1` for **every one of the 100 episodes**, in
  both arms. Confirms fully open-loop.
- `e0_planning_nas2`: `replans` is 3 for episodes that run the full 30 steps and
  1 for episodes that succeed inside the first 10 raw steps. Confirms ~3 replans.

So the project's own description of its protocol is accurate. **Clean verdict.**

## 1.2 Is the context fed to the 2nd and later replans correct? — FACT, CLEAN

`scripts/run_e0_planning.py:305-312`:

```python
obs_td = make_obs_td(obs["visual"], obs["proprio"], device)   # top of replan loop
...
for a in action:
    obs, reward, done, info = base_env.step(a)                # obs REBOUND here
```

`obs` is rebound on every raw env step inside the execution loop, so at the top
of replan *k+1* `obs` holds the **latest** observation, not a stale one. It is
then re-wrapped into a fresh TensorDict and handed to `agent.act()`, which calls
`self.model.encode(obs)` (`gc_agent.py:174`) — i.e. the observation **and the
proprio** are genuinely **re-encoded from scratch at every replan**.

There is no offset bug, no cached latent, no reuse of the episode's first
encoding.

**VERDICT: CLEAN. The stale-context / wrong-offset / not-re-encoded failure
mode the user suspected does not exist in this code. Claim N5 is not
invalidated by that mechanism.** Severity: n/a. Threatened claim: N5 —
**not threatened by this**.

## 1.3 Goal, success criterion and raw step budget across nas settings — FACT, CLEAN

Identical in both configurations, verified by reading the code and by
recomputing from the JSONL:

- Goal: `sample_dataset_init_goal(states, seq_lengths, rs, ...)` with
  `rs = np.random.RandomState(seed)` freshly constructed per episode
  (`run_e0_planning.py:238`), and `seed == episode index` everywhere in this
  script. The number of `rs` draws consumed varies with retry count but `rs` is
  *fresh per episode*, so both arms and both nas settings draw the same
  init/goal for the same episode index.
- **Empirically confirmed:** for `e0_planning_n100` (100 episodes) and
  `e0_planning_nas2` (20 episodes), `init_block_pos_diff` matches **exactly
  (0 mismatches at float equality)** between the baseline arm and the `ln_act`
  arm. Pairing is real.
- Success criterion: `block_success()` (`run_e0_planning.py:208-227`), checked
  after **every raw env step** in both configs, identical formula.
- Raw step budget: `max_steps = 30` in both; termination is
  `elapsed >= max_steps`, checked identically.

**VERDICT: CLEAN — the nas=6 and nas=2 runs are comparable on goal, success
criterion and raw-step budget.**

## 1.4 Asymmetry A: `steps_left` is wrong but inert — FACT, LOW

`scripts/run_e0_planning.py:286` and `atlas/harness_e4.py:248`:

```python
steps_left_model = (n_replans_target - replan_idx) * num_act_stepped
```

`n_replans_target` is a **loose loop bound**, not the true number of replans, so
this over-states the remaining budget. At nas=2 replan 2, the true remaining
model steps are 2 but this passes 26.

**Why it is inert:** `CEMPlanner.plan()` uses it only as
`plan_length = min(self.horizon, steps_left)` (`planner.py:275`), and
`horizon=6`. Since every value this expression produces is ≥ 6, `plan_length`
is always exactly 6 in every configuration checked. **No numerical effect on
any result on disk.**

Severity: **LOW**. Threatened claim: none (inert). Affects results on disk:
**no** — but it is a landmine if `horizon` is ever raised above the smallest
`steps_left` value, and it means the planner never shortens its horizon near
the end of an episode.

## 1.5 Asymmetry B: nas=2 buys 3× the planner compute, not just "more replans" — FACT/INFERENCE, MEDIUM

Because `plan_length` stays 6 (§1.4), each of nas=2's three replans runs a
**full 6-model-step (30-raw-step) CEM search** — `num_samples=300 ×
iterations=30` — and then executes only the first 2 model steps. So the nas=2
episode consumes **3× the CEM search compute** of the nas=6 episode while
covering the same 30 raw environment steps.

This is standard receding-horizon MPC and is not a bug. But it does mean the
+10.0pp N5 effect is **not attributable to "closed-loop feedback" alone** — it
confounds feedback with a 3× planner-compute increase. The paired
chart-vs-baseline comparison *within* nas=2 is still fair (both arms get the
same budget); it is the nas=6-vs-nas=2 narrative that is confounded.

Severity: **MEDIUM** (scientific framing, not a coding error). Threatened
claim: **N5**. Affects results on disk: **yes — `atlas_out/e0_planning_nas2`.**
The numbers are correct; the causal attribution in the write-up is what needs
qualifying.

## 1.6 `base_env.step(a)` instead of `regime.step(a)` — FACT, LOW *here*, HIGH *if corruption is ever added*

`scripts/run_e0_planning.py:310` and `atlas/harness_e4.py:263` both step the
**raw** env, bypassing the `PhysicsRegime` wrapper.

- For `PhysicsRegime` this is **harmless**: `atlas/regimes.py:76-102` defines no
  `step()` override, so `regime.step` is `gym.Wrapper`'s pass-through. All
  physics is applied in `reset()` via `_apply_physics()`, and persists for the
  rest of the episode. **Clean for every run currently on disk.**
- For `VisualCorruption` (`atlas/regimes.py:176-225`) it would be **fatal and
  silent**: that class is a `gym.ObservationWrapper` whose entire effect lives
  in `observation()`, which is only invoked through the wrapper's own
  `step`/`reset`. Calling `base_env.step()` returns the **uncorrupted** image.
  If a corrupted planning episode is ever run through
  `run_e0_planning.run_episode` or `harness_e4.run_e4_episode`, the corruption
  is silently disabled and the run measures nothing.

Severity: **LOW now / HIGH latent.** Threatened claim: **RQ2 / N9** (E2 Cell C)
if this path is ever used for corruption. Affects results on disk: **no** — no
current planning run uses corruption. Note `scripts/run_e0.py`'s collector
*does* wrap correctly (`env = VisualCorruption(env, ...)` then `env.step(...)`),
so E2's collected trajectories are unaffected.

---

# PART 2 — THE NEVER-EXECUTED E4 / CONTINUAL-STREAM PATH (audit item 1)

**Established fact (verified): `atlas_out/` contains no `e4` directory.** None
of `scripts/run_e4.py`, `atlas/harness_e4.py`, or `atlas/loop.py::atlas_step`
has ever executed end to end. **Every finding in this section is therefore
free to fix — no number on disk is affected.** Equally, every one of them
would be baked into the paper's central table if E4 is launched as-is.

These are ordered by how badly they corrupt the result *without crashing*.

## 2.1 🔴 CRITICAL — The full-ATLAS arm can never commit a chart. Verification is dead code.

`atlas/harness_e4.py:213-219` calls `atlas_step` with:

```python
step_info = atlas_step(
    ...,
    next_encoder_output=None, next_actions=None,      # <-- HARD-CODED None
    proprio_ctxt=proprio_ctxt, rng=router_rng,
)
```

`atlas/loop.py:139-149`:

```python
if cfg.expansion_mode == "atlas":
    expander.record(best_umf, encoder_output, actions, proprio_ctxt)
    if (
        next_encoder_output is not None        # <-- ALWAYS False from harness_e4
        and next_actions is not None
        and expander._strikes >= cfg.q
    ):
        probe_outcome = expander.maybe_expand(...)
```

**`Expander.maybe_expand()` is therefore never called for arm 6 (`atlas`).**
Strikes accumulate forever, `probe_outcome` stays `"not_ready"` for every
episode, and `charts_committed_cumulative` stays 0 for the entire stream.

The comment block at `harness_e4.py:198-203` describes a "two-deep chunk
buffer" where chunk *k* is the deficit data and chunk *k+1* is the next-unseen
verification chunk. **That design was never implemented** — the code keeps a
single `prev_chunk` and passes nothing as the next chunk.

**What it silently corrupts:** arm 6 becomes behaviourally identical to arm 4
(`atlas_fixed`). The E3/RQ3 comparison — "ATLAS commits ~2 charts, detect-only
commits more than 2, fixed-library commits 0" — would come back as
`ATLAS: 0 commits, detect-only: many, fixed: 0`. That result **looks like a
publishable finding** ("verification is highly conservative") and is entirely an
artifact of the probe never firing. This is the single most dangerous bug in the
repository.

- Severity: **CRITICAL**
- Threatened claims: **C2, C2-probation, RQ3, L-1, N9-in-stream, C4**
- Results on disk affected: **NO — E4 has never run. Free to fix.**
- Note: `next_proprio_ctxt` is also never passed, so even wiring up the two
  chunks requires threading three arguments, not two.

## 2.2 🔴 CRITICAL — Arm 2 (AdaJEPA) never re-initialises, so it *is* arm 3 (Persistent-AdaJEPA)

This is exactly the failure mode audit item 5 was written to catch, in the
mirror-image direction: it is not that Persistent-AdaJEPA accidentally resets,
it is that **plain AdaJEPA accidentally does not**.

`atlas/adajepa.py:94-104` defines `reset()`, whose docstring says *"Called at the
start of each episode."* Grepping the whole repository for call sites:

- `scripts/smoke_e4.py:142` — the smoke test calls it manually, so the smoke
  test passes and asserts the reset works (`smoke_e4.py:164`).
- **Nothing else.** `atlas/harness_e4.py` never calls it. `scripts/run_e4.py`
  never calls it. `run_e4_episode()` has no reset hook at all.

`build_arm_state()` is called once per `(arm, seed_run)` at
`scripts/run_e4.py:237`, so arm 2 starts pristine at the top of each seed run and
then **adapts continuously across all 120 episodes of the stream without ever
resetting**, exactly like arm 3.

Two consequences, both silent:

1. Arms 2 and 3 differ by **nothing at all** in the code path. Their success
   rates would differ only by CEM sampling noise. The ladder's persistence rung
   — the rung that is *"our modification"* and the whole premise of
   Persistent-AdaJEPA — becomes vacuous, and would be reported as "persistence
   buys ~0pp", a plausible-looking null.
2. The 5-transition buffer (`adajepa.py:88`, `deque(maxlen=5)`) is also never
   cleared for arm 2, so its adaptation window spans episode and **regime**
   boundaries.

- Severity: **CRITICAL**
- Threatened claims: **L-1 (the paper's central table), RQ4, C4, P-3**
- Results on disk affected: **NO — E4 has never run. Free to fix.**

## 2.3 🔴 CRITICAL (suspected) — the motion gate is calibrated on a ~10-model-step displacement but applied to 1-model-step chunks

`scripts/run_e4.py:182-190`:

```python
gate_trajectories = load_regime_trajectories(
    model, prep, regime_a, num_trajs=3, traj_len=10, device=device, seed_offset=20_000)
gate_displacements = torch.tensor([
    (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
    for t in gate_trajectories
])
motion_gate = compute_motion_gate(gate_displacements)
```

This measures the latent displacement across a **whole collected trajectory**
(`encoder_output[-1] - encoder_output[0]`).

E4 runs at `num_act_stepped = 1` (`scripts/run_e4.py:80`), so each replan
executes `1 × frameskip = 5` raw steps, and the chunk re-encoded at
`harness_e4.py:278-289` is `n_raw = 5`, `keep_idx = [0, 5]` — i.e.
**`encoder_output` has exactly 2 frames and `actions` has exactly 1 model
step.** The displacement `umf()` compares against the gate
(`atlas/score.py:87`) is therefore a **single-model-step** displacement.

**INFERENCE (not yet measured on hardware):** a one-model-step displacement is
systematically far smaller than a whole-trajectory displacement, so the gate is
very likely to exceed essentially every E4 chunk's displacement.

**What that silently does if true:** `umf()` returns `None` for every chart on
every replan → `route()` hits `if not valid: return current_idx, {"gated": True}`
(`atlas/router.py:79-81`) → **the router never switches charts**, `best_umf`
stays `None` (`loop.py:130-134`), `Expander.record()` returns immediately
(`expand.py:79-80`) so **no strike is ever counted and no expansion can ever
occur in any arm**. Arms 4/5/6/7 collapse into "always run chart index 0,
never adapt" — i.e. into arm 1 — while still producing an entirely plausible
success-rate table.

Additional aggravating detail: `compute_motion_gate` takes the 10th percentile
of only **3 values** (`num_trajs=3`), which is a meaningless quantile estimate
regardless of the scale mismatch.

- Severity: **CRITICAL if confirmed** (see open question below)
- Threatened claims: **C1, C2, RQ3, RQ4, L-1, G-1**
- Results on disk affected: **NO — E4 has never run. Free to fix.**
- **OPEN QUESTION (could not run on GPU under this audit's constraints):**
  measure the actual distribution of single-model-step latent displacements
  under R0 and compare against `compute_motion_gate`'s output for
  `traj_len=10, num_trajs=3`. This is a ~5-minute check and should be done
  **before** any E4 launch. If the gate does turn out to sit below typical
  1-step displacements, this finding downgrades to MEDIUM (noisy 3-sample
  quantile) rather than CRITICAL.

## 2.4 🟠 HIGH — `requires_grad=False` on the predictor makes E4 arm-order-dependent, and crashes on any arm subset

`scripts/run_e4.py:165-168`:

```python
for p in wm.encoder.parameters():
    p.requires_grad_(False)
for p in wm.predictor.parameters():
    p.requires_grad_(False)     # <-- ALL predictor params frozen
```

Nothing ever re-enables gradients on the chart parameter surface **except**
`AdaJEPA.__init__` (`atlas/adajepa.py:90-91`):

```python
for p in self._params:
    p.requires_grad_(True)
```

`atlas_refine()` (`loop.py:222-231`) and `_fit_candidate()`
(`expand.py:201-214`) both just select params by name and call
`loss.backward()`. If those params have `requires_grad=False` and the encoder
output is detached (it is — it comes from `world_model.encode()` under
`torch.no_grad()`), the loss has no `grad_fn` and `loss.backward()` raises
`RuntimeError: element 0 of tensors does not require grad`.

Because `AdaJEPA` mutates the **shared** `wm.predictor` Parameter objects, and
`load_state_dict` does not reset `requires_grad` flags, the behaviour depends
on **arm ordering**:

- `--arms` default (full ladder, order `frozen, adajepa, adajepa_persist,
  atlas_fixed, ...`): arm 2 flips `requires_grad=True` as a side effect, and the
  ATLAS arms then work by accident.
- `--arms atlas` or `--arms atlas_fixed atlas` (a resume, a shard, a re-run of
  one arm) — **and `--profile`, which forces `args.arms = ["atlas"]` at
  `run_e4.py:210`** — crash at the first `atlas_refine()` call.

This is a crash rather than silent corruption, which is the good news. The
silent part is the **order dependence**: the correctness of the ATLAS arms is
an accidental side effect of an unrelated arm having run first in the same
process.

- Severity: **HIGH**
- Threatened claims: **RQ3, RQ4, L-1** (blocks the run rather than corrupting it)
- Results on disk affected: **NO — E4 has never run. Free to fix.**
- Aggravating note: `atlas_refine()` only fires when `state.current_idx != 0`
  (`harness_e4.py:305`), so the crash does not occur until routing first selects
  a non-c0 chart — i.e. potentially deep into a run, after hours of GPU time.

## 2.5 🟠 HIGH — `Chart.restore_()` does not restore the pretrained weights (for `ln_act` and `full`)

`atlas/chart.py:107-127`:

```python
def restore_(self, predictor):
    if self.kind == "lora4":
        ... remove_parametrizations(...)   # correct
    else:
        self.apply_(predictor)             # <-- re-applies THIS chart
```

For `ln_act` (E4's default kind) and `full`, `restore_()` is a synonym for
`apply_()`. After `chart.restore_(predictor)` the predictor still carries that
chart's weights.

Why it does not corrupt UMF scoring: every scoring path (`score.umf`,
`router._e1_score`, `router._sdyn_score`) calls `chart.apply_()` before its
forward pass, and all charts of a given kind share the identical
`_param_names` surface, so each chart fully overwrites the previous one. So
per-chart scores are still correct. **Clean on that specific point.**

Where it does bite:

1. `atlas/harness_e4.py:245-252` applies a chart for planning and "restores" it
   in a `finally:` — leaving the predictor **permanently dirty** with the last
   selected chart. Any code that assumes it is looking at the pristine
   predictor after that point is wrong.
2. It falsifies the literal wording of **claim P-1** ("updating one cannot alter
   another's parameters — parameter-level retention is *guaranteed*"). The
   guarantee holds only because every chart is fully re-applied before use, not
   because restore works.
3. **Gate G1 cannot catch it**, by construction — see §4.1.

`HANDOFF.md` §4 already documents this and reports **10 production call sites
rely on the current behaviour**, so it must not be "fixed" casually.

- Severity: **HIGH** (latent; a correctness invariant that is not actually held)
- Threatened claim: **P-1**
- Results on disk affected: **not detectably** — the full-overwrite property
  masks it in every current path. INFERENCE, not proof.

## 2.6 🟠 HIGH — `maybe_expand()` picks the incumbent "best chart" using the verification chunk itself

`atlas/expand.py:131-142`:

```python
best_chart = library[_argmin_umf(library, world_model,
                                 next_encoder_output, next_actions, motion_gate,
                                 next_proprio_ctxt)]
candidate = library.clone_from(library._charts.index(best_chart))
_fit_candidate(candidate, ...)
cand_umf = compute_umf(candidate, ..., next_encoder_output, next_actions, ...)
best_umf = compute_umf(best_chart, ..., next_encoder_output, next_actions, ...)
```

The incumbent is chosen as the **argmin over the held-out chunk**, and is then
compared against the candidate **on that same chunk**. The incumbent therefore
gets a max-over-charts selection advantage on the exact data used for the
verdict, and the candidate must beat an optimistically-selected opponent.

This is a **conservative** bias (it makes committing harder), so it is not
leakage in the candidate's favour — but it is a bias, and it makes
"charts committed" a biased estimator of the mechanism's behaviour.

The spec (`expand.py:9-11` docstring, and proposal §2/C2) says the candidate
must beat "the current best chart" — most naturally the *currently selected*
chart, or the argmin over the **deficit** chunks, not over the verification
chunk.

Also note `library._charts.index(best_chart)` uses `list.index` with `==` on
`Chart` objects; `Chart` defines no `__eq__`, so this falls back to identity —
correct, but fragile.

- Severity: **HIGH**
- Threatened claims: **C2, RQ3**
- Results on disk affected: **YES, partially** — `atlas_out/e2_R2_cellB_q1`
  (the "3 charts committed" demonstration behind **N9**) went through this exact
  code path. The bias is conservative, so the 3 commits are, if anything,
  *under*-counted; but the number is not the unbiased quantity the claim
  implies.

## 2.7 🟡 MEDIUM — data-leakage check on `_fit_candidate` vs verification (audit item 6): CLEAN, with one structural caveat

**FACT:** `Expander.record()` (`expand.py:79-88`) appends the current chunk to
`self._deficit_chunks`; `maybe_expand()` fits on exactly those chunks and
verifies on `next_encoder_output/next_actions`, which are passed separately by
the caller. Inside `expand.py` the two sets are genuinely disjoint tensors.
`record()` also correctly `.detach()`es everything it stores
(`expand.py:84-85`), so no gradient path leaks back into a stored chunk.

**Caveat:** the disjointness is a *caller* obligation, and `expand.py` does not
assert it. In `harness_e4.py` the caller passes `None` (see §2.1), so the
question is currently moot there; in `scripts/run_e2.py` (not read this pass —
see "What I did not get to") it is unverified.

- Severity: **MEDIUM** (unasserted contract)
- Threatened claim: **C2**
- Results on disk affected: unknown for E2's q=1 diagnostic — **open question**.

## 2.8 🟡 MEDIUM — arms 4/5/6 never refine `c0`, but arms 2/3 always refine

`atlas/harness_e4.py:305`:

```python
elif state.arm in ("atlas_fixed", "atlas_detect", "atlas") and state.current_idx != 0:
```

Arms 2 and 3 refine on **every** replan. Arms 4/5/6 refine only when the router
has selected a non-identity chart. Under `expansion_start_library="full"` the
library is `{c0, chart_B}`, so on the R0 segments — where UMF should correctly
prefer `c0` — arms 4/5/6 perform **zero adaptation** while arms 2/3 keep
adapting.

Claim **L-1** requires each ladder rung to differ from its neighbour by
**exactly one mechanism**. Arm 3 → arm 4 here differs by two: *(a)* gaining a
library and routing, and *(b)* losing continuous adaptation on the R0 segments.

Whether this is a bug or a deliberate "c₀ is never refined" design decision
(`library.py:23` says c₀ "is never refined (callers must enforce this)") is a
**design question the project must answer explicitly**, because the attribution
argument depends on it.

- Severity: **MEDIUM**
- Threatened claim: **L-1**
- Results on disk affected: **NO — E4 has never run.**

## 2.9 🟡 MEDIUM — RQ4's "paired delta" between first visit and final revisit is not measurable under the current seeding

`atlas/streams.py:86-87`:

```python
seed = paired_seed(seg_idx + stream_seed_offset * 1000,
                   ep_idx + seed_run * 10_000)
```

The **segment index is part of the seed key**. Therefore episode *i* of segment
0 (first visit to A) and episode *i* of segment 4 (final revisit to A) have
**different seeds and hence different initial states and goals**.

`RQ4` as written in `CLAIMS_MATRIX.md` requires "success on the final revisit to
regime A exceeds success on the first visit, with **paired** delta > 0". With
segment-dependent seeds, first-visit and final-revisit are two **independent**
20-episode populations — an unpaired comparison.

`scripts/make_tables.py:110-116` compounds this: it reports `sr_first` and
`sr_final` as **bare means with no CI and no test at all**, which
`CLAUDE.md` §5 explicitly forbids ("Every table reports Δ with a CI — never two
bare means").

- Severity: **MEDIUM**
- Threatened claims: **RQ4, C4**
- Results on disk affected: **NO — E4 has never run. Free to fix**, but fixing
  it means changing the seeding scheme (make the seed depend on the *regime
  visit* rather than the segment index) or accepting an unpaired recall test
  and powering it accordingly.

## 2.10 🟡 MEDIUM — `make_t2` pairs on length, not on keys

`scripts/make_tables.py:126`:

```python
if baseline_outcomes is not None and arm != baseline_arm and len(outcomes) == len(baseline_outcomes):
    delta_mean, (delta_lo, delta_hi) = paired_bootstrap(outcomes, baseline_outcomes)
```

`outcomes_for()` sorts each arm's records by `(seed_run, global_episode_idx)`
(`make_tables.py:101-104`) — correct — but the pairing guard is **equal length**,
not **equal key sets**. If two arms have different completed-episode sets that
happen to have the same count (entirely possible with `run_e4.py`'s resume
support, a crashed arm, a `--arms` subset, or a merged multi-container Modal
run), the paired bootstrap and McNemar silently compare **misaligned episodes**
— which is precisely the "unpaired comparison masquerading as paired"
failure mode.

The fix is to intersect on the key set and assert it, not to compare lengths.

- Severity: **MEDIUM–HIGH** (severity depends on whether E4 is ever resumed;
  E4 is *designed* to be resumed, so treat as HIGH in practice)
- Threatened claims: **L-1, RQ4, P-5**
- Results on disk affected: **NO — `make_t2` has never had E4 data to run on.**

## 2.11 🟢 LOW — E4 is not reproducible; resume changes the random stream

**FACT.** `GC_Agent.__init__` (`gc_agent.py:42-45`) creates
`local_generator` / `local_gpu_generator` and seeds them **once** with
`cfg.local_seed` (which is hard-coded to `0` at `run_e4.py:88`,
`run_e0_planning.py:95`, `run_e1.py:97`). `CEMPlanner.plan()`
(`planner.py:290-291`) draws all its samples from that single generator, and it
is **never re-seeded per episode**. `scripts/run_e4.py` constructs one agent
(line 193) and reuses it for all 7 arms × 3 seeds × 120 episodes.

Consequences:

- The CEM noise stream is a single sequence whose state at episode *k* depends
  on every draw consumed before it. Episodes that succeed early consume fewer
  draws, so **arms desynchronise from each other immediately**.
- `run_e4.py`'s resume (`load_completed_keys`, line 106) skips completed
  episodes but does **not** replay the RNG draws they consumed — so a resumed
  run produces a **different** random stream from an uninterrupted one.
- Running a subset of `--arms` gives different results than running the full
  ladder.

This does **not** break the pairing claim **P-5** as literally stated: P-5 is
about identical initial states and goals across arms, which *is* satisfied
(`rs = np.random.RandomState(spec.seed)`, freshly constructed per episode at
`harness_e4.py:176`, with `spec.seed` independent of arm). Planner noise being
independent across arms is a normal and defensible design.

- Severity: **LOW** (reproducibility, not correctness)
- Threatened claim: **P-5** (partially — the *stated* guarantee holds)
- Results on disk affected: same property applies to E0's planning runs, but
  there each arm is a separate process starting from `local_seed=0`, so early
  episodes are approximately CEM-paired and later ones are not.

## 2.12 🟢 LOW — `lora4` chart construction consumes global torch RNG

`atlas/chart.py:269`:

```python
A = torch.randn(rank, in_features) / rank
```

`_inject_lora` draws from the **global** torch RNG on every `Chart(predictor,
"lora4")` construction. An arm that constructs more charts than another
therefore desynchronises the global stream. Irrelevant at E4's default
`kind="ln_act"` (no draws), but a landmine if `--kind lora4` is ever used.

- Severity: **LOW**
- Threatened claim: **P-5**
- Results on disk affected: **no** (E2/E0 lora4 runs use one process per arm).

## 2.13 🟢 LOW — `E4` deviates from plan §6.1/§7.4 on the regime pair, disclosed

`atlas/streams.py:58` defaults `regimes=("R0","R2")` where plan §6.1/§7.4
specifies R0/R1. This is **disclosed in the docstring** (`streams.py:60-67`)
with a stated justification (R2 measured at 40pp degradation vs R1's 20pp) and
the resolved pair is recorded in the run summary
(`run_e4.py:298`). Recorded here for completeness, not as a defect.

---

# PART 3 — STATISTICS (audit item 2)

## 3.1 `atlas/stats.py::paired_bootstrap` — CLEAN

`atlas/stats.py:64-73`:

```python
d = a - b
rng = np.random.default_rng(seed)
idx = rng.integers(0, len(d), (n, len(d)))
bootstrap_means = d[idx].mean(axis=1)
lo, hi = np.percentile(bootstrap_means, alpha), np.percentile(bootstrap_means, 100.0 - alpha)
```

Checked against every failure mode in the brief:

- **Resamples PAIRS, not the two arms independently.** ✅ It forms the
  difference vector `d` first and resamples **indices into `d`**. The two arms
  are never resampled separately. This is the correct paired bootstrap.
- **Shape guard.** ✅ `if a.shape != b.shape: raise ValueError` (line 65-66).
- **Percentile axis.** ✅ `.mean(axis=1)` reduces over the *within-resample*
  axis, leaving `n` bootstrap means; percentiles are then taken over that 1-D
  array. Correct axis, no off-by-one.
- **CI endpoints.** ✅ `alpha = (100-ci)/2 = 2.5`, endpoints 2.5 and 97.5 for
  `ci=95`. Correct.
- **Determinism.** ✅ `np.random.default_rng(seed)`, seed defaults to 0.

**VERDICT: CLEAN. No bug found.**

## 3.2 `atlas/stats.py::mcnemar_paired` — CLEAN

`atlas/stats.py:103-107` builds

```
[[ (a &  b).sum(), (a & ~b).sum() ],
 [(~a &  b).sum(), (~a & ~b).sum() ]]
```

`statsmodels.stats.contingency_tables.mcnemar(table, exact=True)` uses the
off-diagonal cells `table[0][1]` and `table[1][0]` for an exact binomial test.
Here those are `n(a✓,b✗)` and `n(a✗,b✓)` — the correct discordant pairs. The
diagonal cells are irrelevant to the test and are correctly placed anyway.

- Shape guard present (line 100-101).
- `exact=True` is passed, matching "exact McNemar" in the spec.
- Degenerate case `n01 + n10 == 0` yields p = 1.0 from the binomial test, which
  is the right answer, not a crash.

**VERDICT: CLEAN. No bug found.** This is consistent with **N1**'s reported
`McNemar p = 1.000` at 44/100 vs 43/100 (a near-perfectly balanced discordance).

## 3.3 `atlas/stats.py::normalised_recovery` — CLEAN, matches spec

`atlas/stats.py:35-38`:

```python
spread = sr_oracle - sr_random
if spread < min_spread:
    return None
return (sr_fit - sr_random) / spread
```

`min_spread` defaults to `0.10`, matching implementation plan §8 and
`CLAUDE.md` §5. Returns `None` below threshold, including for negative spreads
(`spread < 0.10` catches those). **VERDICT: CLEAN.**

`scripts/make_tables.py:59-65` and `scripts/run_e1.py:202-206` both call it
without overriding `min_spread`, and both render `"—"` when it returns `None`
rather than formatting `None`. Correct.

## 3.4 Pairing is real in E0's on-disk results — VERIFIED FROM DATA

Independently recomputed from `atlas_out/`, not taken from any document:

| run | n paired | `init_block_pos_diff` mismatches between arms | baseline SR | chart SR | Δ |
|---|---|---|---|---|---|
| `e0_planning_n100` | 100 | **0** | 0.44 | 0.43 | −0.010 |
| `e0_planning_nas2` | 20 | **0** | 0.40 | 0.50 | +0.100 |

Episode indices are contiguous 0..N−1 with no duplicates in all four JSONL
files. The Δ values match **N1** (−1.0pp) and **N5** (+10.0pp) exactly.

**VERDICT: the pairing underlying N1 and N5 is genuine.** The arms really are
run on identical (init, goal) pairs.

## 3.5 🟡 MEDIUM — `analyze_n100.py` does not compute the N1 headline statistics it is cited for

`CLAIMS_MATRIX.md` **N1** cites `analysis_n100.json` as the evidence for
"delta −1.0pp, 95% CI [−9.0,+7.0], McNemar p=1.000". But
`scripts/analyze_n100.py::main` (lines 173-230) writes only:
`A1_knock_away_progress`, `A2_sr_by_bucket`, `A3_partial_stratified_kendall`,
`A3_catastrophic_episodes`, `A7_bridge`, `A8_pairing_verification`.

**It never calls `paired_bootstrap` or `mcnemar_paired` at all** (confirmed by
grep: the only call sites in the repo are `make_tables.py` and `run_e1.py`).

So the CI and the McNemar p-value in N1 were produced **somewhere else** — most
likely by an ad-hoc computation in a prior session — and the cited artifact does
not contain them.

- Severity: **MEDIUM** (provenance, not necessarily correctness)
- Threatened claim: **N1**
- Results on disk affected: **the numbers may well be right** (§3.1/§3.2 show the
  functions are correct, and §3.4 shows the pairing is real) — but they are
  **not reproducible from the artifact the claim cites**. A future session should
  recompute them from the JSONL and record where they came from.

## 3.6 🟢 LOW — `partial_kendall`'s p-value on residuals is not valid

`scripts/analyze_n100.py:84-86` residualises both `umf_mean` and binary
`success` on the controls via OLS, then runs `kendalltau` on the residuals.
Kendall's null distribution does not account for the estimated regression
coefficients, so `partial_p` is anticonservative. The **coefficients**
(−0.358, −0.374) that **N2** actually quotes are fine as a descriptive
semipartial statistic; only the accompanying p-value is suspect.

- Severity: **LOW**
- Threatened claim: **N2** (the point estimate stands; the significance claim
  on the *partial* correlation does not)
- Results on disk affected: **yes, `analysis_n100.json`** — but N2's headline
  p-values (`p < 1e-4`) are for the **unconditional** taus, which are computed
  correctly on raw data (`analyze_n100.py:97`).

## 3.7 🟢 LOW — `sr_by_bucket` silently drops out-of-range episodes

`scripts/analyze_n100.py:61-68` uses `pd.cut(..., bins=(0,80,120,300))`. Any
episode with `init_block_pos_diff > 300` becomes `NaN` and is dropped by
`groupby(..., observed=True)` without a warning, so the per-bucket `n` values
need not sum to 100.

- Severity: **LOW**
- Threatened claim: **N1** (supporting analysis only)
- Results on disk affected: `analysis_n100.json`'s `A2` block — check whether
  the bucket `n`s sum to the full episode count before quoting them.

---

# PART 4 — GATES (audit item 10, partial)

## 4.1 G1 — was genuinely rewritten, and now really does apply a chart and call the model. But it cannot catch the `restore_` defect.

**FACT, confirming the documentation.** `scripts/smoke_gates.py:51-133`. The
current `gate_g1` builds a `Chart`, calls `c0.apply_(predictor)`, and runs
`wm.forward_pred(...)` through a `_forward()` closure with a fixed
`torch.manual_seed(0)` synthetic context and action, comparing with
`torch.equal` (bit-identity, not `allclose`). It then calls `c0.restore_()` and
diffs the **whole state_dict** tensor-by-tensor against a pristine snapshot.

This is a real test. **`HANDOFF.md` §7.2's claim that G1 was rewritten on
2026-08-26 is VERIFIED**, and `CLAUDE.md` §0.1's statement that G1 "never
actually applies the chart" is confirmed **stale**.

**But — the gap.** G1 only ever tests **identity** charts (`Chart(predictor,
kind)` constructed from the pristine predictor). For an identity chart,
`restore_() == apply_()` and "re-apply this chart" is *indistinguishable from*
"restore pretrained weights". So G1 **structurally cannot detect** the
`restore_` defect in §2.5, and its passing must not be read as evidence for the
literal wording of **P-1**.

G1 also does not test `kind="full"` (only `ln_act` and `lora4`).

- Severity of the gap: **MEDIUM**
- Threatened claim: **P-1, S-2**
- Results on disk affected: n/a (a gate)

## 4.2 🔴 CRITICAL — G2 asserts nothing. The prequential gate is vacuous.

`scripts/smoke_gates.py:203-217`:

```python
umf_c0 = compute_umf(c0, wrapper, W_prime["encoder_output"], W_prime["actions"], ...)
umf_cx = compute_umf(cx, wrapper, W_prime["encoder_output"], W_prime["actions"], ...)

# For random data, just verify scores are computed without error.
if umf_cx is not None and umf_c0 is not None:
    pass
...
print(f"PASSED  (UMF c0={c0_str}, UMF cx={cx_str} on held-out W')")
```

**There is no assertion.** The `if ... : pass` is a no-op. G2 prints `PASSED`
unconditionally as long as `compute_umf` does not raise. It **never checks that
the over-refined chart `cx` fails to auto-win on `W'`** — which is the entire
thing the gate exists to check.

Two further defects compound it:

1. `W` and `W'` are i.i.d. `torch.randn` (lines 163-167), so `cx` has no
   learnable structure to over-fit and the scenario cannot discriminate even if
   an assertion were added.
2. The over-refine loop (`smoke_gates.py:196-201`) uses a **hand-rolled
   `_one_step_loss`** driving `wm.forward_pred` with a **zeroed proprio** and a
   1-frame context — i.e. exactly the code pattern the 2026-08-25 rollout fix
   removed from production. It is not the `_open_loop_rollout()` path
   `atlas_refine()` actually uses.

**Consequence:** claim **P-2** ("strict prequential order… no chart can win by
construction") and the "G2 passes" line in **S-2** have **zero test support**.
Every document asserting "G2 passes" is asserting only that the function ran to
completion.

- Severity: **CRITICAL** (a claimed correctness guarantee with no test behind it)
- Threatened claims: **P-2, C2, S-2**
- Results on disk affected: n/a directly — but every doc citing "G2 passes" as
  evidence for P-2 must be retracted or the gate rewritten.

## 4.3 🟠 HIGH — G5 does not test what CLAUDE.md §4 says it tests

`scripts/smoke_gates.py:418-434`:

```python
for seg in range(6):
    for ep in range(20):
        s1 = paired_seed(seg, ep, arm="atlas")
        s2 = paired_seed(seg, ep, arm="frozen")
        if s1 != s2: raise AssertionError(...)
```

`CLAUDE.md` §4 defines G5 as: *"Same seeds, two arms ⇒ identical initial states
and goals."* What this code actually checks is that `paired_seed()` ignores its
`arm` argument — which is **trivially true by inspection**, because `arm` is
never referenced in the function body (`atlas/streams.py:39-51`). It constructs
no environment, samples no init state, and samples no goal.

G5 is a tautology test. It cannot fail.

**Mitigating fact:** the property G5 *should* test I verified independently and
empirically against real data in §3.4 — E0's on-disk arms genuinely share
identical `init_block_pos_diff` per episode. So P-5 has real evidence; just not
from G5.

- Severity: **HIGH** (a gate that cannot fail is worse than no gate — it is
  cited as evidence)
- Threatened claims: **P-5, S-2**
- Results on disk affected: n/a (a gate)

## 4.4 G4 — confirmed not wired into `main()`, honestly reported as skipped

**FACT, confirming the documentation.** `scripts/smoke_gates.py:463-527`:
`main()` dispatches G5, then G1/G2/G3a/G3b/G6 behind the model load, and then:

```python
if run_all or run == "G4":
    print("\nNote: G4 requires a running Push-T environment.")
    print("Integrate this gate with the jepa-wms env setup (see README Setup).")
    print("Skipping G4 in headless mode.")
```

`gate_g4` is **never called** — `main()` prints a skip message instead. The
function itself (lines 378-415) has been API-corrected to legacy gym
(`env.seed(seed)` + `env.reset()` + 4-tuple `step()` + `obs["visual"]`), so it
would probably run if wired, but no `env_factory` is ever constructed.

**Confirms the project's own honest reporting.** Claim **P-4** (regimes are real
shifts, "verified by gate G4") therefore rests on the separate
`REGIME_DESIGN_REVIEW.md` analysis, **not** on G4.

Note also that `gate_g4`'s own pass criterion is extremely weak: it compares
mean pixel values and fails only if `mean|diff| < 1e-6`. It is a "not
byte-identical" test, not the "differ visibly **and statistically**" test
CLAUDE.md §4 describes.

- Severity: **MEDIUM** (unrun; and weak even if run)
- Threatened claim: **P-4, S-2**

## 4.5 G3a / G3b — exercise the real `Expander` logic; G3b's case is as crude as the docs concede

**FACT.** Both gates (`smoke_gates.py:245-373`) call `Expander.record()` and
`Expander.maybe_expand()` **directly on the real code**, with real assertions
(`if outcome != "committed": raise` / `if outcome != "rejected_score": raise`).
Unlike G2, these gates can actually fail. Good.

Assessment against **C2** ("verification is non-vacuous"):

- **G3b is not sufficient to establish C2.** Its unfixable case is i.i.d.
  `torch.randn` chunks with zero learnable structure (`smoke_gates.py:337-340`).
  A probe that rejected *nothing else* would still pass G3b. It rules out the
  degenerate "always commits" failure, nothing more. The project's own
  docstring concedes exactly this, and that concession is **accurate**. In
  particular G3b does **not** test the failure mode E0 actually discovered — a
  chart that improves UMF while planning no better (N1/N2's dissociation) —
  which is the failure mode C2 most needs to survive.
- **G3a's "regime shift" is a perturbation of the predictor's own weights**
  (`smoke_gates.py:281-284`, `REL_SCALE=0.3`), i.e. a shift guaranteed to lie
  inside the candidate's own parameter family. It proves the fitting machinery
  works; it does not prove the probe fires on a real physics shift.
- Both gates generate their synthetic ground truth via `_predict_one_step`
  (`smoke_gates.py:219-243`) — a hand-driven `forward_pred` with **zeroed
  proprio** — while `compute_umf`/`_fit_candidate` score with
  `_open_loop_rollout` + a **synthetic non-zero** `proprio_ctxt`. The
  data-generating path and the scoring path therefore disagree about proprio.
  This does not invalidate the commit/reject logic test, but it means the UMF
  magnitudes these gates report are not comparable to production ones.

- Severity: **MEDIUM** (gate scope, honestly disclosed)
- Threatened claim: **C2**
- Results on disk affected: n/a (gates)

## 4.6 🟢 LOW — G6 is correct but narrow; and `smoke_gates.py` loads the model from the *remote* hub

- G6 (`smoke_gates.py:436-460`) constructs an exactly-static chunk and passes
  `motion_gate=0.0`, so it exercises the `observed_displacement <= motion_gate`
  branch at `score.py:88`. It **does not** independently exercise the
  `displacement == 0.0` denominator fallback at `score.py:108-111`, since the
  gate branch returns first. Claim **G6/S-2** holds for what it tests.
- `smoke_gates.py:497-500` calls
  `torch.hub.load("facebookresearch/jepa-wms", ...)` — the **remote** repo —
  whereas every production script uses `source="local"` against the patched
  local cache at `hub/hub/facebookresearch_jepa-wms_main` (`run_e4.py:49`,
  `run_e0_planning.py`). Depending on `TORCH_HOME` this may resolve to the same
  patched cache or to an unpatched fresh clone. An inconsistency worth removing
  before citing gate results.

---

# PART 5 — WORLD-MODEL ROLLOUT (audit item 4, partial)

## 5.1 The 2026-08-25 rollout fix — the parts I could verify are CORRECT

**Claim S-1** asserts four defects were fixed. Checking each against
`atlas/score.py` and the checkpoint's own
`hub/.../modelcustom/simu_env_planning/vit_enc_preds.py`:

1. **"5× wrong time base"** — ✅ **FIXED and guarded.**
   `_open_loop_rollout` (`score.py:208-216`) now raises `ValueError` if
   `actions.shape[-1] != enc_pred_wm.action_dim` (10 = frameskip 5 ×
   raw_action_dim 2). Raw per-step actions can no longer be fed in silently.
   It delegates to `EncPredWM.unroll()`, which chunks actions correctly
   (`vit_enc_preds.py:315-320`). Every caller I traced
   (`harness_e4.py:296`, `run_e0_planning.py:341`) reshapes to
   `[n_raw // frameskip, frameskip*2]` before calling. ✅
2. **"proprio hard-zeroed"** — ✅ **FIXED.** `_make_z_ctxt` (`score.py:160-171`)
   builds a real `TensorDict` with the caller's `proprio_ctxt`, and
   `EncPredWM.unroll` propagates it forward via `proprio_mode`
   (`vit_enc_preds.py:335-345`). Real encoded proprio is threaded through from
   `world_model.encode()` at `harness_e4.py:288-290` and
   `run_e0_planning.py:333-335` (`proprio_enc[:, 0:1]`). ✅
3. **"context window fixed at 1 frame instead of ctxt_window=2"** — ⚠️
   **NOT what the doc implies, but CORRECT anyway.** `_make_z_ctxt` still builds
   a **tau=1** context: `visual = z0.reshape(1, 1, 1, grid, grid, D)`
   (`score.py:170`). `EncPredWM.unroll` then slides `vid_feats[:, -ctxt_window:]`
   over an accumulating buffer (`vit_enc_preds.py:327-331`), so the **first**
   prediction genuinely sees only 1 context frame, and subsequent ones see 2.

   **This is nevertheless correct, because it matches deployment exactly.**
   `GC_Agent.act()` (`gc_agent.py:170-176`) encodes a **single** observation
   TensorDict and hands that tau=1 latent straight to `plan()` → `unroll()`.
   So the planner also plans from a 1-frame context. UMF scoring and CEM
   planning use the same context depth. ✅ **Clean — but note that
   `CLAUDE.md` §0.1's wording ("fixed… context window") overstates it: the
   context is still 1 frame; what changed is that the sliding window is now
   delegated to the checkpoint's own `unroll()`.**
4. **"a correct implementation shipped in the wrapper and went unused"** — ✅
   `_open_loop_rollout` is now built on `EncPredWM.unroll()`. ✅

`_open_loop_rollout`'s output alignment is also consistent with tau=1:
`vid[1:]` drops exactly one prepended context frame (`score.py:231-232`),
yielding `[T, N, D]` compared against `encoder_output[1:]`. ✅

## 5.2 UMF numerator/denominator — CLEAN

`score.py:101-113`. Numerator `(z_hat - z[1:]).pow(2).sum()` and denominator
`(z[1:] - z0).pow(2).sum()` are summed over the **same** `[T, N, D]` extent, so
the ratio is dimensionless and matches the C3 formula in the module docstring.
The `displacement == 0.0` fallback returns `None` rather than dividing by zero.
No broadcasting hazard: both operands are explicitly `[T, N, D]`
(`z0.unsqueeze(0)` broadcasts intentionally over T). **CLEAN.**

## 5.3 Frozen-encoder check (done by direct `requires_grad` inspection, not by name)

**FACT.** `scripts/run_e4.py:165-166` and `run_e0_planning.py:451-452` call
`p.requires_grad_(False)` on `wm.encoder.parameters()`. Independently, the
encoder is only ever invoked inside `EncPredWM.encode()`, which is decorated
`@torch.no_grad()` (`vit_enc_preds.py:354`), and every production call site
additionally wraps it in `with torch.no_grad():`
(`harness_e4.py:287`, `run_e0_planning.py:333`). The `encoder_output` tensors
that reach `atlas_refine` / `_fit_candidate` are therefore **detached constants**.

Optimizer param groups were checked directly, not by variable name:
- `atlas_refine`: `params = [p for n, p in predictor.named_parameters() if n in chart._param_names]` (`loop.py:222`)
- `_fit_candidate`: same construction (`expand.py:201`)
- `AdaJEPA`: same construction (`adajepa.py:89`)
- `harness_e4.py:308-310`: same construction

**No path constructs an optimizer over encoder parameters, and no gradient can
reach the encoder.** ✅ **CLEAN — claim P-3's "frozen DINOv2 encoder" and
CLAUDE.md §1.2 hold in every path I traced.**

One residual note: `loss.backward()` in `atlas_refine`/`_fit_candidate`
populates `.grad` on **all** predictor params in the graph that have
`requires_grad=True`, not only the chart's. Only the chart's are stepped, so
this is numerically harmless, but stale `.grad` accumulates on non-chart
predictor params. Severity **LOW**; no claim threatened.

---

# PART 6 — ROUTING & HYSTERESIS

## 6.1 🟠 HIGH — the hysteresis margin is exactly inert for any 2-chart library

`atlas/router.py:94-101`:

```python
current_score = valid.get(current_idx)
if current_score is not None:
    spread = max(valid.values()) - min(valid.values())
    relative_gap = (current_score - best_score) / spread if spread > 0 else 0.0
    if relative_gap < hysteresis:
        return current_idx, ...
```

The 2026-08-25 change (`T12 #5`) replaced the absolute margin with one
normalised by the replan's own score spread. With a **2-chart library**:

- if the current chart is the argmin, `relative_gap == 0.0` → keep (correct);
- otherwise the current chart is by definition the **max**, so
  `relative_gap == (max − min)/(max − min) == 1.0`, which always exceeds
  `m = 0.05` → **always switch**.

So for a 2-chart library the hysteresis mechanism is **exactly disabled**. It
only has any effect with ≥3 charts, and even then only for near-ties.

`CLAUDE.md` §1.7 lists `m = 0.05` as a **fixed non-negotiable hyperparameter**.
The normalisation changed its meaning without changing its value, and the
consequence — total inertness at K=2 — is not documented anywhere I found.

- Severity: **HIGH** (a named mechanism is silently a no-op in the most common
  configuration)
- Threatened claims: **C1, N7, RQ1**
- Results on disk affected: **YES.** Every `*_posthysteresis` E2 directory
  (`e2_R1_posthysteresis`, `e2_R2_posthysteresis`, `e2_R1_lora4_posthysteresis`)
  ran under this rule — that is the basis of **N7** (UMF 0.833 vs S-dyn 0.570).
  `e2_confusion_matrix` (**N6**) uses a 3-chart library, where the rule is
  non-trivial but still very permissive. Also applies to **all of E4** if
  launched with `expansion_start_library="full"` (library = `{c0, chart_B}` =
  exactly K=2).

## 6.2 Router internals otherwise CLEAN

- `_score_all` (`router.py:106-128`) applies each chart before scoring and the
  gate is a pure function of `encoder_output`, so either **all** charts return
  `None` together or **none** do — meaning `valid.get(current_idx)` is never
  spuriously `None`. ✅
- `_route_oracle` (`router.py:200-219`) raises loudly on a missing label or an
  out-of-range index rather than silently defaulting. ✅
- `_route_random` (`router.py:194-197`) takes a seeded `random.Random`; E4
  threads one per episode (`run_e4.py:251`). ✅ (Unused at `router="umf"`.)
- `_e1_score` / `_sdyn_score` correctly run the **full** rollout and then index
  `z_hat[0]`, rather than trying to encode a single raw action — the docstrings
  explain why, and the reasoning checks out against
  `VideoWM.encode_act`'s chunking. ✅

---

# PART 7 — SEEDING (audit item 3, partial)

## 7.1 `paired_seed` — CLEAN

`atlas/streams.py:39-51`. SHA-256 of `f"seg={segment_idx}:ep={episode_idx}"`,
first 4 bytes big-endian → an int in `[0, 2^32)`. The `arm` parameter is
accepted and deliberately unused, so **all arms share the seed by
construction**. Range is valid for both `np.random.RandomState` and legacy
gym's `env.seed()`. ✅

## 7.2 Draw-count symmetry across arms — CLEAN where it matters

The brief's specific worry — one arm consuming a **different number of draws**
from a shared generator, silently desynchronising pairing — was checked:

- `rs = np.random.RandomState(spec.seed)` is constructed **fresh inside each
  episode** (`harness_e4.py:176`; `run_e0_planning.py:238`). Even though
  `sample_dataset_init_goal`'s retry loop consumes a variable number of draws,
  the generator is discarded at end of episode, so **no drift can accumulate
  across episodes** and every arm sees the identical init/goal. ✅ Confirmed
  empirically in §3.4 (0 mismatches over 100 + 20 episodes).
- The CEM generator **is** shared and **does** desynchronise (§2.11), but it is
  planner noise, not episode construction. Documented above as LOW.
- `_inject_lora`'s global-RNG draw (§2.12) is the one place where chart
  construction itself consumes randomness. LOW at `kind="ln_act"`.

**VERDICT: the pairing guarantee P-5 depends on is intact.**

## 7.3 🟢 LOW — `stream_s2`'s seed key mixes offsets in a collision-prone way

`atlas/streams.py:86-87` computes
`paired_seed(seg_idx + stream_seed_offset*1000, ep_idx + seed_run*10_000)`.
Because `episodes_per_segment` defaults to 20 and the multiplier is 10,000,
collisions between `seed_run` values are impossible in practice. Recorded only
because the scheme is fragile if `episodes_per_segment` ever exceeds 10,000.
No action needed.

---

# PART 8 — SUMMARY OF VERDICTS

## Clean verdicts (verified, no bug found — these are load-bearing)

| Path | Verdict | Where |
|---|---|---|
| Multi-replan observation/proprio context re-encoding | **CLEAN** — no stale-context bug; N5 not invalidated by that mechanism | §1.2 |
| CEM searches per episode at each `nas` | **CLEAN** — 1 at nas=6, 3 at nas=2, confirmed against JSONL | §1.1 |
| Goal / success criterion / step budget across `nas` | **CLEAN** — identical | §1.3 |
| `stats.paired_bootstrap` resamples PAIRS | **CLEAN** | §3.1 |
| `stats.mcnemar_paired` 2×2 table + exact test | **CLEAN** | §3.2 |
| `stats.normalised_recovery` `None` below 0.10 | **CLEAN**, matches spec | §3.3 |
| E0 arm pairing on real data (0 mismatches, n=100 & n=20) | **CLEAN** | §3.4 |
| `paired_seed` arm-independence & draw-count symmetry | **CLEAN** | §7.1–7.2 |
| Frozen encoder — no optimizer group, no gradient path | **CLEAN** | §5.3 |
| Rollout time base / proprio threading / output alignment | **CLEAN** | §5.1–5.2 |
| G1 was genuinely rewritten and now applies a chart + calls the model | **CONFIRMED** | §4.1 |
| G4 is not wired into `main()`, honestly reported as skipped | **CONFIRMED** | §4.4 |
| `expand.py`'s fit-set / verify-set are disjoint tensors | **CLEAN** (caller contract unasserted) | §2.7 |

## Bugs that would silently corrupt an E4 run if it is launched as-is

Ordered by how badly they corrupt the result without crashing. **All of these
are free to fix — E4 has never run.**

1. **CRITICAL — `harness_e4.py:216-217` hard-codes `next_encoder_output=None`,
   so `maybe_expand()` is never called and arm 6 (ATLAS) can never commit a
   chart.** Arm 6 silently degenerates into arm 4. RQ3 would report
   "ATLAS: 0 commits" as a finding. Threatens **C2, RQ3, L-1, C4**. (§2.1)
2. **CRITICAL — `AdaJEPA.reset()` is never called in production
   (`atlas/adajepa.py:94` has no caller outside `smoke_e4.py:142`), so arm 2 is
   behaviourally identical to arm 3.** The persistence rung of the central
   table becomes vacuous. Threatens **L-1, RQ4, C4**. (§2.2)
3. **CRITICAL (suspected, needs one measurement) — the motion gate is
   calibrated on whole-trajectory displacement (`run_e4.py:182-189`,
   `traj_len=10`, `num_trajs=3`) but applied to 1-model-step chunks at
   `nas=1`.** If it gates everything, routing freezes, no strike is ever
   recorded, and arms 4/5/6/7 collapse into arm 1 while still producing a
   plausible table. **Measure before launching.** Threatens **C1, C2, RQ3,
   RQ4, L-1**. (§2.3)
4. **HIGH — `run_e4.py:167-168` freezes all predictor params; only
   `AdaJEPA.__init__` re-enables them.** ATLAS arms therefore work only if an
   AdaJEPA arm ran earlier in the same process. `--profile` (which forces
   `--arms atlas`) and any arm-subset re-run crash at the first refine, possibly
   hours in. (§2.4)
5. **HIGH — `expand.py:131-133` selects the incumbent "best chart" by argmin on
   the verification chunk itself**, biasing the commit test conservatively.
   Threatens **C2, RQ3**. (§2.6)
6. **HIGH — `router.py:96-99`'s spread-normalised hysteresis is exactly inert
   for a 2-chart library**, which is exactly E4's default
   (`expansion_start_library="full"` ⇒ `{c0, chart_B}`). The `m=0.05`
   non-negotiable is a no-op. Threatens **C1**. (§6.1)
7. **HIGH — `chart.restore_()` does not restore pretrained weights for `ln_act`
   / `full` (`chart.py:126-127`)**, leaving the predictor permanently dirty
   after `harness_e4.py:252`. Currently masked by full-overwrite, but falsifies
   the literal wording of **P-1**, and G1 cannot detect it. (§2.5, §4.1)
8. **MEDIUM–HIGH — `make_tables.py:126` pairs arms by equal *length*, not equal
   *key set*.** With E4's resume support this can silently produce a misaligned
   "paired" bootstrap and McNemar. Threatens **L-1, RQ4, P-5**. (§2.10)
9. **MEDIUM — RQ4's first-visit-vs-final-revisit delta cannot be paired**,
   because `streams.py:86` puts `segment_idx` in the seed key; and
   `make_tables.py:110-116` reports the two as bare means with no CI, violating
   `CLAUDE.md` §5. Threatens **RQ4, C4**. (§2.9)
10. **MEDIUM — arms 4/5/6 skip refinement whenever `current_idx == 0`
    (`harness_e4.py:305`) while arms 2/3 always refine**, so the arm-3→arm-4
    rung differs by two mechanisms, not one. Threatens **L-1**. (§2.8)
11. **LOW — E4 is not reproducible**: the CEM generator is seeded once at agent
    construction and never re-seeded, and resume does not replay consumed draws.
    (§2.11)

## Bugs affecting numbers already published in the project docs

1. **HIGH — spread-normalised hysteresis was inert (K=2) or near-inert (K=3)
   in every E2 `*_posthysteresis` run.** This is the basis of **N7** (+26.3pp,
   UMF 0.833 vs S-dyn 0.570) and contributes to **N6**. The mechanism
   `CLAUDE.md` §1.7 pins at `m=0.05` did not operate as specified.
   Affected: `atlas_out/e2_R1_posthysteresis`, `e2_R2_posthysteresis`,
   `e2_R1_lora4_posthysteresis`, `e2_confusion_matrix`. (§6.1)
2. **HIGH — `expand.py`'s conservative incumbent-selection bias** applied to the
   only real expansion demonstration on disk, `atlas_out/e2_R2_cellB_q1`
   ("3 charts committed"), which is the evidence for **N9**. The bias is
   conservative, so the count is if anything under-stated — but it is not the
   quantity the claim describes. (§2.6)
3. **MEDIUM — `analyze_n100.py` never computes the CI or McNemar p that
   `CLAIMS_MATRIX.md` N1 cites `analysis_n100.json` for.** Those numbers came
   from somewhere else and are not reproducible from the cited artifact.
   (The functions themselves are correct and the pairing is real, so they are
   probably right — but the provenance is broken.) (§3.5)
4. **MEDIUM — N5's +10.0pp confounds closed-loop feedback with a 3× increase in
   CEM search compute**, because `plan_length` stays pinned at `horizon=6`
   regardless of `steps_left`. The paired within-nas=2 comparison is fair; the
   nas=6-vs-nas=2 narrative is not. (§1.4–1.5)
5. **CRITICAL for documentation, not for numbers — every document asserting
   "G2 passes" (`CLAUDE.md` §0.1, `HANDOFF.md` §7.2, **S-2**) is asserting
   nothing.** G2 contains no assertion (§4.2). **P-2** has zero test support.
6. **HIGH for documentation — "G5 passes" is likewise vacuous** (§4.3). The
   underlying property is nevertheless true, verified independently from the
   E0 JSONL (§3.4).
7. **LOW — `analyze_n100.py`'s partial-Kendall p-value is invalid** (§3.6) and
   `sr_by_bucket` can silently drop episodes above 300px (§3.7). Affects
   supporting analysis in **N1/N2**, not the headline taus.
8. **Documentation drift — `CLAUDE.md` §0.1's claim that `gate_g1` "never
   actually applies the chart" is STALE.** G1 was genuinely rewritten and now
   does. `EXPERIMENT_STATUS.md`'s note that G1 was rewritten on 2026-08-26 is
   the accurate one. (§4.1)

---

# PART 9 — PASS 2: S-5 train/deploy mismatch, E4 regime-persistence safety, E2 leakage, cost-ranking mechanism

## 9.1 🔴 CONFIRMED — S-5 (`OPUS_REMAINING_TASKS.md` item 10) is TRUE on all four sub-claims. `closed_loop`'s "clean rejection" framing is not supportable as written.

Read `scripts/run_e0.py::main()` (lines 455-878) and the `closed_loop` branch of
`load_regime_trajectories()` (lines 155-393) in full, specifically the
argparse defaults and the call sites that consume them.

**Sub-claim (a) — "collected with the frozen predictor: on-policy for c0,
off-policy for the chart under training" — TRUE.**

`run_e0.py:604-618`:

```python
collector_agent = None
if args.data_source == "closed_loop":
    ...
    collector_agent = GC_Agent(
        build_cfg(args.collect_num_samples, args.collect_iterations, horizon=6,
                  num_act_stepped=1),
        wrapper, dset=None, preprocessor=prep)
```

`collector_agent` is built once, from `wrapper` (the wrapper around `wm`)
**before** the per-`kind` fine-tuning loop starts (that loop begins at
`run_e0.py:685`). The predictor is only ever mutated inside that loop
(`wm.predictor.load_state_dict(pristine_predictor_state)` at line 690, then
`run_e0_finetune()` at line 754). The code's own comment at
`run_e0.py:600-603` states the design intent explicitly: *"Built from the
PRISTINE predictor, before any chart is applied: the point of on-policy
collection is data showing what the FROZEN model gets wrong."*

**Aggravating detail beyond what S-5 alleges:** `train_trajectories` and
`val_trajectories` are collected exactly **once per regime**
(`run_e0.py:642-649`), **outside** the `for kind in args.kinds:` loop
(`run_e0.py:685`). So the identical closed-loop trajectories — generated
entirely by the frozen c0 predictor's own CEM plans, reacting only to c0's
own mistakes — are reused verbatim to fine-tune **all three** chart kinds
(`ln_act`, `lora4`, `full`) for that regime. None of the three charts' own
corrections ever influence the data they are trained on. This is a single
non-reactive round, not a DAgger-style loop, despite the collector's
"reactive" framing in the `--data-source` help text (`run_e0.py:517-528`,
"CLOSED-LOOP... contains the model's own overshoot AND the correction it
then attempts" — true only for c0, not for the chart being fit).

**Sub-claim (b) — "CEM 100×10 at collection vs 300×30 at planning-eval" — TRUE.**

- Collection: `--collect-num-samples` defaults to `100` (`run_e0.py:532-538`),
  `--collect-iterations` defaults to `10` (`run_e0.py:539-541`), both fed
  directly into `build_cfg()` at the `collector_agent` construction
  (`run_e0.py:613`). The `--collect-num-samples` help text is candid about
  why: *"300x30 would be ~9x this, putting a 28-trajectory collection into
  GPU-hours. Deviation from the validated planner config — record it with
  any result."*
- Eval: `scripts/run_e0_planning.py:386` (`--num-samples`, default `300`) and
  `:389` (`--iterations`, default `30`) — the substrate's own validated
  config, per that file's own docstring at line 17
  (`num_samples=300, iterations=30, ..., num_act_stepped=6`) and per
  `CLAUDE.md`'s T6 resolution.

So collection uses **3× fewer iterations and 3× fewer samples per
iteration** than the config the resulting charts are then evaluated under —
i.e. **~9× less CEM search budget**, exactly as S-5 alleges.

**Sub-claim (c) — "nas=1 at collection vs nas=6 at eval" — TRUE, and stronger
than "a default": it is hardcoded, not configurable.**

`run_e0.py:613-614`: `build_cfg(..., horizon=6, num_act_stepped=1)` — `1` is a
literal in the call, not sourced from any argparse flag; there is no
`--collect-num-act-stepped` option at all. The comment at `run_e0.py:610-611`
confirms this is deliberate: *"num_act_stepped=1 is the whole point: it
forces a replan every model chunk, which is what makes the collected
trajectory reactive."* Eval's default is `--num-act-stepped 6`
(`run_e0_planning.py:391`). Per Part 1 §1.1 of this file (already verified
against real JSONL on disk), `num_act_stepped=6` means **exactly 1 CEM search
for the whole 30-step episode** at eval, vs. **6 CEM searches** (one per
5-raw-step chunk) at `nas=1` collection. So collection is maximally
closed-loop (replans every chunk) while eval is maximally open-loop (plans
once and executes blind) — the two conditions are opposite ends of the
replan-frequency spectrum, not merely "different."

**Sub-claim (d) — "only 20 trajectories" — TRUE (as the default; a per-run
override was not checked, see caveat below).**

`run_e0.py:474-481`: `--num-train-trajs` defaults to `20`. Help text confirms
this was sized for real (`dataset`) replay data volume, not for
`closed_loop`'s per-chunk-CEM cost: *"Bumped from the original 3 now that
--data-source=dataset (T9) gives real, diverse trajectories... intended for
Modal (24GB)."* Nothing in the `closed_loop` branch scales this down further,
so by default 20 full CEM-driven episodes (6 replans × 100×10 CEM searches
each = 120 CEM searches per training trajectory) are collected per regime.
`--num-val-trajs` defaults to `8` (`run_e0.py:489-493`).

**Caveat (not verified this pass):** these are argparse *defaults*. Whether
the actual `closed_loop` run(s) that produced the numbers in
`E0_RESULTS.md`/`E0_RECOVERY_PLAN.md` invoked `run_e0.py` with these defaults
or with explicit overrides was **not checked** — no shell history, Modal launch
config, or run-log capturing the actual command line was located in this
pass. If a future session can find the actual invocation (Modal launch
script, wandb run config, or a `--collect-*`/`--num-train-trajs` value
recorded in `atlas_out/e0/*closed_loop*` output), that should be checked
before citing this section as proof the specific numbers on disk used these
exact values — only that the **code's own defaults**, which is what an
unqualified re-run would use, match S-5's allegation exactly.

**VERDICT: all four S-5 sub-claims are TRUE against the code as written; none
is refuted.** `OPUS_REMAINING_TASKS.md` item 10's characterization — that
`closed_loop` was "tested with a broken instrument," not cleanly rejected —
is the technically accurate framing. **`E0_RECOVERY_PLAN.md`'s apparent
framing of `closed_loop` as a clean negative result should not be cited
without this caveat**: a chart trained on data collected with ~9× less CEM
search and 6× less replanning than the config it is then judged under is
confounded on capability *and* task-distribution grounds before its
UMF/planning numbers are even examined. This does not establish that
`closed_loop` charts would have succeeded under matched compute — only that
the experiment as run cannot distinguish "closed-loop replay doesn't help"
from "the collector's own CEM budget was too weak to produce competent
demonstrations."

- Severity: **CRITICAL for the write-up** (a standing claim's evidentiary
  basis is weaker than stated), **not a code bug** — every piece of behaviour
  above is deliberate and disclosed inline in `run_e0.py`'s own comments and
  help text. This is a **framing/interpretation gap in the downstream docs**,
  not a silent-corruption bug in the collector itself.
- Threatened claims: **S-5 itself (now CONFIRMED, not merely alleged)**, and
  any claim in `E0_RECOVERY_PLAN.md` that cites `closed_loop`'s failure as
  evidence against on-policy/reactive data collection as a mechanism.
- Results on disk affected: whichever `atlas_out/` directory holds the
  `closed_loop` P4 run(s) referenced in `E0_RESULTS.md` — **not re-verified
  against a specific directory this pass**; a future session should locate it
  and attach this finding directly to that artifact.

## 9.2 ✅ CLEAR — E4 regime-persistence safety check: `PushTEnv.reset()` fully rebuilds the pymunk space and all shapes every call. R1's `friction=2.0` CANNOT leak into a subsequent R0 episode.

This was flagged in pass 1 as *"the single most important thing to check
before E4 is ever launched."* Read `PushTEnv.reset()` and `PushTEnv._setup()`
in the hub cache actually used at runtime:
`hub/hub/facebookresearch_jepa-wms_main/evals/simu_env_planning/envs/pusht_env/pusht_env.py`
(confirmed this is the copy `run_e0.py`/`run_e0_planning.py`/`run_e4.py` all
import via `sys.path.insert(..., hub_path)` and
`torch.hub.load(hub_path, ..., source="local")` — not the `.venv` site-packages
or `vendor/jepa-wms` copies, which are untouched by this fix path and not what
executes).

`pusht_env.py:432-433`:

```python
def reset(self, **kwargs):
    self._setup()
```

`_setup()` is called **unconditionally, first thing, on every `reset()`
call** — not conditionally, not only on the first call. `_setup()`
(`pusht_env.py:684-716`) does:

```python
def _setup(self):
    self.space = pymunk.Space()          # <-- BRAND NEW Space object
    ...
    self.agent = self.add_circle((256, 400), 15)     # <-- BRAND NEW Body+Shape
    self.block = self.add_shape(self.shape, (256, 300), 0, ...)  # <-- BRAND NEW Body+Shape
```

`add_circle()` (`pusht_env.py:723-730`) constructs a fresh
`pymunk.Body`/`pymunk.Circle` pair and calls `self.space.add(body, shape)` —
i.e. the **old** `self.agent`/`self.block` Python objects (whatever
`shape.friction`/`shape.elasticity` a prior `PhysicsRegime._apply_physics()`
call had mutated on them) are **discarded outright and become unreferenced**;
`self.agent`/`self.block` are rebound to genuinely new pymunk objects created
in a genuinely new `pymunk.Space()`.

Cross-checked against `atlas/regimes.py:99-102` (`PhysicsRegime.reset()`):

```python
def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)   # <-- rebuilds space/shapes FIRST, fresh defaults
    self._apply_physics()                   # <-- THEN applies (or doesn't) this regime's shift
    return obs, info
```

The ordering is exactly right for safety: every episode's shape mutation is
applied to that episode's **own freshly-created** shapes, and R0's
`_apply_physics()` (`regimes.py:106-107`, `if not self._cfg: return`) is a
no-op — it does not need to "reset" friction to 0 because `_setup()` already
created the new shapes at pymunk's un-set default. Verified the default is
genuinely 0 (not 1, despite the misleading `body.friction = 1` lines at
`pusht_env.py:726`/`777`/etc — those set a nonexistent/inert attribute on
`pymunk.Body`, which has no physically-meaningful `.friction`; only
`shape.friction`, which is never set at construction, is what pymunk's
collision solver actually reads, and it defaults to `0.0`). This independently
confirms `atlas/regimes.py:60-65`'s own comment ("both were unset, defaulting
to 0.0, in the shipped env") against the checkpoint's actual source, which
pass 1 had flagged as an **unverified DOC CLAIM**.

**VERDICT: SAFE. There is no regime-persistence bug.** Whether `E4`'s stream
driver constructs a fresh `PushTEnv`/`PhysicsRegime` per segment or reuses one
`base_env` instance across regime switches is **irrelevant to this
question** — every `reset()` call, on any instance, discards and rebuilds the
entire physics world from scratch before the next regime's config is
applied. An R1 episode's `friction=2.0` cannot survive past that episode's
own `reset()` boundary under any code path that goes through
`PhysicsRegime.reset()` → `PushTEnv.reset()` → `PushTEnv._setup()`.

- Severity: **n/a (clean)** — recorded as a resolved OPEN QUESTION, not a bug.
- Threatened claim: **none** — this *removes* a threat to **RQ4/L-1/C4** (the
  S2 alternating-regime stream) that pass 1 had left open.
- Caveat: this only covers the **physics-shape** persistence question that
  was flagged. It does not re-verify `VisualCorruption`'s persistence (a
  `gym.ObservationWrapper`, unaffected by `_setup()`'s space rebuild by
  construction — it rewrites `observation()` per call, not shape state — and
  was already the subject of a separate, already-documented finding in §1.6).
  It also does not verify that `run_e4.py`/`atlas/streams.py` actually calls
  `PhysicsRegime.reset()` (not `.env.reset()` directly, which would skip
  `_apply_physics()` entirely and silently run every episode as R0) at every
  regime transition — that is a **different**, still-open question about the
  *caller*, not about `PushTEnv`/`PhysicsRegime` themselves, and was out of
  this check's scope.

## 9.3 ✅ CLEAR — `scripts/run_e2.py`'s N9 diagnostic (`e2_R2_cellB_q1`/`e2_R2_cellC_q1`) verifies on genuinely unseen data. No leakage.

Read `scripts/run_e2.py` in full (500 lines, previously unread per pass 1's
Priority 6). The relevant loop is `main()`'s per-episode body,
`run_e2.py:296-339`:

```python
for ep, traj in enumerate(tqdm(trajs, ...)):
    ...
    expander.record(_best_umf(umf_info), enc, acts, proprio_ctxt=proprio_ctxt)   # traj = trajs[ep]
    if ep + 1 < len(trajs):
        nxt = trajs[ep + 1]
        outcome = expander.maybe_expand(
            probe_library, wrapper, nxt["encoder_output"], nxt["actions"],
            motion_gate, next_proprio_ctxt=nxt["proprio"][0:1].unsqueeze(0))
```

At the point `maybe_expand()` is called for episode `ep`, `nxt = trajs[ep+1]`
has **not yet** been passed to `expander.record()` — that only happens on the
**next** loop iteration (`ep+1`'s own top-of-loop `record()` call, using
`traj = trajs[ep+1]` at that point). So `expander._deficit_chunks` at the time
of verification for episode `ep` contains only `trajs[0..ep]`, and the
verification chunk `trajs[ep+1]` is genuinely chronologically and
set-theoretically disjoint from it. This matches (and now confirms with a
concrete caller) pass 1's §2.7 finding that `expand.py`'s internal fit/verify
split is clean but the disjointness is a caller obligation.

**Confirmed the trajectories themselves are independent draws, not slices of
one continuous rollout:** `trajs = load_regime_trajectories(..., num_trajs=args.episodes, ...)`
(`run_e2.py:269-273`) returns one dict per `traj_idx` from
`run_e0.py:199-393`'s loop, each seeded as
`seed = seed_base + traj_idx * max_tries + attempt` (`run_e0.py:203`) — a
disjoint window of `max_tries` seed values per `traj_idx`, so `trajs[ep]` and
`trajs[ep+1]` are two independently-collected, independently-seeded
trajectories (different init physics state, different random target/aim),
not two windows cut from one shared episode. There is no shared tensor
between them prior to the loop.

One structural note, not a bug: because each `trajs[ep+1]` chunk is used
**once** as a held-out verification set (at iteration `ep`) and is then
folded into the deficit pool on iteration `ep+1`, a chunk that helped verify
an earlier commit can later become training data for a subsequent commit.
This is the standard, correct behaviour for a rolling online verification
scheme (each verification uses only data not yet trained on *at the time of
that decision*) — it would only be leakage if a chunk were used to verify a
commit *and* had already been fit on, which does not happen here.

**VERDICT: `atlas_out/e2_R2_cellB_q1`'s "3 charts committed" (the basis of
N9) is a valid demonstration of C2 on unseen data — not a leakage artifact.**
This resolves pass 1's Priority 6 open question in the favourable direction.

- Severity: **n/a (clean)** — resolves an OPEN QUESTION, does not create a
  new finding.
- Threatened claim: **N9** — this removes a threat pass 1 had left open, and
  strengthens the existing §2.6 finding (the *conservative* incumbent-bias in
  `maybe_expand()`) as the only remaining caveat on N9, rather than leakage.

## 9.4 `scripts/analyze_cost_ranking.py` and `scripts/diagnose_cem_costs.py` (N3/N3b) — read for the first time; mechanism appears sound, one methodological note on the CI

`scripts/analyze_cost_ranking.py` (76 lines, fully read) does **not** itself
compute Spearman rho — it re-reads `d["pooled"][kind]["ci95_of_mean_seed_rho"]`
and `"mean_of_per_seed_rhos"` straight from `diagnose_cem_costs.py`'s output
JSON (`analyze_cost_ranking.py:61-62`) and only computes **regret** and
**top-10-vs-batch** locally:

```python
argmin_idx = int(costs.argmin())
regret = true_dist[argmin_idx] - true_dist.min()          # line 43-44
top10_idx = np.argsort(costs)[:10]
top10_means.append(true_dist[top10_idx].mean())            # line 47-48
```

This is the right construction for "regret": the true (physically-simulated)
distance-to-goal of the candidate CEM's own cost function would pick, minus
the best true distance any candidate in that same batch actually achieved.
Both `costs` and `true_dist` are read directly from the upstream JSON's
per-candidate arrays (`r["costs"]`, `r["true_dist"]`), same length, same
candidate index — no join/alignment step that could silently mismatch them.

**The rho computation itself lives in `scripts/diagnose_cem_costs.py`**
(traced, not fully read — read lines 1-130 and grepped the rest):

- `rollout_true_outcomes()` (`diagnose_cem_costs.py:98-124`) resets
  `base_env` to the **identical** `init_state` before **every** candidate
  (`prepare_with_visual(base_env, regime, seed, init_state)` inside the
  per-candidate loop, line 115) then executes that one candidate's raw
  actions for real and records the resulting `block_pos_diff`. Each
  candidate's true outcome is therefore independent of every other
  candidate's — no shared mutable env state, no leakage between candidates.
- All kinds (baseline + charts) score the **same** iteration-0 candidate
  batch by construction — CEM's first draw is model-independent given a fixed
  `local_seed=0` (`instrument_cost_function`'s docstring,
  `diagnose_cem_costs.py:62-78`, and `capture_iteration="first"` is the
  default) — so the cross-kind comparison is on identical inputs, not a
  cherry-picked or per-kind-resampled set.
- `spearmanr(costs.numpy(), true_dist)` (`diagnose_cem_costs.py:223`) is the
  correct correlation for "does a lower model cost predict a lower true
  distance," since it tests monotonic rank agreement, not a specific linear
  relationship — appropriate given `block_pos_diff` and CEM cost are not
  expected to be linearly related.
- The per-seed CI (`diagnose_cem_costs.py:265-284`) is a **normal-approximation
  CI on the mean of per-seed rhos** (`mean ± 1.96 * sd/sqrt(n_seeds)`), not a
  bootstrap and not a Fisher z-transform of rho (the textbook-correct way to
  CI a correlation coefficient, since rho is bounded in [-1,1] and its
  sampling distribution is not normal near the boundary). With `n_seeds=10`
  (the filenames cited in the module docstring, `seeds0-1-2-3-4-5-6-7-8-9`)
  this is a small-sample normal approximation on a bounded statistic — likely
  fine for rho values well inside (-1,1) like the ones reported (-0.072,
  +0.501), but not the most rigorous choice, and could understate CI width
  asymmetry near either boundary. This is the one methodological caveat
  found — **not** a coding bug, and not severe given the reported point
  estimates are far from ±1.
- No seed leakage found: each seed's `rollout_true_outcomes()` call uses that
  seed's own `(init_state, goal_state)` pair (via
  `sample_dataset_init_goal`, imported from `run_e0_planning.py`, the same
  sampler already verified pairing-clean in Part 1 §1.3 of this file), and
  costs/true_dist are aggregated per-seed before any cross-seed statistic is
  computed (`per_seed.append(...)` at `diagnose_cem_costs.py:255`, consumed
  by the pooled block at lines 260+).

**VERDICT: the N3/N3b mechanism computation is methodologically sound.** The
regret and rho definitions do what the claim describes; the one caveat is a
non-critical choice of CI method (normal approximation instead of a
Fisher-z or bootstrap CI on a bounded correlation coefficient), not a
correctness bug, seed leakage, or a wrong correlation type.

- Severity: **LOW** (CI methodology, not a correctness bug)
- Threatened claim: **N3, N3b** — point estimates stand; CI width should be
  treated as approximate, not exact, if directly quoted to 3 significant
  figures in the paper.
- Results on disk affected: whichever `atlas_out/cost_ranking_*` directories
  back N3/N3b — not independently re-run this pass (would require GPU).
- **Not done this pass, still open:** the file itself was not executed
  against a hand-computed toy case (out of scope for a read-only pass, as
  with `atlas/stats.py` in Part 3); and `diagnose_cem_costs.py` was read via
  targeted grep + the first 130 lines, not end-to-end — the `main()` body
  past line 260 (pooling/printing) was skimmed via grep context, not read in
  full line-by-line.

---

# What I did not get to

*Updated after pass 2 (2026-08-27). Pass 2 resumed exactly the priorities pass 1
flagged as unfinished — see PART 9 above for the full writeups. Status lines
below are updated in place; text describing what pass 1 had done is otherwise
left as pass 1 wrote it.*

**Priority 8 (train/deploy mismatch, S-5) — NOW DONE, see §9.1.** All four
S-5 sub-claims (frozen-predictor collection / 9x weaker CEM budget /
nas=1-vs-6 / 20 trajectories) are **CONFIRMED TRUE** against `run_e0.py`'s
own code and argparse defaults. `E0_RECOVERY_PLAN.md`'s framing of
`closed_loop` as a clean rejection should not be cited without this caveat.
One thing §9.1 itself did NOT verify: whether the actual historical
`closed_loop` run(s) behind the numbers in `E0_RESULTS.md` used these
defaults or explicit overrides — no launch command/log was located this pass.

**Priority 9 (regime persistence, E4 safety) — NOW DONE, see §9.2. VERDICT:
SAFE, no bug.** `PushTEnv._setup()` (called unconditionally at the top of
every `reset()`) constructs a brand-new `pymunk.Space` and brand-new
`Body`/`Shape` objects for the agent and block on every call — the previous
episode's `shape.friction`/`shape.elasticity` mutations are discarded with
the old objects, not carried forward. `PhysicsRegime.reset()`'s ordering
(rebuild via `self.env.reset()` first, apply this episode's regime shift
second) is correct. **E4's alternating-regime stream is not at risk of
regime bleed through this mechanism.** Remaining unverified (out of this
check's scope, flagged in §9.2): whether `run_e4.py`/`atlas/streams.py`'s own
per-episode env-construction path actually calls `PhysicsRegime.reset()` (not
`.env.reset()` directly, which would skip `_apply_physics()` and silently run
every episode as R0) — that is a question about the *caller*, not about
`PushTEnv`/`PhysicsRegime` themselves.

**Priority 6 remainder (`scripts/run_e2.py`, N9 leakage) — NOW DONE, see
§9.3. VERDICT: CLEAN, no leakage.** The chunk passed as `next_encoder_output`
in `run_e2.py`'s per-episode loop (`trajs[ep+1]`) is a genuinely
independently-seeded trajectory not yet folded into `expander._deficit_chunks`
at verification time. `e2_R2_cellB_q1`'s "3 charts committed" is a valid N9
demonstration.

**Priority 2 remainder (`scripts/analyze_cost_ranking.py`, N3/N3b) — NOW
DONE, see §9.4.** `analyze_cost_ranking.py` (76 lines) itself only computes
regret/top-10 metrics from upstream per-candidate arrays; the actual rho
computation lives in `scripts/diagnose_cem_costs.py`, which was traced (first
130 lines read in full, remainder via targeted grep, not read end-to-end).
No seed leakage, correct correlation type (Spearman), sound regret
definition. One non-critical methodological note: the per-seed CI uses a
normal approximation on a bounded [-1,1] statistic rather than a Fisher-z or
bootstrap CI — likely fine given the reported values are far from ±1, but not
the most rigorous choice if the CI width is quoted precisely in the paper.

---

*Everything below this line is pass 1's original unresolved-items list,
unmodified except for the four items just closed out above.*

This pass was cut short by a budget limit. The following priority items from
the audit brief are **unchecked or only partially checked**. A future session
should resume here.

**Priority 1 — E4 code (partially done).**
Done: `scripts/run_e4.py` (full read), `atlas/harness_e4.py` (full read),
`atlas/loop.py` (full read), `atlas/expand.py` (full read),
`atlas/adajepa.py` (full read), `atlas/library.py`, `atlas/chart.py`,
`atlas/router.py`, `atlas/regimes.py`, `atlas/score.py`, `atlas/streams.py`,
`atlas/stats.py`.
**Not done:** `scripts/smoke_e4.py` (221 lines — read only its `reset()` call
sites; its assertions were not audited, and since it is the only thing that
"validates" E4 its own correctness matters a great deal);
`modal/modal_e4.py` (179 lines — completely unread; a Modal launcher can
introduce its own arm-ordering, sharding and resume hazards, and §2.4 and
§2.10 are both order/shard-sensitive); `atlas/harness.py` (22.7 KB —
`run_e1_episode` is the pattern `harness_e4` says it copies "verbatim"; the copy
was not diffed against the original); `scripts/run_e5.py:45` — confirmed to
exist but **not opened**; the brief said to note the `NotImplementedError` stub
and move on.

**Priority 2 — Statistics (mostly done).**
Done: `atlas/stats.py` all four functions, read and reasoned through;
`scripts/analyze_n100.py` full read; `scripts/make_tables.py` `make_t1`/`make_t2`
read.
**Not done:** `scripts/analyze_cost_ranking.py` — **completely unread**. This is
the analysis behind **N3** and **N3b** (the "mechanism" result: per-seed mean
rho −0.072 under R2 vs +0.501 under R0, regret 8.5px vs 88.1px, "certain and
wrong" at convergence). Those are among the paper's most load-bearing numbers
and have had **no code audit at all**. Also not done: numerically executing
`stats.py` against hand-computed cases (the brief encouraged this; the functions
were verified by reading, not by running); `make_tables.py::make_t5`;
`scripts/merge_planning_shards.py` (shard merging is a classic place to
duplicate or drop episodes and thereby break pairing —
see §2.10, and note `e0_planning_sweep_60/100` both contain a merged
`ln_act_R2.jsonl` **plus** `_shard0`/`_shard1` files).

**Priority 3 — Seeding.** Partially done (§7). **Not done:** an exhaustive sweep
of every `seed` / `rng` / `default_rng` / `manual_seed` / `random.` call site
across `atlas/*.py`, `scripts/*.py` and `modal/*.py`. I checked the E0-planning
and E4 paths and the `manual_seed` grep, but not `run_e0.py`'s collector seeding,
`run_e1.py`, or `run_e2.py`.

**Priority 4 — World-model rollout.** Mostly done (§5). **Not done:** verifying
`world_model.encode()`'s `[B,T,V,H,W,D] → [T+1,N,D]` squeeze/flatten chain at
`harness_e4.py:289` (`enc["visual"].squeeze(0).squeeze(1).flatten(1,2)`) against
a real tensor — a silent broadcast there would be invisible; and checking
`proprio_enc[:, 0:1]`'s shape against `_make_z_ctxt`'s expected
`[1, 1, P_tok, D]`. Both need a live tensor and were out of scope for a
read-only pass.

**Priority 5 — Persistent arms.** Done for the reset question (§2.2), the buffer
semantics (`deque(maxlen=5)`, `adajepa.py:88`), and parameter carryover.
**Not done:** confirming that `pretrained_state`'s `state_dict()` keys actually
intersect `param_names` (which come from `named_parameters()`) for `kind="ln_act"`
— if the key namespaces ever diverge, `load_state_dict(..., strict=False)` would
**silently restore nothing** and even a correctly-called `reset()` would be a
no-op. This is a 2-minute check with `scripts/dump_params.py` and should be done.

**Priority 6 — Data leakage.** Partially done (§2.7) for `expand.py`'s internal
disjointness. **Not done:** `scripts/run_e2.py` (500 lines, completely unread) —
specifically whether the chunk passed as `next_encoder_output` in the
`e2_R2_cellB_q1` / `e2_R2_cellC_q1` diagnostics is genuinely disjoint from the
deficit chunks, or merely later in a list that was already consumed. This
directly determines whether **N9** is a valid demonstration of C2.

**Priority 7 — Closed-loop / replan path.** **DONE** (Part 1). This was the
user's top concern and the verdict is clean on the specific mechanism doubted.

**Priority 8 — Train/deploy mismatch in chart training (`scripts/run_e0.py`).**
**Barely started — the most important remaining gap after
`analyze_cost_ranking.py`.** I reached `run_e0.py`'s `closed_loop` collector
(lines 155-300) and confirmed from its own inline comments that it (a) imports
`run_e0_planning`'s sampler and env-reset path, and (b) requires
`num_act_stepped=1` at collection. I did **not** get to verify or refute
`OPUS_REMAINING_TASKS.md` item 10's specific four-mismatch allegation
(**claim S-5**): collected with the frozen predictor (on-policy for c0, off-policy
for the chart being trained); CEM 100×10 at collection vs 300×30 at eval;
nas=1 at collection vs nas=6 at eval; only 20 trajectories. The
`collect_num_samples` / `collect_iterations` argument defaults and the
`--num-trajs` default all need reading from `run_e0.py`'s `main()`. **This is
the single highest-value unfinished item**, because `E0_RECOVERY_PLAN.md`
currently frames the `closed_loop` result as a clean rejection and S-5 alleges
that framing is wrong and still standing in the source docs.

**Priority 9 — Regime application.** Partially done. Confirmed: `_apply_physics`
sets **both** the agent's and the block's shapes (`regimes.py:126-135`), which
is the pymunk trap the module docstring describes — ✅ correct; `_apply_physics`
is called from `reset()` (`regimes.py:99-102`) so it survives the env's space
rebuild — ✅ correct by construction; `PhysicsRegime` defines no `step()`, so
`base_env.step()` is harmless for physics but fatal for `VisualCorruption`
(§1.6). **Not done:** confirming that `PushTEnv.reset()` genuinely rebuilds the
space and recreates the shapes on every reset (I relied on the code comment at
`regimes.py:116-119` asserting `_setup()` runs on every reset — this is a
**DOC CLAIM I did not verify against `pusht_env.py`**). If `_setup()` does *not*
recreate shapes, then R1's `shape.friction=2.0` would **persist into a
subsequent R0 episode on the shared `base_env`**, silently contaminating E4's
alternating stream — a serious, silent, regime-mixing bug. **Check this before
launching E4.**

**Priority 10 — Smoke gates.** **DONE** (Part 4), for G1, G2, G3a, G3b, G4, G5,
G6. Not done: actually *running* `scripts/smoke_gates.py` (out of scope — it
loads the checkpoint and would consume GPU).

**Other files never opened this pass:** `scripts/run_e1.py`, `scripts/run_e2.py`,
`scripts/run_e5.py`, `scripts/smoke_e1.py`, `scripts/smoke_e4.py`,
`scripts/diagnose_cem_costs.py`, `scripts/diagnose_umf_locality.py`,
`scripts/analyze_cost_ranking.py`, `scripts/audit_e0_train_planning_overlap.py`,
`scripts/merge_planning_shards.py`, `scripts/make_figures.py`,
`scripts/make_e2_figure.py`, `atlas/plots.py`, `atlas/harness.py`,
all of `modal/`, and everything under `tests/`.
