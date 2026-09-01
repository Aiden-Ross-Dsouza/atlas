# DAY-1 ADDENDUM — replan cadence, metric validity, and the contact collapse

**Date:** 2026-08-31 · **Scope:** three analyses run on the `phase0_v3` settle2 artifacts that are
not recorded elsewhere. All numbers recomputed from raw JSONL with `.venv/Scripts/python.exe`;
every comparison is paired on the same 20 tasks (seeds 0–19) unless stated.
**Regime is R2 (`damping=0.5`) throughout except where the row says R0.** Read-only — nothing
launched, no code changed.

Source runs: `phase0_v3/c2_settle2_{baseline,ln_act}_{nas2,nas6}/`,
`phase0_v3/c2_settle2_R0_baseline_nas6/`. All use `--settle-steps 40` applied to **every**
episode (not only successes).

---

## 0. The three metrics, defined (they are routinely confused)

| metric | what it measures |
|---|---|
| **pass-through SR** | `block_success` is checked after **every raw step** and the episode **breaks immediately** on the first hit (`run_e0_planning.py:329-333`). The block may still be moving fast. "Did the block ever *transit* the goal zone?" |
| **settled SR** | After the episode ends, the agent holds position for 40 raw steps (zero relative action) and the block coasts to a stop; `block_success` is re-checked then. "Did the block come to **rest** at the goal?" |
| **settled distance** | The block's distance to goal after that same 40-step hold. The continuous version of settled SR — usable when settled SR is at a floor. |

Both SRs require `pos < 20 px` **and** `angle < 20°`. Under R2 `space.damping = 0.5` means the
block retains 50 % of its velocity per second, so it **glides**; under R0 (`damping = 0`) it stops
dead. That is why the two SRs diverge under R2 and coincide under R0.

---

## 1. Replan cadence: the pass-through metric INVERTS a control-theory result

`nas=6` with `max_steps=30, horizon=6` executes one full plan — effectively **open loop**.
`nas=2` executes 2 of 6 planned model-steps then replans, ×3 — **closed-loop MPC**.

| arm | cadence | pass-through SR | settled SR | settled distance |
|---|---|---:|---:|---:|
| frozen `c₀` | nas=2 (closed) | 9/20 | 0/20 | 137.3 px |
| frozen `c₀` | nas=6 (open) | **11/20** | 0/20 | 138.9 px |
| chart | nas=2 (closed) | 2/20 | 0/20 | **77.6 px** |
| chart | nas=6 (open) | **5/20** | 1/20 | 111.1 px |

Paired cadence contrast (same tasks):

| arm | pass-through | settled distance | closed better in | p |
|---|---|---|---:|---:|
| frozen | open loop looks better (11 vs 9) | 137.3 vs 138.9 — no difference | 9/20 | 0.96 |
| chart | open loop looks better (5 vs 2) | **77.6 vs 111.1 — closed better** | **16/20** | **0.0136** |

**Inference.** On pass-through, open loop beats closed loop for both arms — which contradicts
standard MPC theory. On settled distance the result inverts back: closed loop is equal (frozen)
or significantly better (chart). **The pass-through criterion reverses a control-theory result.**
This is the *second* independent conclusion that metric flips (the first being the adapter
comparison itself), which makes the criterion-validity finding a systematic property rather than
a one-off.

---

## 2. Why: residual momentum. The single most legible table in this study

Distance the block coasts after the agent stops (settle step 1 → 40):

| arm | cadence | contacts | block coasts |
|---|---|---:|---:|
| frozen | nas=2 | 4.00 | +74.2 px |
| frozen | nas=6 | 6.00 | +72.0 px |
| chart | nas=6 | 3.35 | +64.0 px |
| **chart** | **nas=2** | 2.75 | **+3.2 px** ← at rest |

**Inference.** The adapted chart under closed-loop replanning is the **only** configuration in
this study that brings the block to a stop. Every other arm leaves 60–75 px of residual momentum.
That is exactly what feedback control buys — observe, correct, decelerate — and exactly what one
committed open-loop push cannot. The pass-through metric pays for the ballistic push (it transits
the goal zone) and fines the controlled stop, which is why it ranks them backwards.

Note also: the **only genuine settled success anywhere in the R2 dataset belongs to the chart**
(nas=6, 1/20). The frozen model has 20 pass-through "successes" across both cadences and **zero**
settled ones.

---

## 3. The contact collapse is a REGIME effect, not a cadence or chart effect

| regime | arm | cadence | contacts/episode | pass-through | settled |
|---|---|---|---:|---:|---:|
| **R0** | frozen | nas=6 | **15.10** | 19/20 | **19/20** |
| R2 | frozen | nas=6 | 6.00 | 11/20 | 0/20 |
| R2 | frozen | nas=2 | 4.00 | 9/20 | 0/20 |
| R2 | chart | nas=6 | 3.35 | 5/20 | 1/20 |
| R2 | chart | nas=2 | 2.75 | 2/20 | 0/20 |

Decomposition of the drop: **R0 → R2 is −60 %** (15.10 → 6.00), cadence within R2 is −33 %
(6.00 → 4.00), chart within R2/nas=6 is −44 % (6.00 → 3.35).

The same collapse is present in the *training* data: on-policy contacts per trajectory are
**R0 17.07 vs R2 4.50 (−74 %)**.

**Inference, and it is a concern not a result.** The dominant cause of "the planner barely touches
the block" is the **R2 regime itself**, not closed-loop replanning and not the adapter. Under
`damping = 0.5` the frozen planner engages the block ~2.5× less than under R0, at both evaluation
and collection time. This is exactly the pre-registered §15-2 contact-collapse check: it was
measured at −74 %, its pre-registered fallback is `damping = 0.1`, and **the fallback was never
applied** (`P0_CLOSEOUT_AUDIT` B-2). Every R2 result in this project is therefore obtained in a
regime where the substrate's own planner is close to disengaged, and the R0 row shows what a
functioning cell looks like: 15.10 contacts, 19/20, and pass-through == settled.

**This should be treated as an open threat to R2's construct validity, disclosed in the paper's
limitations, and it strengthens the case for the R0 control arm.**

---

## Main-session assessment (2026-08-31, independent re-check)

Every number in §1–§3 re-verified from raw JSONL in a separate pass: contact table exact,
coast distances exact (chart nas=2 median **+0.0 px**, mean +3.2; every other arm +54–75 px
median), cadence contrast exact (chart settled dist nas2 77.6 vs nas6 111.1, 16/20, Wilcoxon
p=0.0136). The analysis holds.

Two points of nuance on the framing:

1. **The contact collapse under R2 is not obviously a *bug* — it may be the phenomenon.** The
   project's thesis is "a shift where the world model's predictions decouple from control." Under
   R2 the CEM cost-vs-outcome rank correlation is ρ≈0.001 (N3) — the planner's objective is
   uninformative, so it disengages. That is arguably *what you want to study*. The real question
   is narrower: is R2 so severe that there is nothing left to measure, or is it a clean instance
   of the decoupling? `damping=0.1` is the test.

2. **"Closed loop should have done well" (the user's intuition) fails for a specific reason.**
   MPC replanning only helps when the model's cost signal is informative. Under R2 it is ρ≈0, so
   replanning = re-sampling a broken ranking more often. The chart at nas=2 *is* the only arm that
   brings the block to rest — but it stops it 77.6 px from goal. Controlled, not competent. Closed
   loop cannot rescue a broken objective, and that is consistent with the paper's core claim, not
   a contradiction of it.

**Recommendation:** a 20-episode frozen baseline at `damping=0.1` (~$0.5) is now higher-value
than the 1.7 N=50 replication. Reframed decision rule: if the dissociation (UMF improves / control
does not) *survives* at `damping=0.1` with contacts restored → materially stronger paper (not an
extreme-regime artifact). If it *vanishes* → the paper's scope narrows honestly to severe shifts
and R2's contact collapse becomes an owned limitation rather than an unexamined confound.

## 4. damping sweep — `0.5` is over-severe, `0.1` is a graded shift (RAN 2026-08-31, §1.10)

Frozen `c₀`, it=10, N=300, `--settle-steps 40`, seeds 0–19. Every number recomputed from raw JSONL.

| regime | cadence | contacts/ep (mean/med) | pass-through SR | **settled SR** | settled dist (mean px) | median coast | block moved **toward** goal |
|---|---|---:|---:|---:|---:|---:|---:|
| **R0** (damping 0) | nas=2 | 15.8 / 14.5 | 13/20 | **13/20** | 23.0 | +0.0 | 19/20 |
| **R0** | nas=6 | 15.1 / 13.5 | 19/20 | **19/20** | 17.6 | +0.0 | 19/20 |
| **damping 0.1** | nas=2 | 5.5 / 5.0 | 11/20 | **2/20** | 41.2 | +0.8 | **16/20** |
| **damping 0.1** | nas=6 | 7.8 / 5.5 | 14/20 | **4/20** | 47.8 | +9.7 | **16/20** |
| **damping 0.5** | nas=2 | 4.0 / 3.0 | 9/20 | **0/20** | 137.3 | +54.8 | 6/20 (**moves away 14/20**) |
| **damping 0.5** | nas=6 | 6.0 / 5.0 | 11/20 | **0/20** | 138.9 | +62.9 | 3/20 (**moves away 16/20**) |

**Inference — this is a qualitative break, not a gradient.** At `damping=0.1` the frozen planner
is *degraded but functional*: it pushes the block **toward** the goal in 16/20 episodes at both
cadences, ends 41–48 px away, and the block **comes to rest** (coast ≈ 0). At `damping=0.5` the
planner is *anti-functional*: it moves the block **away** from the goal in 14–16/20 episodes and
leaves 55–63 px of residual momentum. Settled SR: R0 95% → 0.1 ≈ 20% → 0.5 **0%**.

**Consequence for the project.** Every R2 result on disk (C-1, C-2, the settle checks, the
"variance compression" / "less destructive" readings) is measured at `damping=0.5`, i.e. against a
frozen baseline that is not just weak but *directionally wrong*. "The adapter improves prediction
but the frozen planner controls better" was already retired by the settle check; "the adapter is
less destructive than the frozen planner" survives, but the honest framing is now **"at a shift
severe enough that the substrate planner pushes the block backwards, a prediction-fitted adapter
is merely inert."** Whether the prediction/control dissociation holds against a *functional*
baseline (`damping=0.1`) is **untested** — the chart has never been trained or evaluated there.

**Decision rule outcome (§1.10):** case **(c) intermediate** — contacts 5.5–7.8 (not ≥10),
settled SR 2–4/20 (not 0). No regime change without explicit human sign-off (§15). The scientific
choice — move the R2 story to `damping=0.1` (re-collect + re-run, ~$15–20) vs. keep `0.5` with the
collapse disclosed vs. run only the chart arm at `0.1` as a control — is escalated to the user.

Artifacts: `phase0_v3/c2_settle2_dmp01_baseline_nas{2,6}/`.

---

## Verification status (`CLAUDE.md` §1.9)

Recomputed this session from raw JSONL: every number above. Pairing asserted (identical
`init_block_pos_diff` / `init_agent_block_dist`) across all four R2 runs. Coast distances derived
from `settled_trace` steps 1 → 40 within each run, so no cross-run pairing is involved.
Training-data contacts read from `trajs_{R0,R2}.pt` `n_contacts` fields.
**Not established:** why the chart's plans are contact-poor at nas=2 specifically (candidate
actions are not persisted); whether `damping = 0.1` would restore engagement (never run).
