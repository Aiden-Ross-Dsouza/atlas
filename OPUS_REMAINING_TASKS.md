# Remaining tasks from Opus's review — not yet done

*Written 2026-08-26, revised after a full line-by-line re-read (the first pass
missed several items that weren't in Opus's numbered to-do lists — they were
in the narrative body of the messages). Cross-referenced against everything
actually completed this session (N=100 baseline/`ln_act`, the 20/60/100-trajectory
training-size sweep + paired N=40 re-evals, the 10-seed R2 cost-ranking
diagnostic, the nas=2 N=20 closed-loop arm, E2's sequential-hysteresis fix +
3-chart confusion matrix, and `ATLAS_SUMMARY.md`). Only items still
outstanding are listed — cross-check `E0_RESULTS.md`/`E2_RESULTS.md`'s current
top sections if in doubt about something not listed here.*

> **STATUS UPDATE (2026-08-27):** All 8 items in Section A are now DONE, and
> 3 of Section C's 5 items (C.23, C.24, C.25 — the three cheapest, run in
> cost-ascending order per explicit instruction) are DONE. C.26 and C.27
> remain open (C.26 is blocked on a sample-size decision; see its entry
> below). Section B (write-up-only) and Section D (deprioritized) were
> explicitly out of scope this session and are untouched. Every completed
> item is marked ✅ inline below, with a pointer to where the real result
> lives — nothing here is deleted, per this project's own no-silent-editing
> convention. Full detail for every ✅ item: `E0_RESULTS.md` and
> `E2_RESULTS.md`'s 2026-08-27 top sections, and `ATLAS_SUMMARY.md`
> (every result added there too, not just the source docs).

---

## A — Zero GPU, pure analysis on data already on disk

1. **✅ DONE.** **Knock-aways + mean progress at N=100, both arms.** The pre-registered
   mechanism test (§0.5: *"a working damping chart should specifically reduce
   knock-away failures"*) was confirmed at N=20 (5/20→2/20) but never
   recomputed at N=100. Two outcomes, both worth having: knock-aways still
   drop while SR stays flat (a second, mechanism-specific dissociation
   instance), or they don't (the N=20 confirmation was noise too).
2. **✅ DONE.** **SR by init-displacement bucket at N=100** (0–80px / 80–120px /
   120–300px) — does the chart help in *any* stratum, even if the aggregate
   is flat?
3. **✅ DONE.** **Partial/stratified Kendall τ**, controlling for `init_block_pos_diff`
   (and ideally `total_contacts`) — task difficulty plausibly drives both UMF
   and success jointly, so the τ=−0.406/−0.449 result may be measuring
   difficulty rather than model quality. If τ survives conditioning, the
   claim is strong; if it collapses toward zero, rescope to "UMF tracks
   episode difficulty," not "UMF tracks model quality." Also eyeball the
   scatter for a handful of catastrophic episodes carrying the whole
   correlation.
4. **✅ DONE.** **Degeneracy check on the existing R2 cost-ranking output**: per seed,
   what fraction of the 300 candidates actually made contact with the block?
   Random iteration-0 draws mostly do nothing — if most candidates never
   touch the block, ρ≈0 is guaranteed by a near-constant outcome, not by
   anything about the cost function. Report per seed: contact fraction,
   min/median/max `true_dist`, and ρ restricted to the contact-making subset.
5. **✅ DONE (including the R0 extension — item 24 below).** **Regret and top-tail metrics for the existing cost-ranking data**, not
   just Spearman ρ (which throws away exactly what CEM consumes): per seed,
   `true_dist` of the cost-argmin candidate, `min(true_dist)` over the batch,
   **regret = argmin-by-cost − batch min**, mean `true_dist` of the
   cost-ranked top-10 vs. the batch mean. Apply the same treatment to the R0
   run (item C.14) once it exists, not just R2.
6. **✅ DONE.** **Report the cost-ranking per-seed mean as a proper CI, not a bare
   number.** With per-seed n=300 (SE≈0.058) and 10 seeds (SD≈0.25, SE≈0.08),
   the correct statement is "ρ = −0.07, 95% CI [−0.23, +0.09] — no reliable
   ranking signal," not "slightly inverted" — the inversion isn't
   statistically established as currently written.
7. **✅ DONE.** **The "bridge" experiment.** If the cost-ranking diagnostic's 10 seeds
   used the same `sample_dataset_init_goal` construction as the N=100
   planning run's episodes 0–9, compare `init_block_pos_diff` between the two
   files for seeds/episodes 0–9 — if they match, ask directly: do the seeds
   where cost-ranking is inverted correspond to planning episodes that
   failed? n=10 is thin but informative; extend if the seeding does line up.
8. **✅ DONE.** **Verify the training-size sweep's pairing and confirm the N=100 result
   actually landed correctly** — a pairing break would invalidate the whole
   sweep. (Likely already true given the 0.000000px mismatch checks done
   this session, but Opus lists it as a "verify, don't assume" step.)

---

## B — Write-up-only fixes (no computation, edit existing docs)

9. **Retract the "strict superset" argument in `E0_RESULTS.md`'s P3
   section.** It argues the N=20 effect is real partly because it was "a
   strict superset of baseline... noise flips episodes in both directions;
   this did not." At N=100 the discordant pairs are b=8, c=9 — episodes flip
   in both directions, exactly what that argument said wouldn't happen if
   the effect were real. Mark this explicitly where the original argument
   lives, not only in the newer N=100 section.
10. **`closed_loop` was not correctly evaluated — say so explicitly, as its
    own retraction, not folded into a general "hypotheses rejected"
    statement.** Four train/deploy mismatches were stacked: collected with
    the frozen predictor (on-policy for `c0`, off-policy for the chart being
    trained), CEM 100×10 at collection vs. 300×30 at eval, `nas=1` at
    collection vs. `nas=6` at eval (flagged by Opus as the biggest one and
    not previously written down anywhere), and only 20 trajectories.
    **"Hypothesis 2 was not rejected. It was tested with a broken
    instrument."** The current framing in `E0_RESULTS.md` reads as a clean
    rejection; it should read "inconclusive — confounded by collection
    budget and horizon mismatch."
11. **Retract or narrow the `hybrid`-collector conclusion**: `E0_RECOVERY_PLAN.md`
    §0.6 states "the open-loop-vs-closed-loop question (P2b) is answered:
    dataset replay wins." `hybrid` changed two variables at once relative to
    `dataset` (open-loop→closed-loop *and* real recorded actions→scripted
    aimed-walk actions), so its worse performance is at least as
    parsimoniously explained by the degraded action distribution as by the
    open/closed-loop axis. That sentence overstates what was actually shown.
12. **State explicitly that `full` and `lora4`'s capacity results are
    confounded with training-set size, separately from `closed_loop`'s
    confound.** The training-size sweep this session only retrained
    `ln_act`. `full` and `lora4` were never retrained at 60/100 trajectories,
    so "capacity hurts" is really "capacity-at-20-trajectories hurts" for
    those two kinds specifically — this needs its own sentence, distinct
    from the general capacity-confound note already in `E0_RESULTS.md`.
13. **E2: state explicitly that "correct" is defined by regime label, not by
    demonstrated planning competence.** `correct_idx = 1` whenever `regime ≠
    R0`, but E0 measured that `chart_R2` isn't reliably better at the task
    than `c0`. So E2 measures whether UMF recovers the *regime label*, which
    is in tension with the project's own framing ("Measure Fitness, Don't
    Infer the Regime") unless this is stated plainly. There's a real defence
    (UMF recovers the label without ever being told it exists, purely by
    measuring fitness) — but the ground truth definition needs to be named,
    not implied.
14. **E2: note that `K_max` library-cap pressure is completely untested.**
    The 3-chart confusion matrix (this session) answers "only 2 charts, chance
    0.5," but nothing in the project tests behavior as the library approaches
    or exceeds `K_max=10` (eviction behavior, whether routing degrades as the
    library grows). Not necessarily worth running before the deadline, but
    worth one disclosed-limitation sentence.
15. **Add the R1-vs-R2 interpretive note to `E2_RESULTS.md`**: the 3-chart
    confusion matrix shows UMF separates *shifted-vs-unshifted* dynamics
    cleanly, not necessarily *which specific shift type* — one sentence,
    pre-empts a reviewer question.
16. **Stop describing E2 and E0 as "orthogonal."** Opus pushes back on this
    framing directly: E2 shows UMF identifies the chart that predicts
    better; E0 shows that chart plans no better. Those are the two halves of
    one finding, not two unrelated ones — reframe wherever "orthogonal"
    currently appears in the docs (`ATLAS_SUMMARY.md` already does this
    correctly; the source docs still use "orthogonal").
17. **Audit `run_episode`'s use of `base_env.step(a)` instead of
    `regime.step(a)`.** Currently believed harmless for damping/friction
    (pymunk space/shape attributes persist past `reset()`), but it silently
    bypasses any wrapper-level per-step logic, and would silently bypass
    `VisualCorruption` entirely if a corrupted planning episode is ever run.
    Opus: *"verify before E4."* A one-hour code audit, not a re-run — unless
    it turns out to matter, in which case affected results need flagging.
18. **Disclose the train/val split is 2-way, not 3-way** —
    `--num-val-trajs` is used both for early-stopping decisions *and* as the
    reported "Eval UMF" number, meaning the same held-out set serves two
    roles. Minor, but should be a disclosed limitation, not silently assumed
    to be a clean 3-way split.
19. **Disclose single-environment/single-checkpoint/single-seed scope.**
    Every result in this project uses one environment (Push-T), one
    checkpoint (`dino_wm_pusht`), and charts trained from a single random
    seed each. Opus lists this explicitly among "what will get pressed on" —
    not fixable before the deadline, but should be a named limitation rather
    than an implicit one.
20. **Don't lean on "pre-registered" as a shield in the write-up.** The
    criteria evolved during the project (the 15pp bar isn't in the original
    proposal; R2 was accepted at 46.7% against a ≤40% pre-registered rule;
    the q=1 diagnostics deviate from the pre-registered q=3). Every
    deviation is individually documented and justified, which is good
    practice — but it reads better if the write-up owns the evolution in a
    footnote than if a reviewer has to piece it together themselves.
21. **Build the dissociation figure** — UMF vs. planning SR across the 5
    arms (`c0`, `ln_act`, `lora4`, `closed_loop`, `full`), with the inversion
    marked. Every number needed is already in `E0_RESULTS.md`; this is a
    plotting task, not a new measurement.
22. **Reframe the E0-failure narrative**: the pre-registered rule ("smallest
    kind reaching ≥90% of `full`'s gain") became inapplicable once `full`'s
    gain went negative — correctly reported as inapplicable, not
    reinterpreted. But the *replacement* 15pp bar that E0 was then judged
    against was invented during the recovery process and appears nowhere in
    the original proposal or implementation plan, sitting close to the
    minimum detectable effect of its own N=20 sample. The honest framing is
    "a chart-capacity screen, evaluated under a one-open-loop-plan protocol,
    at N=20, did not clear an ad-hoc 15pp bar" — not "the method doesn't
    work." Worth one clarifying paragraph wherever the E0 failure is first
    introduced.

---

## C — Cheap GPU runs, not yet launched

23. **✅ DONE.** **Re-run E2's original 2×2 cells (A/B/C/D) under the sequential-hysteresis
    fix.** The 3-chart confusion matrix already used the fix, but the
    *original* 2×2 numbers (Cell B's "0.880 vs. 0.324" headline) are still
    the pre-fix `current_idx=0` values — `E2_RESULTS.md`'s own Limitations
    section already flags this as unresolved. Opus, twice: *"your headline
    positive is currently un-citable alongside the 3-chart numbers."* Cheap
    (~$0.10, local, no GPU planner needed) and closes the credibility gap on
    the project's one clean positive result.
24. **✅ DONE.** **Cost-ranking diagnostic on R0** — same design as the existing R2 run
    (10 seeds, baseline + `ln_act`, iteration-0 candidates, ~$1). Called
    **non-negotiable** across two separate messages — see the explanation
    above. *Verbatim: "the single thing that would most damage the paper
    right now is publishing the cost-ranking result without R0."*
25. **✅ DONE.** **Cost-ranking at converged CEM candidates** (not iteration-0) — at
    least 3 seeds, full 300×30 search, capture the *final* iteration's
    population instead of the raw prior draw, roll those out for real
    (~$2). Answers the standing objection that iteration-0 is an untrained
    random draw, not what CEM actually executes from.
26. **⏳ BLOCKED (2026-08-27) — waiting on an N decision, not yet launched.**
    An N=100 launch was started and stopped before burning real GPU time,
    per an explicit interrupt; still open. **The R1 planning eval — baseline vs. `ln_act`.** Discussed and planned
    earlier this session (`lora4` was explicitly dropped from scope by
    user decision, see note below) but **never actually launched** — the
    session moved to nas=2/training-sweep/cost-ranking instead. The single
    largest gap against Opus's recommendations: the only way to test whether
    the null generalizes to a second physical mechanism (friction vs.
    damping), and whether the UMF-gap (2.4% at R1 vs. 14.1% at R2, from E2)
    predicts planning benefit the way the dissociation thesis implies it
    should. Opus's later recommendation is N=100 for symmetry with R2
    (~$7); an earlier message said N=40 minimum.
27. **nas=2 at higher power.** The completed run used N=20 per arm (+10.0pp,
    CI [−10,+30], not significant). Opus's later message reprioritizes this
    *above* the R1 eval for a reason distinct from the original ask: at
    nas=6 (one replan), UMF-vs-outcome is necessarily *retrospective*
    (scored after the trajectory that produced it is already fixed); nas≥2
    is the only setting where UMF can be scored *prospectively* — an early
    chunk predicting the rest of the episode — which is closer to what the
    method actually needs. A properly-powered re-run (same N=100 scale-up
    already done once for the main result) would resolve whether +10pp is
    real or noise.

---

## D — Explicitly deprioritized or context-only (not action items)

- **`lora4` on R1, N=40** — Opus's shorter to-do list includes this under
  "if time remains" (lowest priority tier), but this session and the user
  explicitly decided to drop `lora4` from the R1 track specifically because
  it was never given a proper power check on R2 either — bundling it into
  R1 first would have been scope creep relative to what needed answering.
  Listed here for completeness since Opus did recommend it, not because
  it's currently considered a live gap.
- **E5** — answers an objection E2 doesn't have (E2 never used a planner).
- **E3** — runs inside E4, moot if E4 doesn't run.
- **τ/q sensitivity sweep** — explicitly out of scope given the deadline.
- **E1 proper** — already closed analytically (oracle−random spread
  2.5–3.3pp, below the 10pp reporting threshold).
- **E4 full stream** — at most a hard-timeboxed smoke test (4h max, abandon
  on failure) was ever recommended, never the full stream. Opus's last
  message additionally notes E4's likely outcome is a predictable "null
  ladder" given the dissociation thesis, lowering its value further.

---

## Suggested order, if resuming

The zero-GPU items (A) and write-up fixes (B) cost nothing, several of them
directly gate whether the cost-ranking result is even safe to cite as-is
(A.4–A.6, B.10), and Opus consistently sequenced "today, zero GPU" work before
any new spend across all messages. Doing A and B first is the right order.
Among the new runs (C), item 24 (R0 control) is the single highest-priority
one — it's the one Opus called non-negotiable twice — and item 26 (R1
planning eval) is the largest scientific gap still open.
