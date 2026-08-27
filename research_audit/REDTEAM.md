# ATLAS — REDTEAM

**Last updated: 2026-08-27, pass 2 — N8/RQ1's fabricated data point has been
replaced with a real 20-episode evaluation, and the conclusion REVERSES: the
oracle-vs-random spread is now 13.3pp [CI 3.3,25.0], clearing the reporting
threshold the fabricated version missed. See the N8 section and Section D's
RQ1 entry for the update; the "strongest honest paper" recommendation at the
bottom is also updated. Everything else below is unchanged pass-1 content.**

**Pass 1 banner (preserved) — full adversarial pass against every L3+ claim
in `CLAIMS_MATRIX.md`, using `LITERATURE_AUDIT.md`, `PROPOSAL_CODE_ALIGNMENT.md`
(passes 1+2), `CODE_AUDIT.md` (passes 1+2) and `RESULTS_AUDIT.md` (passes 1+2)
as the evidence base. This pass does not re-derive numbers or re-read code; it
attacks the evidence those four files already established, per the workshop's
Idea-Track standard (negative results welcome; the bar is "is the claim
actually established," not "is the result positive"). Two findings flagged by
the coordinating session as especially load-bearing (N8's fabricated
`chart_R1` proxy row, and S-5's confirmation that `closed_loop` was tested
with a materially weaker/mismatched instrument, not cleanly rejected) are
folded into the relevant per-claim attacks below, not treated as a separate
section.**

See `research_audit/CLAIMS_MATRIX.md` for claim IDs and
`research_audit/EXPERIMENT_STATUS.md` for implemented/run status. Evidence
levels (L0-L7) per `.claude/skills/research-audit/SKILL.md`.

---

## How to read this file

Each entry: the claim, the attack(s), the outcome (**SURVIVES** /
**WEAKENED** / **DEFEATED**), and — for anything not SURVIVES — the minimum
fix and its rough cost against the ~$70 remaining GPU budget (~$0.80/GPU-h,
so ~87 GPU-h) and ~78 hours of wall clock. Claims below L3 (never run, e.g.
E3/E4/E5, RQ3, RQ4, L-1, C2 as an empirical matter, G-1) are not attacked
individually here in the same way — there is no result to attack, only an
absence, which is already the finding. They are addressed once, together, in
the closing framing section, because that is where they actually matter for
what to submit.

---

## Section A — The E0 planning results (N1, N2, N4, N5)

### N1 — "Well-powered null": chart doesn't improve planning success (44.0% vs 43.0%, N=100, CI [-9,+7], McNemar p=1.000)

**Attack 1 — is the protocol capable of showing the effect at all?**
`PROPOSAL_CODE_ALIGNMENT.md` item G.4 and `CODE_AUDIT.md` Part 8 item 4 both
independently conclude this is decisive: at `num_act_stepped=6`, every
episode in this dataset consists of **exactly one CEM search, executed
before the agent has observed a single consequence of its own actions**,
then run to completion blind. A better world model has exactly one channel
to express itself (a better single plan) and zero channels to correct an
error the plan made — which is precisely the capability a regime-adapted
model exists to provide. This is not a hypothetical: N3 (below) shows the
planner's own cost ranking is near-zero-correlated with true outcome under
R2, and under an open-loop protocol a bad initial ranking is the entire
episode outcome, with no chance to recover. This is the single most
parsimonious alternative explanation available for the null — not "the
adapter doesn't help," but "the protocol structurally cannot let it help."
**Outcome: WEAKENED, materially.** The number is real (`RESULTS_AUDIT.md`
independently recomputed it exactly: 44/100, 43/100, CI, McNemar), but its
interpretation as "the method doesn't work" is not supported — only "the
method doesn't work under a one-shot open-loop planning protocol" is
supported, and the paper's own code comments (`run_e4.py:74-79`) already say
this protocol "cannot exercise routing, refinement, or verification."
**Minimum fix:** report N1 explicitly and only as an open-loop finding, and
either (a) drop the "the chart does not help" framing entirely in favor of
"open-loop CEM planning is largely insensitive to adapter quality under this
protocol" or (b) run a properly powered closed-loop comparison — see N5
below for cost.

**Attack 2 — is the training-data-to-eval pathway actually testing the
adapter's competence, or a mismatched instrument?** `PROPOSAL_CODE_ALIGNMENT.md`
item I and `CODE_AUDIT.md` §9.1 (S-5, now CONFIRMED on all four sub-claims)
establish that the *default* chart behind N1 (`ln_act`×R2, `atlas_out/e0_v3_dataset`,
`--data-source dataset`) is fit on **replayed expert demonstrations recorded
under R0**, played back open-loop under R2 — i.e. the chart is trained to
predict what an expert-under-R0 would have done, not what a CEM planner
operating under R2 needs modeled. This is a second, independent, equally
parsimonious explanation for the null that has nothing to do with whether
adapters can absorb dynamics shifts: the chart may be fit to the wrong
target distribution entirely. **Outcome: WEAKENED further.** Two
structurally distinct explanations (protocol can't show the effect; training
data doesn't match the eval task) both compete with "the method doesn't
work" and neither has been ruled out. **Minimum fix:** the `closed_loop`
collector exists and was built to test exactly this, but (per S-5) was
itself confounded with a ~9x-weaker CEM search budget and opposite-extreme
replan frequency relative to eval — so re-running it is not free; see N5's
cost note.

**Attack 3 — is UMF's reported improvement itself trustworthy, or an
artifact of a 2-way split?** `PROPOSAL_CODE_ALIGNMENT.md` item H.2: the
8-trajectory "validation" set is reused for both early-stopping (checkpoint
selection, consulted up to 80 times) and the final reported `eval_umf`. This
does not touch N1's *planning*-success number directly, but it means the
capacity story that would justify believing the chart is competent (UMF
0.336→lower) is itself optimistically biased, undermining any argument that
"the chart clearly reduced predictive error, so the null in planning is
informative about a genuine dissociation rather than about a bad chart."
**Outcome for N1 overall: WEAKENED.** The number itself is a correctly
computed, well-powered null (verified independently at L5/L6 by
`RESULTS_AUDIT.md`), but its scope must shrink to "under one-shot open-loop
planning, with a chart trained on off-policy replayed data" — not a general
statement about the method.

### N2 — The dissociation: UMF predicts success within an arm, not across arms

**Attack 1 — non-random exclusion.** `RESULTS_AUDIT.md` §2(b) independently
found that the within-arm Kendall tau (τ≈-0.4, both arms) is computed on
n=92-94/100, not the full 100, because G6's motion gate nulls UMF for entire
episodes when the single open-loop replan window has low motion — and the
excluded episodes are **not random**: they are exactly the easy,
small-displacement, always-successful episodes. The correlation is genuine
(recomputed independently, matches exactly) but is measured over the harder
~93% of the sample, not the full one. This does not defeat the dissociation
claim, but a reviewer would ask why the easy episodes are gone and the
current write-up (per `CLAIMS_MATRIX.md`) does not disclose it.
**Outcome: WEAKENED (disclosure gap, not a wrong number).** **Minimum fix:**
one sentence in the write-up stating the n=92/94 and why; no compute cost,
~10 minutes.

**Attack 2 — is the within-arm correlation itself doing the causal work the
claim implies, or is it a restatement of episode difficulty?** The partial
correlation controlling for `init_block_pos_diff` and `total_contacts` still
shows τ≈-0.36 to -0.37 (both matched exactly by `RESULTS_AUDIT.md`), which
does address the most obvious confound (harder episodes have both worse UMF
and worse success) — this attack does not succeed as strongly as it might.
One residual concern: `RESULTS_AUDIT.md` §3.6 flags the partial-tau p-value
as anticonservative (OLS-residualized Kendall doesn't correct its null
distribution for the estimated regression coefficients) — but the point
estimate itself, which is what the claim leans on, is unaffected.
**Outcome: SURVIVES**, with the n=92/94 caveat from Attack 1 attached.

**Attack 3 — novelty.** `LITERATURE_AUDIT.md` §6 is unambiguous and
directly on point: the *general* phenomenon (a model-quality metric
dissociates from downstream control performance) is established since
Lambert et al. 2020 (L4DC, the foundational "objective mismatch" paper the
proposal itself cites), with a value-equivalence theoretical companion
(Grimm et al. 2020) and at least two 2026 JEPA-specific papers making
essentially the same point (RC-aux, already cited by the proposal; "The
Objective Is the Bottleneck," arXiv:2608.12959, not yet cited). **If N1/N2
is framed as "we discovered prediction accuracy doesn't imply planning
competence," this is DEFEATED by prior art** — a reviewer who knows this
literature (plausible at a Continual World Models workshop) will read it as
naive. **If framed narrowly** — first demonstration of this dissociation for
a *frozen visual foundation-model* JEPA with a *persistent adapter*, plus
the specific training-data-budget sweep (N4) which the literature audit
found no direct precedent for — **the narrower claim SURVIVES.** This is a
framing choice, not an evidence gap, and it is binary: get it right or the
paper is scoopable in the abstract's first sentence. **Minimum fix:** cite
Lambert et al. 2020, Grimm et al. 2020, and "The Objective Is the
Bottleneck" explicitly; state the dissociation as a replication in a new
substrate, not a discovery. Zero compute cost.

### N3 / N3b — CEM cost ranking degrades under R2 specifically (rho +0.501 R0 vs -0.072 R2), converged plans land farther from goal

**Attack 1 — is this the model's fault or the planner's?** This is the
strongest finding in the project and the attack fails. `RESULTS_AUDIT.md`
§2(d)/(e) and `CODE_AUDIT.md` §9.4 both independently verify the mechanism
computation is sound (correct Spearman use, no seed leakage, matched
init/goal pairs across kinds, regret defined correctly) and the numbers
reproduce to 3 significant figures from raw per-candidate arrays. Critically,
the same planner and same frozen model rank candidates *well* under R0
(+0.501) and *badly* under R2 (-0.072) — this is a controlled, within-planner,
within-model comparison, so "maybe the planner is just bad in general" is
ruled out by the R0 control. **Outcome: SURVIVES as a mechanism claim.**

**Attack 2 — novelty.** `LITERATURE_AUDIT.md` §7 (confirmed again in §9's
supplementary pass, via direct WebFetch of arXiv:2608.12959) finds a
structurally similar but **mechanistically distinct** finding in "The
Objective Is the Bottleneck": that paper's cost-ranking collapse is a
horizon/geometry-conditioned saturation of a *fixed* squared-latent-distance
cost, present even with **zero distribution shift**, on one model throughout.
ATLAS's N3 is **regime-shift-conditioned**: the identical planner/model pair
works under R0 and fails under R2, with R0 as an explicit no-shift control.
These are genuinely different failure modes even though both fall in the
same "planner ranking, not predictor accuracy, is the bottleneck" cluster
(which also includes RC-aux and Operator-on-F). **Outcome: SURVIVES, but
must cite the cluster explicitly** — a reviewer who knows this fast-moving
2026 sub-area and sees none of it cited will discount the paper's awareness
of its own territory. **Minimum fix:** cite arXiv:2608.12959, arXiv:2607.04464,
and the RC-aux paper (already cited) as the lineage this result extends;
state the regime-shift-vs-horizon distinction explicitly. Zero compute cost.

**Attack 3 — is this independent of N1's open-loop protocol, or is it the
same finding twice?** `PROPOSAL_CODE_ALIGNMENT.md` item G.4 point 3 makes
this connection explicit and it should be treated as a strength, not a
weakness, if disclosed: N3 is *causally upstream* of N1 under the current
protocol — under one-shot planning, a broken cost ranking under R2 *is* the
entire outcome, with no replanning to partially recover from it. This means
N1 and N3 are not two independent negative results corroborating each other;
they are one mechanism (planner ranking collapse) observed twice, once as
its direct cause (N3) and once as its consequence under a protocol that
cannot mitigate it (N1). **Outcome: SURVIVES as a mechanism finding, but
N1 and N3 must not be presented as independently corroborating evidence** —
that would double-count one phenomenon. **Minimum fix:** state the causal
relationship explicitly in the write-up. Zero cost.

### N4 — More training data monotonically improves UMF, buys nothing in planning success (20/60/100 traj sweep)

**Attack 1 — is the "monotone UMF improvement" itself trustworthy?**
`PROPOSAL_CODE_ALIGNMENT.md` item H.2, independently corroborated by
`RESULTS_AUDIT.md` §2(c): the 60/100-trajectory points are confirmed
genuinely held-out (disjoint seed manifests), but the **20-trajectory point
(`e0_v3_dataset`) has no seed manifest at all** — its held-out status cannot
be independently verified, only taken on the prose description in
`E0_RESULTS.md`. More importantly, **all three points share the identical
fixed 8-trajectory selection/eval set** while training data grows 5x, so the
model-selection procedure itself (checkpoint chosen by validation loss on
that fixed 8-trajectory set) is consulted with an ever-larger training pool
behind it — some fraction of the "monotone improvement" is expected simply
from more data producing a checkpoint that happens to fit the fixed
selection set better, independent of whether it generalizes any further.
**Outcome: WEAKENED, not defeated.** The direction (more data helps UMF,
doesn't help planning) is very likely real and is independently
recomputed/matched exactly by `RESULTS_AUDIT.md` at L5/L6 for the planning
side (all CIs span zero at 60 and 100 trajectories). But the *monotonicity*
and *magnitude* of the UMF trend specifically should not be over-claimed as
a clean effect-size result, and the missing 20-traj manifest is a real
provenance gap. **Minimum fix:** state the shared-selection-set caveat
explicitly; regenerate the 20-traj point with a saved manifest if time
allows (~1-2 GPU-h, cheap, but not essential — the finding survives even
without this fix, just with an honest caveat).

**Attack 2 — is "buys nothing in planning success" independently informative,
or the same open-loop-protocol artifact as N1?** Yes to the second half —
this is the identical Attack 1 made against N1, and it applies here with
equal force, since N4's planning-success arm uses the same nas=6 protocol.
**Outcome: WEAKENED for the same reason as N1** — this is "the cleanest form
of the dissociation" *under one-shot open-loop planning*, not in general.
`RESULTS_AUDIT.md`'s own L6 assessment calls this "the cleanest of the
claims checked" for the recomputation-accuracy question, which is a
different (and satisfied) bar than the causal-interpretation question raised
here.

### N5 — Closed-loop (nas=2) flips positive: 40.0% vs 50.0%, +10.0pp, CI [-10,+30], McNemar p=0.625, N=20

**Attack 1 — is this evidence of anything, on its own terms?**
`RESULTS_AUDIT.md` §2(f) states this most bluntly and correctly: with 20
paired episodes, only 4 are discordant (the other 16 agree regardless of
arm and contribute zero information to McNemar). A 3-vs-1 split among 4 coin
flips is indistinguishable from chance (p=0.625 confirms it). **Outcome:
DEFEATED as evidence of a positive effect** — the paper's own framing
("explicitly labelled not significant") is already honest about this, and
that honesty should be preserved, not upgraded, in any write-up. This
finding cannot support a directional claim at any reasonable confidence and
must not be presented as "N5 points the other way" without immediately
attaching this caveat.

**Attack 2 — is the nas=6-vs-nas=2 comparison itself clean?**
`CODE_AUDIT.md` Part 1 §1.4-1.5 finds the closed-loop path itself is CLEAN
(no stale-context bug, real re-encoding at every replan — this specific
user-flagged concern is resolved favorably) but flags a real confound: because
`plan_length` stays pinned at `horizon=6` regardless of `steps_left`, nas=2's
three replans each run a full 6-model-step CEM search, so the nas=2 episode
consumes **3x the total CEM search compute** of the nas=6 episode over the
same 30 raw steps. The paired chart-vs-baseline comparison *within* nas=2 is
fair (both arms get equal budget), but "closed-loop feedback helped" is
confounded with "3x more search compute helped." **Outcome: the within-nas=2
comparison SURVIVES as a clean (if underpowered) paired test; any narrative
attributing the +10pp specifically to feedback rather than compute is
WEAKENED.**

**Minimum fix for N5 as a class:** a properly powered closed-loop run at
matched compute (either give nas=6 3x the CEM budget for a fair
compute-matched comparison, or accept the compute confound and just power
up nas=2). At N=100 pairs (matching N1's power), with ~3x nas=2's compute
per episode, this is roughly 100 episodes x 3 replans x 300x30 CEM search x
2 arms — likely 15-30 GPU-h depending on the checkpoint's actual per-search
wall time (unmeasured in the audit; `scripts/profile_episode.py` exists for
exactly this but its own output was not located in any audit file). This is
plausibly affordable within the remaining ~$70/87 GPU-h budget but would
consume a large fraction of it and of the 78 remaining hours, and is the
single highest-value experiment left if the team wants a positive result
rather than a well-scoped negative one.

---

## Section B — The one positive result (N6, N7) and the mechanism claims (N8, N9)

### N6 — 3-chart UMF routing: 60.3% vs chance 33% vs S-dyn 36.5%

**Attack 1 — is this measuring routing, or measuring planning competence?**
`OPUS_REMAINING_TASKS.md` item 13 (recorded in `CLAIMS_MATRIX.md` S-7,
independently unaddressed by any audit file that found a rebuttal) makes a
sharp point: E2 defines routing "correctness" by **regime label** — the
correct chart is whichever was trained for the true regime, not whichever
chart a planner would actually do better with. This is in direct tension
with the paper's own stated framing ("Measure Fitness, Don't Infer the
Regime"): if correctness is defined by inferring the regime, the experiment
is silently measuring exactly the thing the method claims to avoid needing.
Given N1/N2's dissociation (UMF-improvement does not imply planning
improvement), a chart that is "correctly" selected by E2's own criterion is
not thereby shown to plan any better. **Outcome: WEAKENED — the framing
claim ("fitness, not regime-inference") is in real tension with the metric
actually used, and no planner is in the loop for E2 at all (disclosed and
defensible as a cost-saving deviation per `EXPERIMENT_STATUS.md`, but it
means N6 is a claim about selector-accuracy-vs-regime-label, not about
planning outcomes).** **Minimum fix:** either rename what N6 measures
explicitly ("regime-label routing accuracy," not "fitness-based competence")
or run a small planner-in-the-loop check on a subset of E2's episodes to
confirm regime-correct routing correlates with better planning outcomes
(would require reusing the CEM planner harness against a handful of E2's
already-collected trajectories/seeds — a few GPU-h, not the full E2 rerun).

**Attack 2 — could a simpler explanation (S-dyn is just a bad baseline)
explain the margin, rather than UMF being good?** The confusion matrix
(independently recomputed exactly by `RESULTS_AUDIT.md` §2(g)) shows S-dyn's
column 0 dominates every row (61,62,65 out of ~90-104) — i.e. S-dyn nearly
always picks the identity chart regardless of true regime. This is a real,
qualitatively distinct failure mode (S-dyn isn't discriminating at all, not
merely discriminating worse), which strengthens rather than weakens N6 —
it rules out "UMF just barely edges out a competent baseline," and instead
shows UMF is doing real discriminative work that a fingerprint-style
baseline structurally cannot. **Outcome: SURVIVES.**

**Attack 3 — hysteresis confound.** `CODE_AUDIT.md` §6.1 (independently
corroborated in `RESULTS_AUDIT.md` §10) proves algebraically that the
spread-normalized hysteresis margin is **exactly inert at K=2** and only
"non-trivial but still very permissive" at K=3. N6 uses a 3-chart library,
so this attack is weaker here than for N7 (below) but not fully neutralized
— `RESULTS_AUDIT.md` explicitly did not re-simulate how often the K=3
condition actually binds in the 720-record confusion-matrix data.
**Outcome: SURVIVES with an open, unresolved caveat** — flag as unverified
rather than clean.

### N7 — Post-hysteresis-fix Cell B: UMF 0.833 vs S-dyn 0.570 (+26.3pp), "down from pre-fix +55.6pp, margin roughly halved"

**Attack — the causal attribution is DEFEATED, the number itself survives.**
This is the sharpest, cleanest kill in the whole audit. `CODE_AUDIT.md` §6.1
proves algebraically (not empirically — a clean derivation from
`router.py:94-101`) that for any exactly-2-chart library, the
spread-normalized hysteresis margin is **mathematically forced to a no-op**:
the incumbent, if not already the argmin, is by construction the *maximum*
of a 2-element set, so `relative_gap = (max-min)/(max-min) = 1.0`, which
always exceeds `m=0.05`. Cell B is exactly a `{c0, chart_R}` library, i.e.
K=2. `RESULTS_AUDIT.md` §10 traces this forward: the "pre-fix" 0.880/0.324
raw data no longer exists on disk (overwritten by a directory-name reuse),
but an intermediate run (`e2_R2`, before the final "posthysteresis" label)
shows 0.828/0.575 — already close to the final 0.833/0.570 — and
`E2_RESULTS.md`'s own text discloses **two separate fixes happened**, not
one: a "sequential hysteresis" fix (an architectural change to how
`current_idx` persists across chunks) and the margin-formula change
CODE_AUDIT.md's proof concerns. Since the margin-formula change is proven
inert at K=2, **whatever moved the number from ~0.88/0.32 to ~0.83/0.57 was
not the margin fix** — it must be the sequential-`current_idx` change, or
something else undocumented. **Outcome: the underlying numbers SURVIVE**
(independently recomputed exactly at L5), **but the causal narrative
"post-hysteresis-fix... margin roughly halved" is DEFEATED** — a reviewer
who checked the router code (as `CODE_AUDIT.md` did) would catch this
immediately and it would cost credibility on every other claimed fix in the
paper. **Minimum fix:** rewrite the attribution to credit the actual
mechanism (the sequential `current_idx` fix), or drop the causal claim and
report only the before/after numbers without attributing them to a named
mechanism. Zero compute cost — this is a two-sentence correction.

### N8 — Oracle-minus-random spread over the real library closes E1 analytically, "no routing algorithm can manufacture benefit the library does not contain"

**>>> UPDATE 2026-08-27, pass 2: the fabrication this section attacks has been FIXED with real data, and the underlying conclusion REVERSES. <<<** The coordinating session ran the missing `chart_R1`×R2 evaluation for real (`atlas_out/e0_chart_r1_on_r2/ln_act_R2.jsonl`, 20 episodes). Real result: oracle 60.0%, random 46.7%, **spread 13.3pp, CI [3.3, 25.0]** — clears the 10pp bar, CI excludes zero. This is the opposite of the 2.5-3.3pp/CI-touches-zero result the attacks below were written against. **Attack 1 (fabrication) is now moot — the fabricated data was replaced, not merely flagged.** Attack 2's finding (thin sample) is weakened but not eliminated — n=20 is still small, though no longer driven by a single discordant episode (real discordance now spans multiple episodes/arms, per `RESULTS_AUDIT.md` §11). **The bottom-line implication changes: "the decision not to run a full E1 SURVIVES as reasonable resource allocation" (this section's original conclusion, and Section D's RQ1 entry below) no longer holds** — a real denominator now appears to exist, so the argument that justified skipping E1 is no longer sound. Whether to actually run a real E1 evaluation is a fresh decision for the team; this section's original attacks (preserved below for the record) were sound analysis of the data as it stood at the time, and the fabrication-hunting methodology that caught this is exactly why it's now fixed.

*Original pass-1 attacks, preserved for the record — written against the now-superseded fabricated data:*

**Attack 1 — fabricated data point, exactly as flagged by the coordinating
session.** `RESULTS_AUDIT.md` §7(j-2) tested the hypothesis that the
3-chart row's `chart_R1` entry was produced by duplicating the baseline
(`c0`) success array, and it reproduces the claimed oracle (50.0%), random
(46.7%), spread (+3.3pp) and CI (`[0.0,+10.0]`) **exactly, 4-for-4**. No
directory anywhere in `atlas_out/` contains a real R1-trained chart evaluated
on R2 planning episodes. `HANDOFF.md` §7.1 presents this as a genuine
3-chart library result **without disclosing** that one of the three "charts"
is the baseline counted twice — a defensible placeholder assumption
(untested chart against a regime it wasn't fit for probably behaves close to
frozen) but a choice, not a measurement, and undisclosed as such.
**Outcome: DEFEATED as currently written.** A reader of `HANDOFF.md` or
`CLAIMS_MATRIX.md` N8 would reasonably believe `chart_R1` contributed real,
independent data. It did not. This must be corrected before any paper draft
cites the 3-chart row — either disclose the proxy explicitly and relabel the
row as a sensitivity check, or drop it and report only the 2-chart
`{c0, chart_R2}` result (which is genuinely measured and matches exactly:
oracle 50.0%, random 47.5%, spread +2.5pp, CI [0.0,+7.5]).

**Attack 2 — is the 2.5pp spread itself a strong finding, or a single-episode
artifact?** `RESULTS_AUDIT.md` §7(j-3) finds, independently and not
previously stated anywhere in the project's documents, that **19 of the 20
episodes have identical outcomes between baseline and chart** — the entire
2.5pp spread, the 47.5% random rate, and the CI are a restatement of the
outcome of a **single episode** (episode 17: baseline fails, chart
succeeds). This is also why both CIs' lower bound lands at exactly 0.0 — a
structural floor effect of the oracle-minus-random construction, not a bug,
but one that makes the "spread" framing (2.5-3.3pp) sound like more evidence
than 1-episode-out-of-20 actually is. **Outcome: this attack SUCCEEDS at
weakening the narrative, but STRENGTHENS the substantive conclusion** — as
`RESULTS_AUDIT.md` states plainly, if the library's oracle-vs-random
denominator is this thin (one discordant episode in twenty), that is
*stronger*, not weaker, evidence that E1's pre-registered 10pp reporting
threshold genuinely cannot be met by this library, and that closing E1
analytically rather than running a full 180-episode E1 was the right call
resource-wise. **Outcome overall: WEAKENED on the "confident closure"
framing, SURVIVES on the substantive "not worth running E1" decision** —
but this nuance (thin sample, structural floor, single-episode driver) must
be stated explicitly, and Attack 1's fabrication issue must be fixed
regardless. **Minimum fix:** disclose the `chart_R1`-proxy assumption
explicitly, disclose the single-discordant-episode driver, and — if the team
wants a real (not proxy) 3-chart spread before the deadline — the cheapest
fix is not a fabricated stand-in but training and evaluating a genuine
`chart_R1` on R2 planning episodes: this needs ~20 already-collected episode
evaluations against an already-trained `chart` × R1 checkpoint (exists per
`E0_RESULTS.md`'s R1 results) — roughly 1-3 GPU-h, cheap, and should be done
before N8 is cited in any paper draft rather than left as a proxy.

### N9 — Verification-gated expansion fires correctly (3 charts committed, Cell B) and stays quiet correctly (0.0% chunks exceed tau, Cell C)

**Attack 1 — is this evidence about the ATLAS method, or about a different
code path?** `PROPOSAL_CODE_ALIGNMENT.md` item A.2 and `CODE_AUDIT.md` §2.1
(both CRITICAL-rated, cross-confirmed) establish that the demonstrated
expansion path is `scripts/run_e2.py:331`'s **direct call to
`Expander.maybe_expand()`**, bypassing `atlas_step()` entirely — and that
`atlas_step()`'s own expansion guard (`atlas/loop.py:140-150`) is
**unreachable** in the only production caller (`atlas/harness_e4.py`), which
hard-codes `next_encoder_output=None` unconditionally. This means: **the
mechanism that fired is real and correctly implemented (`Expander` itself is
sound), but the controller the paper describes and would actually deploy
(`atlas_step`) has never once executed this code path and, as currently
wired, structurally cannot.** N9 as literally worded ("the verification-gated
expansion mechanism has been demonstrated to fire") is true of a library
function; false, or at minimum radically overclaimed, as a statement about
the ATLAS system. **Outcome: DEFEATED as a claim about ATLAS-the-method;
SURVIVES narrowly as a claim about `Expander`-the-function.** This
distinction is load-bearing for C2 and RQ3 (see Section D) and must be made
explicit in any write-up — "we verified the expansion primitive works in
isolation; we have not verified the full controller invokes it."

**Attack 2 — leakage between fit-set and verify-set.** `CODE_AUDIT.md` §9.3
traced `run_e2.py`'s actual caller loop line-by-line and found it CLEAN: the
verification chunk (`trajs[ep+1]`) is genuinely a disjoint, independently-
seeded trajectory not yet folded into the deficit pool at verification time.
**Outcome: SURVIVES** on this specific concern.

**Attack 3 — is the "0.000 chunks exceed tau" data point in Cell C correct?**
`RESULTS_AUDIT.md` §8 independently recounted raw per-chunk UMF scores in
two different files at two sample sizes and found the true rate is **≈1.28%
(2/156 and 1/78)**, not 0.000. This does not overturn the substantively
important "0 commits" result (which remains the best-available L4 summary
number, since the per-chunk commit log itself was never saved), and if
anything the corrected framing ("the probe occasionally triggers and
correctly declines to commit anyway") is a *stronger* demonstration of
verification working than "never even triggers" — but the specific "0.000"
figure as currently stated in `E2_RESULTS.md` is factually wrong by direct
recount. **Outcome: WEAKENED on the specific number, SURVIVES on the
substantive zero-commits conclusion.** **Minimum fix:** correct "0.000" to
"≈1.3%, with 0 commits" wherever cited. Zero compute cost — a one-line
correction against already-collected data.

**Attack 4 — incumbent-selection bias in `maybe_expand()`.**
`CODE_AUDIT.md` §2.6 finds the incumbent "best chart" a candidate must beat
is selected as the argmin **over the verification chunk itself**, giving the
incumbent a look-ahead advantage on the exact data used for the verdict.
This is a conservative bias (harder to commit, not easier), so it does not
inflate the "3 charts committed" figure — if anything it understates it —
but it means "3 charts committed" is not the unbiased quantity the claim's
wording implies. **Outcome: WEAKENED, minor** — the bias direction is
favorable to rigor, not to the paper's claim, so this is a documentation
precision issue, not a threat to the conclusion.

---

## Section C — The novelty claims (C1, C2, C3)

### C1 — Predictive-fitness routing among persistent adapters on a frozen visual world model, novel because no prior work does this specific combination

`LITERATURE_AUDIT.md` §1 ran two broad, recency-weighted searches plus ~20
direct paper reads and found no paper doing measured-rollout-error routing
among a persistent library of adaptation modules on a frozen visual
foundation-model world model in an online deployment setting. The closest
candidates (HERA, EPM-JEPA, verdi) were each read directly and ruled out on
clear, stated mechanistic grounds (memory retrieval not module routing;
single continuously-modulated adapter not a library; whole-model
optimization-strategy transfer across research campaigns not deployment-time
routing). **Attack: is a search, however broad, proof of absence?** No — the
literature audit itself states this plainly ("no threat found in a genuinely
serious search," not "provably absent"). **Outcome: SURVIVES**, with the
appropriate epistemic hedge the audit itself already applies. This is the
strongest-surviving claim in the whole matrix.

### C2 — Verification-gated expansion (commit only after demonstrating on future unseen data)

`LITERATURE_AUDIT.md` §3 read all six comparison papers' actual expansion
mechanisms directly (DEN, CN-DPM, Dynamic TMoE, ShiftEx, MBCD, CLARE) and
confirmed all six commit a new module on the same data that triggered
detection, with no held-out future verification step in any of them.
**Attack: even if the idea is novel in the literature, does the project's
own implementation demonstrate it?** No — see N9 Attack 1 above: the
controller that would actually run this in deployment (`atlas_step`) cannot
reach its own expansion branch. **Outcome: the idea's novelty SURVIVES
against the literature; the project's empirical demonstration of it is
DEFEATED as a claim about the ATLAS system** (though it SURVIVES narrowly as
a claim about the `Expander` primitive in isolation, per N9). A paper
claiming C2 as an *idea* is on solid ground; a paper claiming C2 as
*demonstrated* is not, given the current wiring.

### C3 — UMF validated against planning success, "addressing JEPA-WM's open appendix question"

**Attack — is the "open question" framing accurate?** `LITERATURE_AUDIT.md`
§5 read JEPA-WM's Appendix G.3 directly and found it is not actually left
open: it runs a real between-model, across-training-epoch Spearman
correlation study and finds moderate-to-strong positive correlation
(ρ≈0.70-0.86 on Push-T/Wall). **Outcome: WEAKENED.** The proposal's framing
overstates the gap. The genuinely defensible and narrower claim — that
ATLAS's granularity (within-arm, episode-level, for adapter *selection*) is
different from JEPA-WM's granularity (between different model
architectures/training runs) — SURVIVES and is where the actual dissociation
finding (N2) lives, since JEPA-WM's coarser between-model methodology would
not have surfaced a within-arm-fine/across-arm-blind pattern. **Minimum
fix:** reword from "addressing an open question" to "extending JEPA-WM's own
appendix analysis to a new granularity." Zero compute cost.

---

## Section D — Never-run claims (RQ0, RQ1, RQ3, RQ4, L-1, C4, G-1) and process claims (S-1 through S-8)

These are addressed together because the attack is structurally identical
for all of them: **there is no result to attack, only an absence, and the
absence is itself the finding.**

- **RQ0** (E0 capacity decision rule): the pre-registered ≥90%-of-full's-gain
  rule is undefined, not merely inapplicable — `full`×R1 was never trained at
  all (`PROPOSAL_CODE_ALIGNMENT.md` H.3), so even setting aside the
  negative-gain problem the project cites, the rule has no denominator for
  half its 2x2 design. The R2 trio's training budgets are also unrecorded
  and not confirmed matched (H.3). **Outcome: DEFEATED as a pre-registered
  decision procedure.** What survives is a descriptive statement: "at the
  budgets and protocol actually used, no adapter kind cleared an
  after-the-fact 15pp bar that was invented during analysis and appears in
  neither design document" (S-6). This is honest and fine to report **only**
  if stated exactly this way, not as "RQ0's pre-registered criterion was not
  met."

- **RQ1** ("THE GATE," E1): **UPDATE 2026-08-27 pass 2 — REOPENED.** Originally closed analytically via N8, which rested on a single discordant episode and an undisclosed fabricated proxy. The missing evaluation has now been run for real (see N8's update above): spread 13.3pp, CI [3.3,25.0], clears the 10pp bar. **Outcome: the decision not to run a full E1 no longer SURVIVES as reasonable resource allocation — the denominator this argument required not existing has been shown, on real data, to exist.** Whether to actually commit to a real E1 run given remaining time/budget is now a genuine open decision, not a settled one.

- **RQ2** (Cell B/C, appearance vs dynamics): this is the one arm of the
  ladder with real evidence — N6/N7's Cell B margin and N9's Cell C
  near-zero commit rate. **Outcome: SURVIVES** as the strongest empirically
  grounded claim in the paper, modulo the N7 attribution fix and N9's 0.000→
  1.3% correction above.

- **RQ3, RQ4, L-1, C4, G-1**: zero episodes exist for E3/E4/E5
  (`RESULTS_AUDIT.md` §5, confirmed by direct directory listing: no `e3*`,
  `e4*`, `e5*` under `atlas_out/`). Beyond the absence itself,
  `PROPOSAL_CODE_ALIGNMENT.md` items A.2/A.3, C.1, C.2, F.3 and
  `CODE_AUDIT.md` Part 2 independently found that **if E4 were launched
  today as-is, it would not measure what the ladder claims to measure**:
  arm 6 cannot commit a chart (dead verification), arm 2 is behaviorally
  identical to arm 3 (dead persistence rung), arm 3→4 differs by four things
  not one (buffer size, conditional refinement, offline pre-training,
  routing), and any ATLAS arm run alone crashes (`--profile`, resumed shards,
  Modal's proposed one-container-per-arm layout would all hit this).
  `CODE_AUDIT.md` explicitly flags the most dangerous version of this: under
  the project's own **default** `expansion_start_library="full"`, correct
  ATLAS behavior and the A.2 bug's behavior are **both** "0 commits" — so a
  naive E4 run would produce a result that looks like a genuine, positive
  finding ("verification is highly conservative") and is entirely a wiring
  artifact. **Outcome: DEFEATED, comprehensively, for all five claims.**
  There is no attack to make beyond recording this, because there is no
  result — but the more important point for the closing section is that
  **these bugs must be fixed before E4 is launched at all**, or a
  launch-under-deadline-pressure will produce exactly the kind of false
  positive this audit exists to prevent. Given ~78 hours and ~$70 remaining,
  fixing the three CRITICAL/HIGH E4 wiring bugs (A.2's `None` hard-code,
  C.1's missing `reset()` call, F.3's `requires_grad` freeze order) is code
  work of perhaps 2-4 hours; even after fixing them, a single properly
  powered E4 run at the plan's 360-episodes-per-arm x 7-arm spec is a
  multi-day, tens-of-GPU-hour undertaking that does not fit in the remaining
  window at any reasonable confidence level. **This experiment cannot be
  salvaged for this submission — see the closing recommendation.**

- **S-1 through S-8** (process claims): S-5 is now CONFIRMED (all four
  sub-claims true against the code, `CODE_AUDIT.md` §9.1) — `closed_loop`
  was tested with an instrument ~9x weaker in CEM search budget and at the
  opposite extreme of replan frequency from the eval protocol, so
  `E0_RECOVERY_PLAN.md`'s "clean rejection" framing is **DEFEATED** and
  should not be cited without this caveat in any paper draft. S-2 ("all
  headless gates pass") is **WEAKENED**: G2 asserts nothing at all
  (`CODE_AUDIT.md` §4.2, `if ... : pass` — a literal no-op that prints
  PASSED unconditionally) and G5 is a tautology that cannot fail
  (`CODE_AUDIT.md` §4.3) — both gates "pass" in the sense of "ran to
  completion," not in the sense of having tested anything. G1 genuinely was
  rewritten and does test something real, but only for unrefined charts, so
  it cannot detect the confirmed `restore_()` defect. **None of these gate
  weaknesses have corrupted any number currently on disk** (every
  clean-verdict item in `CODE_AUDIT.md` Part 8 confirms the properties these
  gates were supposed to check hold anyway, verified independently from raw
  data rather than from the gates) — but "the gates pass" cannot be cited as
  independent evidence of correctness in a paper without this caveat.

---

## What survives cleanly, stated plainly

For a project this deep into an audit, it is worth stating explicitly what
did **not** get defeated, so the closing recommendation is not read as
uniformly pessimistic:

1. **N3/N3b (planner cost-ranking collapse under R2, with a clean R0
   control)** is the single strongest, most carefully verified finding in
   the project — mechanism sound, arithmetic independently reproduced to 3
   significant figures, novel at the specific measurement level even inside
   a crowded 2026 literature cluster.
2. **C1's novelty claim** survives a genuinely serious search.
3. **N6's confusion-matrix structure** (UMF beats S-dyn, and S-dyn's failure
   mode is qualitatively degenerate — defaults to c0 regardless of regime)
   is real and well-supported, modulo the regime-label-vs-planning-outcome
   framing caveat.
4. **Every statistics function in `atlas/stats.py`** (paired bootstrap,
   McNemar, normalised recovery) is independently verified clean, and every
   pairing claim checked against raw data (N1, N4, N5, N8) is confirmed
   genuine — zero unpaired-comparisons-masquerading-as-paired were found
   anywhere the audit checked.
5. Across roughly 60 individually recomputed numeric claims spanning N1
   through N9 (`RESULTS_AUDIT.md`'s own tally), **zero were arithmetically
   wrong.** Every defeated or weakened claim above is a problem of framing,
   attribution, disclosure, or protocol scope — not of fabricated or
   miscalculated numbers, with the one clear exception of N8's undisclosed
   `chart_R1` proxy.

---

## The strongest honest paper this project can submit

Stated bluntly, per the instruction, because a diplomatic answer here is a
useless one.

**This project has one genuinely strong, defensible, novel empirical result
(N3/N3b) and one well-supported secondary result (N6/N7, appearance-vs-
dynamics routing) sitting inside a project whose two headline
ambitions — E1's "THE GATE" and E4's continual-stream ladder — have either
been closed on a data point that turns out to rest on one episode and one
undisclosed fabrication (E1/N8), or never run at all and would not measure
what it claims to if launched under deadline pressure (E4). The project
cannot be rescued into "ATLAS the continual-learning system works" in 78
hours. It can be rescued into a smaller, true paper.**

### Recommended framing: "Prediction Accuracy Does Not Predict Planning Competence Under a Physics-Regime Shift — And Neither Does Regime-Label Routing Accuracy"

Built entirely from evidence that SURVIVES above:

- **Lead with N3/N3b** as the primary contribution: the CEM planner's own
  cost ranking, not the model's prediction accuracy, is what breaks under a
  regime shift (rho +0.501→-0.072, regret 8.5px→88.1px, converged plans land
  farther from goal in 3/3 seeds) — framed explicitly as extending the 2026
  "planner-ranking-not-predictor-accuracy" cluster (RC-aux, "The Objective
  Is the Bottleneck," Operator-on-F, JEPA-WM's own appendix) to a
  **regime-shift-conditioned** (not horizon-conditioned) setting, with a
  clean no-shift control.
- **Report N1/N2/N4 as a corroborating, explicitly-scoped negative result**:
  a lightweight adapter monotonically improves the world model's own
  prediction-error metric with more data (N4) and correlates with success
  within a fixed planning arm (N2, with the n=92/94 caveat disclosed), but
  produces no detectable improvement in one-shot open-loop CEM planning
  success at N=100 (N1) — stated explicitly as a finding *about one-shot
  open-loop planning combined with off-policy-replay-trained adapters*, not
  as a general statement that adaptation doesn't work. Cite Lambert et al.
  2020 and Grimm et al. 2020 as the lineage.
- **Report N6/N7 as a secondary, smaller positive result** on
  appearance-vs-dynamics discrimination — UMF-based selection beats a
  dynamics-fingerprint baseline that degenerately defaults to the identity
  chart, with N7's causal attribution corrected (drop the "hysteresis fix"
  framing; the margin is proven inert at K=2) and N9's Cell C number
  corrected (1.3%, not 0.000).
- **State C1's novelty claim as written** (it survives cleanly) but **drop
  C2 and C4 as demonstrated claims**, retaining C2 only as an *idea*
  supported by the literature gap, explicitly disclosed as
  not-yet-empirically-validated-in-this-system (the `Expander` primitive
  works in isolation per N9; the controller that would deploy it does not
  reach it).
- **Drop RQ3/RQ4/L-1/G-1 and the continual-stream framing entirely — RQ1/N8
  is UPDATED as of 2026-08-27 pass 2 and no longer belongs in this drop
  list.** N8's fabricated proxy row has been replaced with a real 20-episode
  evaluation; the real spread (13.3pp, CI [3.3,25.0]) clears the reporting
  bar. This is now citable — report it as a real, if still small-sample
  (n=20), result establishing that a routing denominator exists in this
  library, not as a closure argument for skipping E1. Whether to spend
  remaining budget on an actual E1 evaluation is a fresh decision, informed
  by this result rather than resolved by it. Do not attempt E4 under
  deadline pressure — the wiring bugs (A.2, C.1, F.3) mean a rushed run
  risks producing a false-positive "ATLAS commits appropriately
  conservatively" result that is actually a dead code path, which would be
  worse for the project's credibility than not submitting the result at
  all.
- **What this framing must explicitly stop claiming:** that ATLAS is a
  demonstrated continual-learning system (it has never adapted online in
  production — `atlas_step()` has never executed); that E1 empirically
  validates fitness-routing over fingerprint-routing at a meaningful sample
  size (it does not — N8 rests on one episode); that verification-gated
  expansion has been demonstrated for the ATLAS controller (only for the
  `Expander` primitive in isolation); and that the negative planning result
  is a discovery rather than a replication in a new substrate.

This framing costs approximately **zero additional GPU-hours** — every
number it needs already exists on disk and has been independently
recomputed. The remaining 78 hours and ~$70 should go to: (1) the ~10
zero-cost text corrections listed above (N7's attribution, N9's 0.000→1.3%,
N8's proxy disclosure or a ~1-3 GPU-h real `chart_R1`×R2 run to replace it,
the literature citations, the C3 "open question" softening); (2) if time and
budget allow after (1), a properly-powered closed-loop N1 re-run (N5's
extension, ~15-30 GPU-h, the highest-value experiment left if the team wants
to actually resolve rather than merely scope the open-loop-protocol
confound); (3) nothing on E4 — it is not salvageable in this window and an
honest paper should say the continual-stream evaluation is future work,
citing the specific wiring reasons (not just "we ran out of time") so a
future submission is credible.

### Alternative 1: Drop N3 entirely, submit only the E2 result

If the team judges N3/N3b too exposed to the "already a known finding-type"
critique (`LITERATURE_AUDIT.md`'s cluster of four 2026 papers) despite its
mechanistic novelty, a much smaller but very safe fallback is a 2-page note
built entirely around N6/N7 (UMF-based routing discriminates dynamics shifts
from appearance shifts; a fingerprint baseline degenerately defaults to one
chart) plus C1's novelty claim. This drops N1/N2/N3/N4/N5 (the entire E0
planning story) and everything about expansion or continual learning. It is
the least interesting paper of the three options but has the fewest moving
parts to attack and requires zero new experiments — only the N7 attribution
fix and N9's number correction. Not recommended as the primary choice
because it wastes the project's single strongest result (N3), but listed
because it is the lowest-risk option if reviewer skepticism about the
crowded 2026 "objective mismatch" literature is judged to be the dominant
risk.

### Alternative 2: Invest the full budget in a matched-compute closed-loop rerun and lead with a positive result

If the team is willing to spend most of the remaining ~$70/87 GPU-h on a
single bet, re-run N1 under closed-loop planning at matched search compute
and N=100 (cost estimate above, ~15-30 GPU-h). If it reproduces N5's
direction at power, the paper can lead with "adapters help planning success
once feedback is restored" — a genuinely more exciting and less-scooped
claim than N1/N2's dissociation. This is a real gamble: N5's point estimate
(+10pp) could easily wash out at N=100 given only 4/20 discordant pairs at
N=20, and a null at N=100 closed-loop would leave the paper with N1
(open-loop null) and N5 (closed-loop null, powered) — i.e. strictly less
interesting than either the recommended framing or Alternative 1, since it
would remove the one piece of evidence (the unpowered nas=2 point estimate)
currently pointing toward "the protocol, not the method, explains N1."
**What it must drop if it fails:** the entire "protocol confound" framing
that currently protects N1 from being read as a clean negative result about
the method — a failed closed-loop rerun would convert N1 from "weakened by
an alternative explanation" to "SURVIVES as a clean negative," which is a
worse outcome for the project's positive-result count even though it is a
scientifically cleaner paper. Only recommended if the team has high
confidence in the +10pp direction and is willing to accept that downside.

---

*End of REDTEAM.md. No source file or `atlas_out/` artifact was modified in
the production of this document.*
