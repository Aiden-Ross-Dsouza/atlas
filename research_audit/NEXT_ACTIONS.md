# ATLAS — Next Actions

**Last updated: 2026-08-27, pass 2 — the audit is COMPLETE. All five files (`CLAIMS_MATRIX.md`, `EXPERIMENT_STATUS.md`, `LITERATURE_AUDIT.md`, `PROPOSAL_CODE_ALIGNMENT.md`, `CODE_AUDIT.md`, `RESULTS_AUDIT.md`, `REDTEAM.md`) have run to completion, including the red-team pass this file's pass-1 version was waiting on. Section 1's inventory table (above) is rewritten to show what was actually checked. Sections 2 and 3 below are superseded by this pass's summary and by `REDTEAM.md`'s own closing section ("The strongest honest paper this project can submit") — read that file directly for the framing decision; it is more current than anything below.**

---

## What this file is

The action register for the ATLAS pre-submission audit (NeurIPS 2026
Workshop on Continual World Models, Idea Track, deadline 29 Aug 2026 AoE).
Repository root: `D:/Shubham/DeepLearning/Atlas/atlas/`.

Section 1 below is the inventory of pre-existing project documentation, with
a one-line note on whether each document's claims need checking by one of
the audit agents. **Almost all of them do.** Every document listed was
written by a prior Claude Code session, not by a human, which under this
audit's operating rules makes each one an unverified claim about the code at
evidence level L0 — not independent confirmation that anything works. See
`.claude/skills/research-audit/SKILL.md` for the evidence-level definitions.

---

## 1. Prior documentation inventory (as of 2026-08-27)

A scale observation worth stating before the list: the repository contains
roughly **470 KB of AI-authored Markdown documentation** against roughly
**190 KB of Python**. There is more prose asserting what the code does than
there is code. That ratio is itself a risk, because a claim repeated across
six documents reads as corroborated when it has only ever been asserted once
and copied forward.

**Status as of 2026-08-27, pass 2 (all five audit files now complete, including `REDTEAM.md`).** The "Needs checking by" column below is rewritten to show what actually happened, not what was planned. "CHECKED" means an audit agent traced the specific claim to code or raw data; "NOT REACHED" means no agent got to it and it remains exactly as untrusted as any other prior-session claim.

| Document | Size | What it claims | Checked? |
|---|---|---|---|
| `ATLAS_proposal_v7.md` | 26 KB | The scientific case: contributions C1-C4, research questions RQ0-RQ4, related-work positioning, the five-experiment plan. | **CHECKED.** `sota-scout` (novelty, §2/§3.4-3.5 — C1/C2 SURVIVE, C3's "open question" framing WEAKENED, see `LITERATURE_AUDIT.md`), `proposal-code-auditor` (§6-8 as spec, both passes — `PROPOSAL_CODE_ALIGNMENT.md`, 4 structural drifts found in the E4 ladder). |
| `ATLAS_implementation_plan_v2.md` | 28 KB | The build spec: repo layout, hyperparameter table (7.7), the 7-arm ladder (7.4), gates (9), statistics (8). | **CHECKED.** `proposal-code-auditor` (all of it), `code-bughunter` (statistics functions all CLEAN, `CODE_AUDIT.md` §3.1-3.3). |
| `CLAUDE.md` | 26 KB → 34 KB | Operating contract + "current status" §0.1, already flagged stale in the AUDIT NOTE appended 2026-08-27. | **CHECKED, confirmed more stale than its own AUDIT NOTE states.** `CLAIMS_MATRIX.md` S-2/S-3 found an additional gate-level staleness (G2/G5 "pass" but assert nothing/a tautology) not in the original AUDIT NOTE. |
| `ATLAS_SUMMARY.md` | 23 KB | Consolidated results doc, written 2026-08-26/27. **The document a paper would most likely be written from.** | **CHECKED, extensively.** `results-auditor` recomputed essentially every headline number in it (N1-N9) independently from raw `atlas_out/` files. Arithmetic: 100% clean. Framing/attribution: several claims (N7, N8, N9, and N1's implicit scope) need correction before citing — see `REDTEAM.md` and `CLAIMS_MATRIX.md`'s Pass-2 summary for exactly which sentences. |
| `E0_RESULTS.md` | 84 KB | E0's results, newest-first, SUPERSEDED banners on invalidated sections. | **CHECKED** (top section fully; below that, spot-checked, not exhaustive). `results-auditor`, `proposal-code-auditor` item K. |
| `E0_RECOVERY_PLAN.md` | 64 KB | Process narrative behind E0's closure. Status banner declares "E0 IS COMPLETE". | **CHECKED — one specific claim DEFEATED.** The `closed_loop` "cleanly rejected" framing is not supportable: `code-bughunter` confirmed all four sub-allegations in `OPUS_REMAINING_TASKS.md` #10 (S-5) are true — the training data was collected off-policy, at ~9x less CEM compute, and at the opposite extreme of replan frequency from eval. Also: its own prescribed "P5" fix to `atlas/harness.py`'s E1 harness was never applied to code (`PROPOSAL_CODE_ALIGNMENT.md` L.6) — doesn't affect any current number, but the doc's own action item was never done. |
| `E2_RESULTS.md` | 14 KB | E2's routing-accuracy results — the project's one clearly positive finding. | **CHECKED.** `results-auditor` (recomputed N6/N7 exactly), `scientific-redteam` (the "correct = regime label" tension — WEAKENED, N6 measures selector-vs-label accuracy, not planning competence, per `REDTEAM.md` Section B). N7's "hysteresis fix" causal attribution is DEFEATED (`CODE_AUDIT.md` §6.1 proves the margin is mathematically inert for any 2-chart library). |
| `HANDOFF.md` | 24 KB | Navigation index + §7 results existing nowhere else: E1's analytic closure (7.1), G1 rewrite (7.2), expansion demo (7.6). | **CHECKED — §7.1 is the single most serious finding of the whole audit.** `results-auditor` found the 3-chart oracle-vs-random comparison's `chart_R1` row is a duplicated baseline array presented as real data, undisclosed, and the whole 20-episode comparison is driven by one discordant episode. §7.2 (G1 rewrite) CONFIRMED genuine. §7.6 (expansion demo) CONFIRMED real but scoped: it demonstrates the `Expander` primitive, not the `atlas_step()` controller, which cannot reach it (`CODE_AUDIT.md` §2.1). |
| `OPUS_REMAINING_TASKS.md` | 16 KB | Prior review's outstanding to-do list; §B lists 14 un-applied write-up corrections. | **CHECKED, all cross-referenced.** `scientific-redteam` treated §B as a pre-supplied weakness list; #10 (S-5) CONFIRMED true, #13 (S-7, regime-label routing) independently corroborated by the redteam's own N6 attack. |
| `E3_E4_IMPLEMENTATION_PLAN.md` | 36 KB | Plan for the continual stream, incl. two-agent file-ownership split. | **CHECKED.** `code-bughunter` read `run_e4.py`, `harness_e4.py`, `loop.py`, `expand.py`, `adajepa.py` in full — found 3 CRITICAL bugs that would silently corrupt any E4 run launched as-is (dead verification path, arm2=arm3, suspected motion-gate miscalibration). None of this plan's content is contaminated (E4 never ran), but its "ready to launch" implication is not accurate until these are fixed. |
| `code-review.md` | 45 KB | Engineering bug log, numbered bugs (e.g. "Bug #7"). | **Spot-checked, not exhaustive.** `code-bughunter`/`proposal-code-auditor` L.6 cross-checked Bug #7's `PlanEvaluator` claim and found it consistent with current code. |
| `REGIME_DESIGN_REVIEW.md` | 23 KB | Derivation that mass-scaling is physically inert against a kinematic pusher; stands in for gate G4 (never run). | **CHECKED, and a related but distinct hazard was independently ruled out.** Two independent direct reads of the actual `pusht_env.py::_setup()` (`code-bughunter` + `proposal-code-auditor`) confirm every `reset()` fully rebuilds the physics objects at hard-coded defaults — so regime settings cannot persist across episodes in an alternating stream. This is the single biggest E4 safety concern and it is now cleared. |
| `ACTION_SAMPLING_REVIEW.md` | 19 KB | Replacing uniform random actions with an aimed-walk collector; contact rate 13-17% → 100%. | **NOT independently re-derived this pass** (flagged as unread in `PROPOSAL_CODE_ALIGNMENT.md` L.6's closing note) — still an open item if anyone wants to chase it, but nothing downstream currently depends on it being wrong. |
| `E0_DIAGNOSIS_AND_PLAN.md` | 19 KB | Diagnosis of the 2026-08-25 rollout bug. | **CHECKED.** `code-bughunter` independently verified the fix is correct for everything checked: time base, proprio threading, output alignment (`CODE_AUDIT.md` §5.1-5.2). |
| `E0_IMPLEMENTATION_PLAN.md` | 27 KB | Superseded by `E0_RECOVERY_PLAN.md`. | Historical, correctly low priority, not separately checked. |
| `E0_HANDOFF.md` | 18 KB | Earlier handoff, superseded by `HANDOFF.md`. | Historical, correctly low priority, not separately checked. |
| `README.md` | 8 KB | Repo README. | **NOT checked** — still genuinely open if anyone wants it, but nothing in the audit depends on it. |
| `graphify-out/GRAPH_REPORT.md` | 47 KB | Auto-generated codebase knowledge-graph report. | Used as a navigation tool by audit agents where available; not itself an audit target. |
| `modal/README.md` | small | Modal deployment notes. | Low priority, not checked. |
| `.claude/skills/codebase-search/SKILL.md` | small | Project skill instructing use of graphify over grep. | Not a claim about results. |

**One-line summary of the inventory:** every document in the top half of
this table makes results claims that no one has independently verified, and
the three most load-bearing single-sourced claims in the project — E1's
analytic closure (`HANDOFF.md` 7.1), the rollout-bug fix
(`E0_DIAGNOSIS_AND_PLAN.md`), and the regime-reality argument
(`REGIME_DESIGN_REVIEW.md`, standing in for the never-run gate G4) — each
exist in exactly one document and were each produced by the session that
asserts them.

---

## 2. START HERE IF YOU ARE A NEW SESSION

*Written 2026-08-27 ~03:30 local, at the end of audit pass 1. If you are a
Claude Code session that has just started on this repository and has no
memory of the audit, read this section in full before doing anything.*

### 2.1 What to do first, in order

1. Read `research_audit/EXPERIMENT_STATUS.md` — what is implemented, what has
   actually been run, what has zero results.
2. Read `research_audit/CLAIMS_MATRIX.md` — every claim, its evidence level,
   and which experiment tests it. Claim IDs (C1, N1, G-1, ...) used
   throughout the audit are defined there.
3. Read `research_audit/LITERATURE_AUDIT.md` — **complete**, and it contains
   the single most consequential finding of the audit (see 2.3 below).
4. Read whichever of `PROPOSAL_CODE_ALIGNMENT.md`, `CODE_AUDIT.md`,
   `RESULTS_AUDIT.md` exist and are populated. **Check each one's first line
   for a `TRUNCATED EARLY` banner** — if present, that file is partial and
   its own "What I did not get to" section lists precisely where to resume.
5. Read the `## AUDIT NOTE` at the bottom of `CLAUDE.md` — it lists the
   places where `CLAUDE.md`'s own section 0.1 is stale. Do not trust section
   0.1 over the audit files; it is older than they are.

### 2.2 State of the audit as of 2026-08-27 pass 1

| Item | State |
|---|---|
| `CLAIMS_MATRIX.md` | Complete (written by the main session) |
| `EXPERIMENT_STATUS.md` | Complete (written by the main session) |
| `LITERATURE_AUDIT.md` | **Complete** — all 8 search priorities finished |
| `PROPOSAL_CODE_ALIGNMENT.md` | Partial — agent was stopped early on budget |
| `CODE_AUDIT.md` | Partial — agent was stopped early on budget |
| `RESULTS_AUDIT.md` | Partial — agent was stopped early on budget |
| `REDTEAM.md` | **NOT RUN AT ALL** — still the placeholder |
| `NEXT_ACTIONS.md` | This file |

The audit was cut short because the session's token budget ran low, not
because the work was finished. The three partial files each carry their own
resume list.

### 2.3 The findings that must not be lost

1. **The project's headline negative result is a replication, not a
   discovery.** `LITERATURE_AUDIT.md` establishes that the dissociation
   between world-model prediction accuracy and downstream planning success
   is well documented: Lambert et al. 2020 "Objective Mismatch in
   Model-based RL" (arXiv:2002.04523); Grimm et al.'s Value Equivalence
   Principle; RC-aux "Predictive but Not Plannable" (arXiv:2605.07278,
   already cited in this project's own proposal); and **"The Objective Is
   the Bottleneck" (arXiv:2608.12959), which is NOT cited by this project
   and which shows a CEM + JEPA planner failing while the predictor stays
   informative** — very close to this project's own N3 mechanism result.
   Anyone writing this paper must read 2608.12959 before claiming novelty
   for the mechanism finding.
2. **Claims C1 and C2 survive the novelty check.** No module-library method
   was found that routes among persistent adapters by measured predictive
   fitness on a frozen visual world model, and none of DEN, CN-DPM, Dynamic
   TMoE, ShiftEx, MBCD or CLARE verifies on future unseen data before
   committing a module. All six were read directly.
3. **Two citation corrections:** MBCD is AAMAS 2021, not ICML 2021 as the
   proposal states. The C3 claim that ATLAS "addresses JEPA-WM's open
   appendix question" overstates the gap — JEPA-WM Appendix G.3 already
   runs a correlation analysis with a positive answer, at a coarser
   between-model granularity than ATLAS's within-arm claim.
4. **Nothing in this project has ever tested continual learning.**
   `atlas_out/` contains no `e4` directory; `atlas/loop.py::atlas_step()`
   has never executed in production and no gate exercises it. The target
   venue is a workshop on *continual* world models. See
   `EXPERIMENT_STATUS.md` section 5 and `CLAIMS_MATRIX.md` row G-1.
5. **Every headline planning number comes from a single open-loop CEM plan
   per episode** (`num_act_stepped=6`, `frameskip=5`, 30 raw steps = one
   search, no reaction to what happens). This is the most parsimonious
   alternative explanation for the entire negative result and the project
   cannot currently rule it out — the one run that could (`nas=2`,
   `atlas_out/e0_planning_nas2`) has N=20 and only 4 informative discordant
   pairs.
6. **The charts were built differently from how the method specifies them.**
   The method (proposal section 6, plan section 7.6) specifies one SGD step
   per replan on a 5-transition buffer. E0's charts are ~2000-step offline
   fine-tunes on 20-100 pre-collected trajectories. E0 therefore measured
   the capacity of a different object than the one ATLAS deploys.

### 2.4 Immediate next steps

1. **Run the red team.** The agent definition exists at
   `D:/Shubham/DeepLearning/Atlas/.claude/agents/scientific-redteam.md` and
   will be auto-registered in any new session (agents are discovered at
   session start, which is why it could not be used in the session that
   created it). Invoke it with the Agent tool, `subagent_type:
   "scientific-redteam"`. It reads the other five files and attacks every
   claim; its brief ends by asking for the strongest honest paper the
   project can still submit. Run this before writing any paper text.
2. **Finish the three truncated audits**, or at minimum the items each file
   flags as unreached. The single highest-value unfinished check is whether
   the multi-replan (closed-loop) code path feeds correct observation and
   proprioception context to the second and later replans — if it does not,
   the `nas=2` result is meaningless.
3. **Then, and only then**, decide the paper's framing.

---

## 3. Decisions needed from the human

These are genuine decisions, not things a session should resolve on its own.

1. **Do we run E4 (the continual stream) at all?** It is the only experiment
   that would make this a continual-learning paper, it has never executed,
   and `CODE_AUDIT.md` should be consulted first for what would silently
   break. Estimated ~$17 for the full 7 arms x 6 segments x 20 episodes x 3
   seeds; a reduced stream is cheaper. Budget was roughly $90 of GPU credit
   as of 2026-08-26.
2. **Do we run a properly powered closed-loop (`nas>=2`) arm?** This is the
   cheaper of the two and it directly defends against the strongest reviewer
   objection to the negative result.
3. **Given the literature finding (2.3 item 1), what does the paper
   actually claim?** The general dissociation is taken. The candidate
   remaining contributions are C1 (routing works — E2 is the positive
   result), C2 (verified expansion — demonstrated but never evaluated
   against detect-only), and the regime-specific quantification of planner
   ranking collapse. This is a framing decision for the human, informed by
   `REDTEAM.md` once it exists.
