# ATLAS — TODO (plain checklist)

**Last updated: 2026-08-27.** This is the one file to check for "what's next." No claim codes (N1, N8, etc.) — everything is described in plain English so you don't need to cross-reference other files to know what an item means.

---

## Done

- [x] **Fixed the fake data point in the routing-benefit argument.** The number used to decide "a smart chart-router can't beat random guessing by enough to matter, so don't bother testing routing for real" was partly fabricated (one of three chart results was a copy-paste of a different result, not a real measurement). Ran the real missing evaluation: with real data, a perfect router actually *can* beat random guessing by a meaningful amount (13.3 percentage points, confidence interval doesn't touch zero). This reopens the question of whether a real routing evaluation is worth running.
- [x] **Strengthened your best/strongest result.** The finding that the planner's own action-ranking breaks down under a physics shift (but ranks fine when there's no shift) now has twice the sample size (20 seeds/regime instead of 10). Same conclusion, tighter confidence intervals — this result is now well-powered.
- [x] **Fixed all the free text/wording problems** — wrong causal claims, a wrong number (0.000 should have been ~1.3%), missing citations, an overstated novelty claim. All applied directly in `PAPER_DRAFT_NOTES.md`, plus one citation fix applied directly in the proposal document itself.
- [x] **Built and staged (but not yet launched) a reduced routing evaluation** — see item 1 below.
- [x] **Ran the missing R1-regime planning-success test (N=40).** Real result: baseline 70.0% vs. chart 60.0%, Δ −10.0pp, CI touches zero, not significant, but negative point estimate (unlike R2's near-flat trend). See item 2 below for detail.
- [x] **Turned your best result into a dose-response curve.** Rho falls smoothly 0.532→0.295→0.169→0.078→0.001 across 5 shift strengths (n=20/point) — a gradual slide, not a threshold break. Now the strongest, most novel evidence in the project. See item 3 below.

## Explicitly decided NOT to do (by you)

- [ ] ~~Train the largest adapter type under the friction regime~~ — you agreed to drop this. It would almost certainly just repeat a result you already have (bigger adapter ≠ better planning), not worth the GPU time.

## My recommendation, NOT yet a decision you've made — still on the table

- [ ] **Continual-learning stream experiment (the one the paper's title is actually about).** I recommended against fixing and running this, because the audit found real bugs in the code that would silently produce a fake-looking result (e.g. one arm can never register a "commit," making it look like the method is being appropriately conservative when it's actually just broken). That's still true. But you never agreed to drop it — it's your call, not a settled decision. If you want to revisit: the bugs are a few hours of code work, then a reduced-scale run (not the full spec) would be needed, cost unknown until scoped. Say the word if you want this back on the list for real, or a fresh look at whether it's worth attempting given remaining time.

---

## Still to do, in priority order

### 1. ~~Launch the reduced routing evaluation~~ — RAN, but the result is UNUSABLE, real bug found

**Ran on `liochessmag`: all three routers (including the perfect-hindsight oracle) got 0% success, 0/60 total.** This is NOT a finding about routing — traced to a real, confirmed bug: this code path samples independently random goal states, but its success check also requires the *agent's own position* to match the goal, which only makes sense for goals drawn from a real correlated trajectory. With random goals this makes success require landing within 20px of an unrelated random point — near-impossible regardless of which chart is applied. **Do not cite this run's numbers for anything.** A known fix already exists (documented in `E0_RECOVERY_PLAN.md`, never applied) — swap in the same dataset-based goal sampling and success check E0's own planning eval already uses. Real cost of doing that fix + a proper rerun: not yet estimated, needs a fresh look.

### 2. ~~Run the missing "does the adapter help planning" test under the OTHER physics regime~~ — DONE 2026-08-27

**Real result (N=40, on `pandereshubham`): baseline 70.0% (28/40) vs. chart 60.0% (24/40), Δ −10.0pp, CI [−27.5, +7.5] (not significant), McNemar p=0.388, 12/40 discordant.** Not significant on its own, but the point estimate is negative here vs. near-flat under R2 — the "no reliable benefit" conclusion holds in both regimes, but the direction isn't stable, which is itself informative (argues for "genuine null" over "neutral/harmless"). Written into `ATLAS_SUMMARY.md` §4.1b, `CLAIMS_MATRIX.md` row N10, `PAPER_DRAFT_NOTES.md` §3.

### 3. ~~Turn your best result into a curve~~ — DONE 2026-08-27

**Real result (n=20/point, on `pandereshubham`): rho falls smoothly and monotonically — 0.532 (no shift) → 0.295 → 0.169 → 0.078 → 0.001 (full shift).** Not a threshold break — a gradual slide proportional to the physics mismatch. This is now your strongest, most novel piece of evidence for the primary mechanism claim. Written into `ATLAS_SUMMARY.md`, `CLAIMS_MATRIX.md` row N3, `PAPER_DRAFT_NOTES.md` §2 (updated as the lead result).

### 4. Check whether your one positive result is measuring the right thing

Your routing result currently checks "did the router pick the chart matching the true regime label" — not "did picking that chart actually lead to a better plan." **Still genuinely open — item 1's attempt to answer this hit a real bug (see above) and produced no usable data, not a null result.** Answering this for real needs either: (a) fix item 1's goal-sampling/success-check bug and rerun, or (b) a different, simpler approach — e.g. reuse E0's already-validated dataset-based goal sampling directly in a small custom script rather than reviving E1's broken harness. Not yet cost-estimated — needs a quick spec pass, and a decision on which approach.

### 5. (Optional, expensive gamble — not required)

Re-run your main negative result (does the chart help planning) with the planner allowed to replan mid-episode instead of committing to one plan upfront, at full statistical power. Could flip your headline result positive. Could also just confirm it as a clean negative. ~15-30 GPU-hours, real money, real risk either way.

---

## Then: write the paper

- [ ] Draft from `PAPER_DRAFT_NOTES.md` — it has the corrected numbers, the recommended framing, a draft abstract, exact citations, and an explicit "do not claim this" list. Do not draft from `ATLAS_SUMMARY.md` directly — use it only as the raw-numbers reference.

---

That's the whole list. Nothing above is required — items 2-5 are ranked by value-for-cost, not by necessity. Item 1 is spec'd and ready; say the word and it launches.
