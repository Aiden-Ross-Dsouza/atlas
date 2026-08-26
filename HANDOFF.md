# HANDOFF — read this first if you're picking up this project cold

*Written 2026-08-26, end of a session that ran E0 through to closure. This file is an index
and a set of practical notes — it does not restate results that already have a home. If
something here conflicts with `CLAUDE.md`, `CLAUDE.md` wins.*

---

## 0. The one thing to understand before anything else: two agents, one repo

This project is being worked by **two agent sessions in parallel, on the same local
checkout**, by design — not an accident of history. `E3_E4_IMPLEMENTATION_PLAN.md` (written
2026-08-25) explicitly sets this up: one agent owns E0/E1, a second owns E3/E4, with a
file-ownership split in its §0.3 to keep them from colliding. **If you are a fresh agent
starting a new session on this repo, assume another session may be active concurrently,
and check `git status` / file mtimes before trusting your own memory of "current state"** —
this session was repeatedly surprised mid-task by files changing under it (chart_full_R2.pt
briefly corrupted, `atlas/regimes.py` and `scripts/run_e0.py` edited from outside this
session, and a full `E2_RESULTS.md` + E2 result set appeared on disk without ever being
mentioned in this session's own chat).

**Practical consequence:** before believing any status claim in this file or elsewhere,
re-check the actual files. Everything below was true as of this session's last look; it may
not be true five minutes later.

---

## 1. Where every result actually lives — read this before re-deriving anything

| Question | Answer is in |
|---|---|
| Is E0 done? What's the verdict? | `E0_RECOVERY_PLAN.md` (status banner + §0–§0.8 = the narrative/process record) and `E0_RESULTS.md` (the top `🟢 E0 CLOSED` section = the results-only summary; everything below it is superseded history, kept for the record, read newest-first) |
| Is E1 running? | No, and it won't be — see §0's day-one argument in `E0_RECOVERY_PLAN.md`, confirmed by every result since. Don't re-litigate this without new evidence. |
| Is E2 done? What's the verdict? | **Yes — `E2_RESULTS.md`.** This is a *separate file*, not a section of the E0 docs. It found the project's one clean positive result (UMF routing discriminates dynamics shift from appearance shift on R2) and is explicit that this does **not** rescue E0 (it validates the selector, not the chart library). **(2026-08-26 update, same file's own top section):** the `current_idx=0`-always and 2-chart-library limitations flagged below are now addressed — sequential hysteresis fixed, and a 3-chart confusion matrix confirms UMF's advantage survives a 3-way library while S-dyn stays near chance. |
| Is E0's N=20 power problem resolved? | **Yes — `E0_RESULTS.md`'s new top section (2026-08-26).** Re-ran baseline vs. `ln_act`/R2 at N=100 paired, same protocol. The null replicates at 5× power (CI roughly halved, point estimate near zero) — this was **not** an underpowered N=20 artifact. Also yielded the first statistically significant within-arm UMF-vs-success Kendall τ in the project. |
| Is E3/E4 done? | No. `E3_E4_IMPLEMENTATION_PLAN.md` is the plan (phases 0–8 + appendix A). `scripts/smoke_e4.py` exists (its Phase-4 smoke test), but there is no results file yet — the real run is gated behind a 🛑 STOP (Phase 5: profile → report → get budget approval) that, as far as this session knows, has not happened. If `scripts/run_e4.py` no longer raises `NotImplementedError`, more has landed since this was written — check the file directly. |
| What are the physics regimes, and why? | `E0_RECOVERY_PLAN.md` §0.3–§0.5 (the mass→friction and elasticity→damping corrections, with the mechanism). `REGIME_DESIGN_REVIEW.md` has the original mass-cancellation writeup. |
| Is E1 closed, and on what evidence? | **Yes, analytically — see §7 below.** Not just "we argued it isn't worth running": the oracle−random spread was *computed* from the existing paired R2 episodes. Numbers live in §7 of this file and nowhere else yet. |
| What is the gate status? | `CLAUDE.md` §0.1 is **STALE on G1**. G1 was rewritten and now passes headless — see §7. G4 remains the only skipped gate. |
| What are the binding rules I can't silently change? | `CLAUDE.md` §1 (non-negotiables), §1.7 (fixed hyperparameters table). This overrides everything else including this file. |
| Where's the plain-English summary for non-technical colleagues? | A published Claude Artifact, "The Overshoot Problem" (link was shared in-chat this session, not saved to a repo file — ask whoever ran this session, or regenerate from `E0_RESULTS.md`'s closing section if the link is lost). It covers E0 only, predates the closed_loop/E2 results in their final form — treat as slightly stale if you find it. |

**Do not duplicate results across files.** If you learn something new about E0, it goes in
`E0_RECOVERY_PLAN.md`/`E0_RESULTS.md`. E2 goes in `E2_RESULTS.md`. E3/E4 goes wherever that
plan's own convention lands (check for an `E3_E4_RESULTS.md` or similar before creating one).

---

## 2. Current state in one paragraph

**E0 failed its pre-registered test, and three independent attempts to rescue it (bigger
adapters, better training data via live on-policy CEM collection, a refined UMF metric) were
each tested and each failed too — E0 is closed as a negative result.** Separately, **E2 (which
tests the router/selector, not chart quality) passed its decisive cell on the R2/damping regime
using the R1 chart trained late in this session** — a real, if narrow, positive result that
does not contradict E0's failure (different question). E1 is correctly not being run. E3/E4 has
scaffolding and a smoke test but no real run yet. The two-agent parallel-work pattern (§0 above)
means the fastest way to get an accurate picture is: read `git status`, `git log -5`, and the
mtimes under `atlas_out/`, then read `E0_RESULTS.md`'s top section and all of `E2_RESULTS.md`.

---

## 3. What this session personally verified vs. what it only relayed

This matters because several results in the docs above came from the *other* agent's work,
relayed into this session's chat by the user, and this session did not independently re-run
them. Treat the distinction below as real, not pedantic — it affects how much to trust a number
without re-checking it yourself.

**Independently run and verified by this session** (real Modal jobs, logs read directly,
pairing/regime_config checked in code): P0–P4's core matrix, the R0 confound check, the
`nas=1` diagnostic (launched, found to be far more expensive than estimated, cancelled by the
user), the `closed_loop` chart's training and full 5-arm planning comparison, the local backup
of every chart `.pt` file (including catching and fixing one corrupted download).

**Relayed from the other agent via the user, not independently re-run by *that* session:** the
`lora4`/`ln_act` R1 chart training numbers (`atlas_out/e0_v6_R1/`), the UMF-token-localization
diagnostic (`atlas_out/umf_locality.json`, `scripts/diagnose_umf_locality.py`), and **all of
E2** (`E2_RESULTS.md`, `atlas_out/e2*/`, `scripts/run_e2.py`, `scripts/make_e2_figure.py`).

**⚠️ CORRECTION added by the planning session (2026-08-26).** The paragraph above is accurate
about what the *E0 session* did, but it should not be read as "nobody verified E2." Those three
items were produced and verified BY THE OTHER (planning) SESSION, from raw logs, in this order:
it launched the R1 training on Modal and read `results.json` off the volume; it wrote and ran
`diagnose_umf_locality.py` locally (catching a corrupt `chart_full_R2.pt` in the process); it
wrote `run_e2.py`, ran all four E2 configurations on Modal, and derived every number in
`E2_RESULTS.md` by re-reading `e2_episodes.jsonl` per-condition rather than trusting the pooled
`e2_summary.json` — which is how the pooled-vs-per-condition discrepancy (0.828 vs 0.880) was
caught in the first place. So each of these HAS been derived from logs once, by the session that
produced it; neither session has independently reproduced the OTHER's numbers. That is a normal
level of verification, not a gap — §6's "verify E2 end to end" is lower priority than it reads.

---

## 4. Practical gotchas from this session (save yourself the debugging time)

- **Modal's `modal app logs` has a retention/tail limit.** Early debug output (contact-rate
  lines, "Loaded N train & M eval trajectories") reliably scrolls out of what `modal app logs`
  returns once a run has produced enough later output (e.g. thousands of tqdm lines). If you
  need that information after the fact, it's gone — capture it while the run is still short, or
  don't rely on retrieving it later.
- **`scripts/run_e0.py`'s `closed_loop` collection loop had no progress output** until this
  session added a `tqdm` bar (`collect_{source}_{regime}`, with per-chunk postfix showing
  contacts/attempt) — if you're on an older checkout this fix predates, expect long silent
  stretches that look like a hang but aren't. Rough sizing rule that held up empirically:
  collection time scales linearly with total CEM searches = trajectories × (traj_len /
  frameskip). A 20+8 trajectory closed_loop collection took ~75 minutes at the default
  100×10 collection budget — don't assume `nas=6` planning-episode timings transfer to
  `closed_loop` collection; they're different workloads.
- **`num_act_stepped=1` evaluation is ~6× the cost of `nas=6` per episode**, not the same cost
  — this session initially estimated the cancelled `nas=1` diagnostic at "~50 min" and it was
  actually tracking toward ~4h52m before being cancelled. Each replan at `nas=1` is a *full*
  CEM search; `nas=6` only pays for one search per episode.
- **A batch-download loop over multiple `modal volume get` calls can silently truncate a file**
  if the shell hits a timeout mid-transfer — the truncated file can still land at a plausible
  byte size and only fail when you actually `torch.load()` it. If you're pulling several chart
  `.pt` files in one batched command, verify each one loads afterward, not just that the
  download command reported success.
- **`Chart.n_params()` sums every stored tensor, including frozen restore-reference copies for
  `lora4`** — its raw output (10,292,640) is not `lora4`'s real trainable capacity (118,176).
  Every results doc in this repo already uses the corrected number; don't let the raw one back
  in from a fresh `Chart.n_params()` call.
- **Windows console + this checkpoint's logging emits unicode (🔮📉🧠) that crashes on the
  default `cp1252` codec.** Prefix Modal/Python invocations with `PYTHONUTF8=1
  PYTHONIOENCODING=utf-8` in PowerShell/git-bash on this machine, or expect `UnicodeEncodeError`
  noise (harmless, but clutters output and can obscure real errors near it).
- **The local GPU is 6GB (RTX 4050 laptop)** — fine for P2a-fixed offline chart training (even
  `full`, 20.8M params, trains locally without OOM once the per-trajectory backward fix is in),
  but the real substrate CEM planning config (300×30) needs ~13.5GB and must run on Modal
  (`modal/modal_e0_planning.py`, L4 GPU, $0.80/h).

---

- **`chart.restore_()` does NOT restore the pretrained weights** — for every kind except
  `lora4` it is literally `self.apply_(predictor)` (`atlas/chart.py:107-127`), i.e. it re-applies
  *that same chart*. The documented way back to pretrained is `c0.restore_(predictor)` — restore
  via the IDENTITY chart. **10 production call sites use the `chart.restore_()` form**
  (`score.py:99`, `router.py:159/186`, `harness.py:241/371`, `harness_e4.py:252`, `loop.py:247`,
  `expand.py:230`, `run_e0.py:438`). That is self-consistent as long as the next `apply_()`
  covers the same parameter names — but any code assuming "the predictor is pristine after
  scoring" is silently wrong. Not refactored (out of scope, nothing currently depends on it);
  flagged so nobody re-derives it painfully. When you need true pristine state, the safe idiom is
  `predictor.load_state_dict(pristine_snapshot)`.
- **`colour_change` is nearly invisible on this env — do not use it as E2's appearance shift.**
  Measured: it alters **5.6%** of pixels (mean |diff| 1.84). Push-T renders are ~97% white
  (mean 248) and an HSV hue rotation is a no-op on desaturated pixels. `dark` changes **100%**
  (mean |diff| 99.6). Plan §6.3 names colour; `run_e2.py` keeps that as the flag default but
  measures the real magnitude on every run and warns below 20%.
- **`atlas/regimes.py::VisualCorruption` had never been run against the real env** and was broken
  three ways before E2: `observation()` called `obs.ndim` on PushTEnv's **dict** obs;
  `gym.ObservationWrapper.reset()` fed PushTEnv's `(obs, state)` **tuple** into `observation()`;
  and `salt_pepper` used an unseeded RNG (which would have broken G5's paired-seed guarantee).
  All fixed. Treat any other never-exercised wrapper in that file with the same suspicion.

## 5. Artifact inventory pointers
 (not exhaustive — check `atlas_out/` directly for the current list)

- E0 training/planning artifacts: `atlas_out/e0_v3_*`, `e0_v4_*`, `e0_v5_*`, `e0_v6_R1*` —
  chart `.pt` files, seed manifests, per-episode JSONLs, all backed up locally as of this
  session (previously only on the Modal volume `atlas-data`).
- E2 artifacts: `atlas_out/e2*/` (six run directories, see `E2_RESULTS.md`'s own artifact table
  for which is which), plus `atlas_out/umf_locality.json`.
- E3/E4: no result artifacts yet as of this session; `scripts/smoke_e4.py` exists for the
  smoke-test phase.

---

## 6. Open items, in rough priority order

*Re-prioritised 2026-08-26 by the planning session, after §7's findings and the budget
correction (§7.5 — ~$90 available, not $12). Time, not money, is the binding constraint.*

1. ~~**Decide what the paper claims, given that E0 is underpowered (§7.3).**~~ **RESOLVED
   (2026-08-26).** N=100 re-run confirms the null at real power (`E0_RESULTS.md` top section) —
   frame this as "replicates at 5× power," not "underpowered."
2. **Training-set-size sweep (~$2).** Still the highest-value run left. Every chart in this
   project trained on **20 trajectories** and that was never varied (§7.4), while
   `E0_RESULTS.md` itself scopes `full`'s failure as "confounded with training-set size."
   That confound is still open and is cheap to close.
3. ~~**High-power decisive eval, N=100 paired (~$7).**~~ **DONE (2026-08-26).** See
   `E0_RESULTS.md`'s top section — baseline vs. `ln_act`/R2, N=100 paired, plus two significant
   within-arm UMF-vs-success Kendall τ values.
4. **E4 smoke (~$1), then E3/E4 full stream (~$17).** E4 *is* the "does continual learning
   help?" experiment. Smoke first: `atlas/loop.py::atlas_step()` has still never run end to end,
   so a cold 21-GPU-h launch will likely burn hours on wiring bugs. §7.6 de-risks E3 itself —
   the expansion path is now demonstrated live.
5. **`nas=2` closed-loop arm (~$4).** Closes explanation B (horizon compounding), the one E0
   hypothesis never tested because the `nas=1` diagnostic was cancelled on cost (§7.4). Still
   the only untested item from the original three protocol confounds.
6. **The plain-English "Overshoot Problem" artifact is stale** relative to the final closed_loop,
   E2, and N=100 results — regenerate or update it before sharing externally again.
7. ~~**Verify E2's numbers end to end**~~ — lower priority than previously written. See §3's
   correction: E2 *was* derived from raw per-condition logs by the session that produced it.
   Neither session has reproduced the other's numbers, which is normal, not a gap.
8. ~~**E2's `current_idx=0` / 2-chart-library limitations**~~ — **DONE (2026-08-26).** See
   `E2_RESULTS.md`'s top section: sequential hysteresis fixed, 3-chart confusion matrix run.
9. **This file will go stale** the moment either agent does more work — re-verify §1's claims
   against `git status` / `atlas_out/` mtimes rather than trusting it past its own session.

**Explicitly NOT worth doing:** E1 proper (closed quantitatively, §7.1); E5 (supplementary);
the τ/q sensitivity sweep; repairing the `hybrid` collector (superseded by `closed_loop`, and
both lost); re-proposing UMF token-localization (tested and rejected, §7.7).

---

## 7. Results and status that exist in NO other file (added by the planning session, 2026-08-26)

Everything here was produced after `E0_RESULTS.md`'s "E0 CLOSED" section and after
`E2_RESULTS.md` was written. None of it is recorded elsewhere.

### 7.1 E1 is closed **quantitatively**, not just by argument

Computed directly from the existing 20 paired R2 planning episodes
(`e0_v3_planning_dataset_baseline` vs `e0_v3_planning_dataset_ln_act`), 10,000-sample paired
bootstrap:

| library | SR_oracle | SR_random | spread | 95% CI |
|---|---:|---:|---:|---|
| `{c0, chart_R2}` | 50.0% | 47.5% | **+2.5pp** | [0.0, +7.5] |
| `{c0, chart_R1, chart_R2}` | 50.0% | 46.7% | **+3.3pp** | [0.0, +10.0] |

E1's pre-registered gate needs **≥10pp** before `normalised_recovery` is reportable
(`atlas/stats.py:35` returns `None` below it). A *perfect* oracle over the real library beats
random selection by ~3pp. No routing algorithm can create a denominator the library does not
contain.

**Scope this honestly in the paper:** it is a bound from single-regime paired data, not an E1
stream run. The oracle is best-case by construction and the spread is a quarter of the gate, so
the stream would confirm a ceiling that is already forced — but say that; do not imply E1 ran.

### 7.2 Gate status — `CLAUDE.md` §0.1 is STALE

**G1 was rewritten and now passes headless.** The previous implementation tested nothing it
claimed: it constructed a `Chart`, **never applied it**, never called the model, and compared two
env rollouts driven by *separate unseeded* `action_space.sample()` draws — so it could only ever
compare different action sequences against each other. It also used gymnasium-style
`reset(seed=)` and 5-tuple `step()` against this legacy-gym env.

It now checks what §1.4 actually requires, for `ln_act` and `lora4`, on synthetic latents with no
env: (1) applying an identity chart leaves predictor output **bit-identical** (`torch.equal`, not
`allclose`); (2) `restore_()` returns every tensor bit-identically; (3) output after restore
matches frozen. Verified: `G1: identity chart bit-identity check... PASSED`.

Because it no longer needs a live env it moved into the headless group and runs under `--all`.
**G4 is now the only skipped gate.** Any claim that "G1 passed" before 2026-08-26 was unfounded.

### 7.3 The methodological weakness nobody has recorded: E0 is UNDERPOWERED

Every E0 planning comparison is **N=20 paired episodes**, giving CIs of roughly ±15–20pp.
`ln_act`'s headline +5.0pp has CI [−10, +20] — **that interval cannot distinguish a real +5pp
effect from zero.** `E0_RESULTS.md`'s P3 section notes the minimum detectable effect is ~3
episodes, but the docs still read as though the negative result is settled. A reviewer will press
here. Treat "no adapter helps" as *not demonstrated at adequate power*, not as proven.

**RESOLVED (2026-08-26).** N=100 paired re-run (`E0_RESULTS.md` top section) — CI roughly halved,
point estimate near zero, McNemar's discordant pairs nearly symmetric. The null is now
well-powered, not just unrejected. Keep this section for the historical record of why the re-run
was needed, but cite the N=100 number, not this N=20 one, going forward.

### 7.4 Three variables that have NEVER been varied

| # | Variable | Status |
|---|---|---|
| 1 | **Training-set size** | Every chart ever trained in this project used **20 trajectories**. Never varied. `ln_act`'s train→eval gap is 4.2×; `full`'s was catastrophic. `E0_RESULTS.md` itself scopes `full`'s failure as "confounded with training-set size" — that confound was never resolved. |
| 2 | **Statistical power** | See §7.3. N=20 throughout. |
| 3 | **Closed-loop evaluation** | Every planning number uses `nas=6` = **one replan**, i.e. 30 raw steps planned open-loop from t=0. The `nas=1` diagnostic was launched and **cancelled on cost** (~4h52m for 20 eps), so explanation B (horizon compounding) was never tested. `nas=2`/`nas=3` are the affordable middle (~2-3 CEM searches/episode). |

### 7.5 Budget: the constraint that shaped earlier planning was WRONG

Much of the sequencing advice in this project's chat history was built on a **$12.47** balance
read off a screenshot. The user corrected this: **~$90 of GPU credit (~112 GPU-h at L4
$0.80/h)**. Several options previously declined as unaffordable are now cheap:

| Experiment | Est. cost |
|---|---:|
| Training-set-size sweep (`ln_act` × R2 at 20/100/200 trajectories) | ~$2 |
| ~~High-power decisive eval (baseline vs best chart, **N=100** paired)~~ | ~$7 — **done, 2026-08-26** |
| `nas=2` closed-loop arm, 2 arms × 20 eps | ~$4 |
| E4 smoke (find `atlas_step()` crashes before committing 21 GPU-h) | ~$1 |
| **E3/E4 full stream** (7 arms × 6 segments × 20 ep × 3 seeds) | **~$17** |
| E1 proper | ~$7 |

**Time, not money, is now the binding constraint** (deadline 29 Aug AoE). Highest value, in
order: (1) the training-size sweep, since it attacks the one confound `E0_RESULTS.md` itself
flags as unresolved; (2) the N=100 re-run, which converts an underpowered null into a real result
either way; (3) E4, which *is* the "does continual learning help?" experiment.

### 7.6 E3's chart-generation path is now DEMONSTRATED, not merely written

`E3_E4_IMPLEMENTATION_PLAN.md` treats expansion as unexercised. It no longer is. E2's `q=1`
Cell B diagnostic (`atlas_out/e2_R2_cellB_q1/`) **committed 3 charts** through the real path:
`Expander.record()` → `library.clone_from()` → `_fit_candidate()` → `library.add()` →
`"committed"`. First time in this project expansion has been observed to create and commit a
chart end to end.

It also produced the τ-exceedance rates that explain why nothing commits at the pre-registered
`q=3`: UMF exceeds τ=0.5 in **0.0%** of appearance-only chunks versus **15.7%** under a real
dynamics shift, and three *consecutive* strikes at 15.7% is ~0.4%. That de-risks E3 and gives its
arms a predicted behaviour to check against.

### 7.7 Small things worth not re-deriving

- **`lora4`'s trainable capacity is 118,176**, not `Chart.n_params()`'s 10,292,640 (also in §4;
  repeated because `atlas_out/e0_v6_R1/results.json` still ships the wrong number on disk).
- **The UMF-locality hypothesis was tested and REJECTED.** Restricting UMF to the top-k
  most-moving tokens made the UMF-vs-success inversion *worse*, not better (the `ln_act`/`lora4`
  gap widens from 0.007 global to 0.037 at top-16, still ranked backwards from planning success).
  Spatial averaging over background is **not** why UMF discriminates poorly at fine grain. Do not
  re-propose it. Numbers: `atlas_out/umf_locality.json`.
- ~~**E2's biggest internal limitation** is that `route()` is called with `current_idx=0` on every
  decision~~ — **FIXED (2026-08-26).** `route()` now carries each router's own previous selection
  forward, reset per (cell, condition, seed) sequence. Verified via the 3-chart confusion-matrix
  run in `E2_RESULTS.md`'s top section; the original 2×2 cell numbers elsewhere in that file are
  still the pre-fix `current_idx=0` ones (not yet re-run, cheap to if the exact numbers matter).
