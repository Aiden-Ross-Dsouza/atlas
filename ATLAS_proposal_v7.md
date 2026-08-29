# ATLAS: Measure Fitness, Don't Infer the Regime
### Routing persistent adapters for continual JEPA world models

**Research question:** *can the same self-supervised signal AdaJEPA uses to* adapt *a world model also be used to* select *among persistent adaptations, and to decide when a new one is warranted?*

*NeurIPS 2026 Workshop on Continual World Models — Idea Track (2–4 pages, non-archival). Deadline 29 Aug 2026 AoE. ~11 days.*

---

## Revision notes (v7): what was cut, and why

v6 specified an experiment suite that would not fit either 11 days or 4 pages. The scientific claim needs far less. **The method is unchanged; only the empirical plan is.**

| Cut | Reason |
|---|---|
| E0 from 6 adapters × 3 regimes → **3 adapters × 2 regimes** | A six-way search is a design-space paper, not a sanity check. Two regimes (not one) because gain-like and contact-like shifts may need different capacity |
| **Goal-progress routing dropped** | It depends on the control objective, so it confounds model quality with planning. Not a competitor to a latent predictive-fitness score |
| **S1 three-regime stream dropped** | S2 (A,B,A,B,A,B) already tests adaptation, persistence, expansion, recall and forgetting |
| **PointMaze dropped** | Optional throughout; MuJoCo 2.1 + `mujoco-py` is the single largest setup hazard |
| **Wall dropped from the main plan** | Push-T alone carries all four RQs; Wall is a robustness extra |
| **Hyperparameter sweeps** (`τ`,`q`,`m`,`n_probe`) reduced to **one small sensitivity run** | A day of sweeping buys nothing an Idea Track reviewer wants |
| **S-obs, non-prequential, capacity-matched AdaJEPA → supplementary** | Each defends a specific question; none carries a headline claim |
| **Cross-policy matrix → supplementary diagnostic** | It answers the sharpest objection to the method, so it must exist — but as a half-day heatmap, not a core experiment |
| **Promoted:** expansion ablation from a sub-table to **its own named result (RQ3)** | C2 is a headline contribution. A reviewer will ask "why not just add a chart whenever loss rises?" — that deserves a dedicated answer |
| **Restructured:** baselines → a **monotone ablation ladder** | Each rung adds exactly one mechanism, so one table attributes the gain. Strictly better than a list of loosely related baselines |

---

## 1. Summary

A world model that adapts during deployment needs somewhere to put what it learns. Modular continual learning supplies a library of small modules — but something must decide *which* module to use, with no task labels and no rewards.

Recent modular methods for foundation models typically **infer the situation from representations and look up the matching module**: DPCore matches source feature statistics, CLARE uses layer-wise feature similarity, WorMI matches prototypes, Dynamic TMoE and ShiftEx spawn experts on MMD-detected drift. This is the natural design for classifiers, which never learn whether they were right. Notably, CN-DPM — the canonical expansion-based task-free method — closes by naming **expert-selection accuracy as the main bottleneck of the approach**.

A world model has an extra signal. **AdaJEPA (Jun 2026) established that the latent prediction error of an executed transition is a useful self-supervised signal for *adapting* a JEPA world model at test time.** ATLAS asks whether that signal can be elevated:

$$\text{adaptation signal (AdaJEPA)} \longrightarrow \text{routing signal} \longrightarrow \text{expansion criterion}$$

**ATLAS** places a library of small **charts** (adapters) on a permanently frozen JEPA world model and runs three rules over one normalised quantity — the fraction of observed latent motion a chart fails to explain:

| Rule | What it does | Alternative to |
|---|---|---|
| **SELECT** | pick the chart with lowest unexplained motion on a fresh chunk | similarity routing (DPCore, CLARE, WorMI) |
| **REFINE** | one SGD step on the selected chart, *after* scoring | — *(identical to AdaJEPA, held as a control)* |
| **EXPAND** | commit a new chart only after *verifying* on unseen data that it closes the deficit | detection-triggered expansion (Dynamic TMoE, ShiftEx, MBCD, DEN) |

Charts are disjoint parameter sets, so updating one cannot alter another's parameters. ~40–220 KB per chart, no gradients through the vision backbone, no reward, no task ID, no growing exemplar store.

---

## 2. Contributions

### **Primary — C1: predictive-fitness routing for persistent JEPA adapters**
Select among persistent adaptation modules by their **measured** action-conditioned rollout error on transitions none has trained on, rather than by similarity to a stored fingerprint.

*New:* to our knowledge no module-library method for frozen visual foundation models routes this way. *Not new, and we say so up front:* prediction-error model-bank switching is well established — Narendra et al.'s multiple-model adaptive control (1992–2003) and MBCD (AAMAS 2021) both do it in low-dimensional state spaces with explicit likelihoods. **We claim no novelty for the mechanism.** The open question is whether it transfers to a frozen high-dimensional visual latent space with no calibrated likelihood, a learned rather than physical geometry, and where JEPA-WM explicitly cautions that accurate unrolling does not imply planning success.

### **Secondary — C2: verification-gated expansion**
Commit a new module only after **demonstrating on future unseen data** that it closes the deficit, rather than instantiating one when a shift statistic fires.

Existing triggers are shift-based (MMD: Dynamic TMoE, ShiftEx), change-detection-based (MCUSUM: MBCD), similarity-based (CLARE), or training-loss-based (DEN, CN-DPM). To the best of our knowledge **none verifies on future unseen data that an added module improves prediction before committing it.** Stated as a hypothesis (RQ3), not an established result.

*Why it earns its place:* a detect-then-spawn chart is untrained, so under honest prequential scoring it can never win the argmin — earlier designs needed a forced-execution probation period that cost task performance. A verified chart arrives already fitted and wins on merit, so **probation is deleted**. It also makes expansion selective under noise, though the hard cap — not the probe — bounds library size.

*Closest precedent, cited:* DEN (ICLR 2018) performs selective retraining and expands on a training-loss threshold. Ours is online and task-free, and verifies on **held-out future** data, testing whether the fix *generalises*.

### **Supporting — C3: UMF, a normalised predictive-fitness score**
The same L2 latent prediction error DINO-WM, JEPA-WM and AdaJEPA already use, normalised by the latent motion actually observed. `0` = perfect; `≈1` = no better than predicting stasis. We claim no novelty for the algebra (this is the world-model analogue of `R²`/NMSE); the contributions are the choice of *predicted stasis* as the null model and **validating it against planning success on public checkpoints**, which addresses JEPA-WM's open appendix question.

### **Supporting — C4: the Deployment Stream protocol**
Plasticity / retention / recall; the parameter-vs-system-level retention distinction; and a cross-policy competence diagnostic. AdaJEPA names cross-episode continual adaptation as future work.

**Unifying statement:** *C1 and C2 apply one idea to the two decisions a module library must make. Selection asks "which module fits?" — the field infers the regime; we evaluate fitness directly. Expansion asks "do I need a new module?" — the field detects a shift; we verify that a new module helps.*

---

## 3. Literature review

### 3.1 Substrate

| Work | Date | Contribution |
|---|---|---|
| **DINO-WM** (Zhou et al.) | 2024/25 | Latent dynamics predictor on **frozen** DINOv2 patch tokens; CEM planning with L2 latent goal cost |
| **JEPA-WM** (Terver, Yang, Ponce, Bardes, LeCun) | Dec 2025 | Design-space study. **arXiv 2512.24497** — public code, data, checkpoints. **Our frozen backbone** |
| **V-JEPA-2-AC** (Assran et al.) | 2025 | Action-conditioned video JEPA |
| **PLDM** (Sobal et al.) | 2025 | JEPA latent dynamics generalise from suboptimal offline data |
| **Temporal Straightening** (Wang et al.) | ICML 2026 | Latent trajectories locally straight; compress to ~8 dims |
| **RC-aux** ("Predictive but Not Plannable") | 2026 | Model queries should match the planner's — motivates horizon-matched scoring |

### 3.2 The caveat the paper is built around

JEPA-WM reports that models which unroll many actions faithfully do not thereby succeed at planning, and asks in an appendix whether any proxy for success rate exists — while still adopting embedding-space unrolling error as its primary planner-independent metric. **We take both halves seriously:** the metric is useful (so we use it, normalised — C3); its correlation with success is imperfect (so RQ1 *measures* whether it supports selection).

### 3.3 Adaptation of world models

| Work | Date | Establishes | Does not address |
|---|---|---|---|
| **ReDRAW** (Lanier et al.) | Apr 2025 | Latent-state dynamics residuals adapt a world model to a new domain | Offline target-domain collection; one fixed target |
| **AdaWM** (Wang et al.) | Jan 2025 | Selective LoRA finetuning of the misaligned component | Single adaptation event; no library |
| **WorMI** (Yoo et al.) | Sep 2025 | Retrieving and **fusing** domain-specific world models works | Prototype-similarity routing; models pre-trained per domain |
| **AdaJEPA** (Wang, Bounou, LeCun, Ren) | **Jun 2026** | **The executed transition's latent prediction error is a useful self-supervised adaptation signal inside MPC.** One SGD step, 5-transition buffer, plan–act–adapt–replan | Each evaluation episode adapts independently from pretrained weights, so **persistent cross-episode retention and recall are outside its scope** |

AdaJEPA's closing sentence names our opening: *"A natural next step is to combine lightweight test-time adaptation with continual and active learning to expand the world model's coverage over time."*

### 3.4 Module libraries: how routing and expansion are currently done

| Work | Venue | Routing signal | Expansion trigger |
|---|---|---|---|
| DPCore | ICML 2025 | source feature statistics | distance to existing prompts |
| CLARE | arXiv 2601.09512, 2026 | layer-wise feature similarity + autoencoder | feature similarity |
| WorMI | 2025 | prototype similarity | — (pre-trained set) |
| Dynamic TMoE | arXiv 2605.20678, 2026 | memory router | MMD drift detection |
| ShiftEx | arXiv 2506.18789, 2025 | expert matching | MMD covariate-shift detection |
| CN-DPM | ICLR 2020 | generative likelihood | DP prior; *"expert selection … the main bottleneck"* |
| MBCD | AAMAS 2021 | **prediction likelihood** | MCUSUM change detection |
| DEN | ICLR 2018 | — | selective retraining, then **training-loss** threshold |
| **ATLAS** | — | **measured rollout fitness** | **verified fixability on unseen data** |

Most route on similarity to inputs — **MBCD is the notable exception and uses a predictive signal**, which is why we position ATLAS as bringing that family of signal into the modern frozen-visual-adapter setting rather than introducing it. Expansion triggers vary (shift statistics, similarity, training loss), but **none verifies on future unseen data that the added module improves prediction before committing it.**

### 3.5 Precedents cited defensively

Narendra et al. (1992–2003) multiple-model adaptive control — model bank, switching on recent prediction error, fixed models retained alongside adaptive ones; assumes low-dimensional plants with physically meaningful residuals. MBCD (AAMAS 2021) — MCUSUM on prediction likelihood, more principled than our strike counter, but needs a likelihood a deterministic JEPA predictor lacks. Herbster & Warmuth (1998) Fixed-Share — the switching-regret theory; named as the principled extension.

### 3.6 Relationship to AdaJEPA and JEPA-WM: complementary, not contradictory

**AdaJEPA:** observe prediction error → adapt the current model → plan better.
**ATLAS:** observe prediction error → *which existing adaptation fits?* → select → adapt it → *is a new one warranted?*

Both rest on the premise AdaJEPA established and we do not re-litigate. **AdaJEPA is the foundation, not the foil.**

**What ATLAS inherits unchanged** — we change one thing and hold everything else fixed:

| Component | Kept exactly as in |
|---|---|
| Frozen DINOv2 encoder + ViT predictor | JEPA-WM / DINO-WM (public checkpoints, unmodified) |
| **L2 distance in latent space** as the prediction metric | DINO-WM, JEPA-WM, AdaJEPA — UMF's numerator *is* this quantity; we add a denominator, not a different distance |
| CEM planner and configuration | JEPA-WM's best-performing setup |
| Self-supervised latent prediction loss, optimiser, lr, 1 step, 5-transition buffer | AdaJEPA |
| Multi-step embedding error as planner-independent quality | JEPA-WM (we normalise it, not replace it) |
| Adapters as the adaptation surface | CLARE, AdaWM |
| Expansion structure: trigger → instantiate → prune | Dynamic TMoE, ShiftEx (we change only the *commit* criterion) |
| Clone / warm-start on expansion | CLARE |

**What ATLAS adds:** normalisation of the existing metric (C3); use of it for *selection* among persistent modules (C1); *verification* before committing a module (C2); a cross-episode persistence protocol (C4).

---

## 4. Problem formulation

A sequence of episodes, each a goal-conditioned MPC rollout in an environment from a **non-stationary, unlabelled** sequence of regimes. No rewards, demonstrations, regime labels, boundary signals, or task IDs — only the agent's own transitions, from its own plans, **self-supervised from the consequences of its own interactions** (which still costs environment interaction).

- **P1 Plasticity** — improve within the current regime. *(AdaJEPA demonstrates this.)*
- **P2 Retention** — do not degrade on visited regimes.
- **P3 Recall** — handle a revisited regime immediately, without re-paying the adaptation cost.

**Primary setting: dynamics shift with matched appearance** (mass, damping, friction). A controlled probe, not a realism claim — real shifts are multimodal, since a heavier object also looks different. We isolate it because the frozen encoder stays in-distribution so the target `E_φ(o)` is trustworthy, and because it is where observation-similarity routing has least signal. Because it is artificial, **S-dyn (the dynamics-fingerprint router) is our primary baseline** and the combined-shift cell is reported as the realistic condition.

---

## 5. Four research questions

| | Question | Experiment | If it fails |
|---|---|---|---|
| **RQ0** | Can a lightweight adapter absorb a physical dynamics shift at all? | **E0** capacity | Report which adapter class is needed — a useful design finding |
| **RQ1** | Does UMF identify the competent chart, and better than dynamics-fingerprint routing? | **E1** routing | Prediction-error routing does not transfer to frozen visual latent spaces — a citable negative result on JEPA-WM's open question |
| **RQ2** | Is UMF responding to *dynamics* rather than *appearance*? | **E2** 2×2 | If it tracks appearance, the "dynamics competence" reading is wrong — important to know |
| **RQ3** | Does verification-gated expansion beat detect-and-spawn? | **E3** expansion ladder | Detection suffices → C2 unnecessary, reported honestly |
| **RQ4** | Does a persistent chart library deliver recall on A→B→A? | **E4** stream | Behavioural retention fails despite parameter isolation → routing is the bottleneck, which E1 quantifies |

**E0 and E1 run first (days 3–5) and gate the project.** Both need only offline charts and an evaluation harness — no ATLAS loop. If RQ1 fails we pivot with six days remaining.

---

## 6. Method

*(Unchanged from v6; §6.1–6.10 of that document stand. Condensed here.)*

```
FROZEN FOREVER:  DINOv2 encoder → JEPA predictor → CEM planner
CHART:           small adapter on the PREDICTOR only, identity-initialised
LIBRARY:         C = {c₀, c₁, …};  c₀ = identity chart, never refined, always competes

EVERY REPLAN, on the newest executed chunk Q (no chart has trained on it):
  1. SCORE   UMF(c; Q) for all c        — skip Q if uninformative
  2. SELECT  c* = argmin UMF, hysteresis margin m
  3. EXPAND  if UMF(c*) > τ for q consecutive informative checks:
                fit a candidate on the DEFICIT chunks;
                commit only if it beats both τ and c* on the NEXT unseen chunk
  4. EXECUTE plan with c* (CEM unchanged)
  5. REFINE  1 SGD step on c*, AdaJEPA's exact loss — strictly AFTER scoring
```

**The score.** `UMF(c;Q) = Σ_k ‖ẑ^c_k − z_k‖² / Σ_k ‖z_k − z_0‖²`, open-loop unroll, uniform weights, computed only on chunks whose observed displacement exceeds the 10th percentile of training displacement. Whether one fixed threshold transfers across regimes is empirical, tested in E1/E4.

**Prequential order.** Refining on `W` then scoring on `W` makes that chart win by construction. Scoring strictly precedes refinement; a consequence is that every score is already a post-refinement verdict on fresh data.

**Expansion.** One library-level strike counter (the library is inadequate iff its *best* chart is). The probe fits **only on the deficit chunks** — a plain recent-window would span regimes and yield a short-term average model rather than a specialist. Verification is on the next unseen chunk: held out from the candidate's training, though still policy-dependent.

**Retention.** Parameter level guaranteed by disjointness; **system level not guaranteed** — routing can still misfire, measured as the oracle-ID routing gap.

---

## 7. Experiments

Five experiments, four of which map one-to-one onto the RQs. All on **Push-T** with `dino_wm_pusht`.

### E0 — adapter capacity (RQ0). *Days 3–4, ~3 GPU-h*
Three adapter classes — **LN(+action), LoRA r=4, full predictor** — fine-tuned offline on **two** regimes (R1 light block, R2 high damping). Two regimes, not one, because gain-like and contact-like shifts may need different capacity.
**Pre-registered rule:** use the smallest class reaching ≥ 90 % of full-predictor gain in both UMF reduction and success.

### E1 — fitness routing (RQ1). *Days 4–5, ~4 GPU-h.* **THE GATE**
Charts from E0, fixed. Per episode: 2 warmup replans under `c₀`, score every chart, select, plan the rest. **Identical seeds across all selectors.**

Routers: **UMF (ours) · one-step `e₁` · S-dyn · Random · Oracle-ID**. *(Goal-progress dropped: it depends on the control objective and confounds model quality with planning. S-obs → supplementary.)*

**Pre-registered pass criterion:** normalised recovery `(SR_UMF − SR_rand)/(SR_oracle − SR_rand) ≥ 0.8`, reported only when the denominator ≥ 10 pp.

### E2 — appearance vs dynamics (RQ2). *Day 9, ~6 GPU-h*

| | Dynamics same | Dynamics differ |
|---|---|---|
| **Appearance same** | A control | **B — the decisive cell** |
| **Appearance differs** | **C — must not over-expand** | D realistic |

Routers: UMF vs S-dyn (+ S-obs if time). The two results that matter: in **B**, UMF routes correctly where appearance carries no signal; in **C**, ATLAS does **not** spawn a chart the world did not need.

### E3 — expansion ladder (RQ3). *Runs inside the E4 stream; reported separately*
Three arms differing only in the expansion rule:

| Arm | Rule |
|---|---|
| **Fixed library** | no expansion; charts from E0 only |
| **Detect-only** | persistent deficit → immediately commit a chart (the field's convention) |
| **ATLAS** | persistent deficit → candidate → unseen-data verification → commit or reject |

Reported: success, charts committed vs. true regime count, probes fired, probes rejected, recall. This answers *"why not just add a chart whenever the loss rises?"* directly.

### E4 — continual stream (RQ4). *Days 7–8, ~21 GPU-h*
**S2 only:** `A,B,A,B,A,B`, 20 episodes per segment × 3 seeds, paired seeding. This single stream exercises adaptation, persistence, expansion, recall and forgetting. *(S1 three-regime stream dropped.)*

**The ablation ladder** — each rung adds exactly one mechanism, all sharing the same adaptation surface, loss and optimiser:

| Arm | Adapts | Persists | Library + routing | Expands | Verifies |
|---|---|---|---|---|---|
| Frozen | | | | | |
| AdaJEPA | ✓ | | | | |
| Persistent-AdaJEPA *(ours)* | ✓ | ✓ | | | |
| ATLAS-fixed-library | ✓ | ✓ | ✓ | | |
| ATLAS-detect-only | ✓ | ✓ | ✓ | ✓ | |
| **ATLAS** | ✓ | ✓ | ✓ | ✓ | ✓ |
| Oracle-ID | — | — | oracle | — | — |

This is the paper's central table: it attributes the gain to a specific mechanism rather than to "our system." **Routing-signal variants (Random, S-dyn) are evaluated in E1/E2 and need no separate stream runs.**

### E5 — cross-policy diagnostic. *Day 5, ~3 GPU-h, supplementary*
`M[i,j]` = chart `i`'s UMF on chunks generated by chart `j`'s plans. Answers the sharpest objection — that charts are judged on whichever chart is currently selected — with a heatmap and per-column argmin accuracy. Half a day, supplementary figure.

### Supplementary if time
S-obs router · non-prequential ablation · capacity-matched AdaJEPA · one small sensitivity run over `τ` and `q` · Wall environment · PointMaze.

---

## 8. Results, sized for 4 pages

**Body — 2 figures, 2 tables.**

| ID | Content |
|---|---|
| **F1** | **Money plot.** Success (rolling mean) vs. episode across S2; regime boundaries dashed; ★ chart committed, ○ probe fired but rejected. Ladder arms + Oracle-ID |
| **F2** | Two panels: **(a)** 2×2 routing accuracy, UMF vs S-dyn; **(b)** library size vs. episode for the three expansion arms, with true regime count marked |
| **T1** | **E1 routing.** Router × {SR, routing accuracy, oracle gap, normalised recovery}, paired bootstrap CIs |
| **T2** | **E4 ladder + recall.** Arm × {SR first visit, SR final revisit, paired Δ [CI], McNemar *p*, charts committed, probes rejected} |

**Supplementary:** E0 capacity table · E5 cross-policy heatmap · UMF traces per chart · UMF-vs-success scatter (C3 validation) · sensitivity run.

---

## 9. Schedule and risks

| Day | Deliverable | Gate |
|---|---|---|
| 1–2 | Fork `jepa-wms`; env; checkpoints; reproduce frozen Push-T; **profile one episode**; dump predictor parameter names | success ±3 pp; compute budget computed |
| 3 | Regime wrappers; paired-seed harness; UMF + gating | **G4 regimes real, G5 pairing** |
| 3–4 | **E0** capacity | adapter class chosen by data |
| 4 | Chart / library code | **G1 bit-identical to frozen** |
| 4–5 | **E1** routing | **T1 → GO / PIVOT** |
| 5 | **E5** cross-policy (supplementary) | heatmap |
| 6 | Router, expansion, loop; ladder arms wired | **G2 prequential, G3 probe fires *and* discriminates** |
| 7–8 | **E4 + E3** stream, 7 arms × 3 seeds | F1, F2b, T2 |
| 9 | **E2** 2×2 | F2a |
| 10 | Sensitivity run; figures; polish | assets final |
| 11 | Finish 4 pages | submitted |

**Write §1–3 from day 6 while streams run** — none of it depends on results.

**Scope-cut ladder:** sensitivity run → E5 → E2 cell D → reduce to 15 episodes/segment → drop the fixed-library arm. **E0 + E1 + E2 + E3 + E4 with T1/T2/F1/F2 is the complete paper.**

| Risk | Detect | Response |
|---|---|---|
| **RQ1 fails** | Day 5 | Pivot to the negative result; T1 already carries the `e₁`-vs-UMF comparison. Six days remain |
| Predictor uses AdaLN, no free LN affine | Day-2 parameter dump | Target the AdaLN conditioning MLP; DINO-WM checkpoints (plain LN) are primary for this reason |
| Physics edits don't change dynamics | **G4**, day 3 | Different parameter; verify by rendering. Never proceed with a fake shift |
| Episodes too slow | Day-2 profile | Cut CEM opt steps uniformly across all arms |
| Probe never/always fires | **G3**, day 6 | Sweep `τ`, `q`; if degenerate, report the sensitivity as the finding |
| S-dyn ties UMF | E1/E2 | Report honestly; finding becomes *route by dynamics, not appearance* |
| Underpowered | Always | Paired seeds + McNemar; 20 ep × 6 segments × 3 seeds = 360 paired episodes per arm |

---

## 10. References

**Substrate.** Terver et al., *What Drives Success in Physical Planning with JEPA World Models?*, arXiv:2512.24497, 2025 · Zhou et al., *DINO-WM*, arXiv:2411.04983 · Assran et al., *V-JEPA 2*, arXiv:2506.09985, 2025 · Sobal et al., *PLDM*, 2025 · Wang et al., *Temporal Straightening*, ICML 2026 · *RC-aux*, arXiv:2605.07278, 2026.

**World-model adaptation.** Wang, Bounou, LeCun, Ren, *AdaJEPA*, arXiv:2606.32026, 2026 *(foundation and primary baseline)* · Lanier et al., *ReDRAW*, arXiv:2504.02252, 2025 · Wang et al., *AdaWM*, 2025 · Huang et al., *AdaPower*, 2025.

**Module libraries.** Zhang et al., *DPCore*, ICML 2025 · Römer, Zhang, Schoellig, *CLARE*, arXiv:2601.09512, 2026 · Yoo et al., *WorMI*, 2025 · *Dynamic TMoE*, arXiv:2605.20678, 2026 · *ShiftEx*, arXiv:2506.18789, 2025 · Lee et al., *CN-DPM*, ICLR 2020 · Yoon et al., *DEN*, ICLR 2018 · Wang et al., *Tent*, ICLR 2021 · Wang et al., *CoTTA*, CVPR 2022.

**Prediction-error model banks.** Narendra & Balakrishnan, IEEE TAC 1997 · Narendra et al., IJACSP 2003 · Alegre et al., *MBCD*, AAMAS 2021 · *Infinite Mixture of GPs*, arXiv:2006.11441, 2020.

**Methodology.** Herbster & Warmuth, *Tracking the Best Expert*, ML 32:151–178, 1998 · Kirkpatrick et al., *EWC*, PNAS 2017 · Dawid, JRSS-A 1984 *(prequential)* · McNemar, Psychometrika 1947.
