"""
scripts/smoke_gates.py — G1–G6 correctness gates (CI + day-by-day sanity checks).

Run all gates with:
    python scripts/smoke_gates.py --all

Or individual gates:
    python scripts/smoke_gates.py --gate G1
    python scripts/smoke_gates.py --gate G4 --regime R1

Gates:
  G1  identity     Library {c₀} only → trajectory bit-identical to frozen; UMF(c₀) << 1
  G2  prequential  Over-refine X on W; score on W' → X must not automatically win
  G3a probe fires  New regime → probe passes, chart commits
  G3b probe discr. Noise → probe rejects, nothing commits
  G4  regimes real 20 rollouts per regime differ statistically (mean ± std on latents)
  G5  pairing      Two arms, same seeds → identical initial states
  G6  denominator  Static chunk → UMF returns None (no score, no strike)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import atlas
from atlas.chart import Chart
from atlas.library import Library
from atlas.score import umf as compute_umf
from atlas.expand import Expander, ExpansionConfig

# Same vendored-checkout sys.path setup as run_e0_planning.py:74-82 -- needed
# by gate_g5's real PushTEnv construction (FIX_SPEC C2). Raw ATLAS_HOME (not
# atlas.ATLAS_HOME, which .resolve()'s and breaks under Modal volume mounts).
_atlas_home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
_HUB_PATH = str(Path(_atlas_home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
if _HUB_PATH not in sys.path:
    sys.path.insert(0, _HUB_PATH)


def _make_synthetic_proprio_ctxt(wm, grid: int, device) -> torch.Tensor | None:
    """Build a correctly-SHAPED (not physically meaningful) proprio_ctxt
    [1, 1, P_tok, D_p] for gates that use synthetic random encoder_output --
    this checkpoint's forward_pred needs a shape-correct proprio tensor
    concatenated into the token channel width even for synthetic data (see
    E0_IMPLEMENTATION_PLAN.md T1's finding: proprio=None is a channel-width
    mismatch, not a graceful no-proprio path). None only if this checkpoint
    genuinely has no proprio_encoder."""
    if getattr(wm, "proprio_encoder", None) is None:
        return None
    prop_dim = wm.proprio_encoder.embed_dim
    proprio_ctxt = torch.randn(1, 1, 1, prop_dim, device=device)
    if getattr(wm, "proprio_encoding", None) == "feature":
        proprio_ctxt = proprio_ctxt.repeat(1, 1, grid * grid, 1)
    return proprio_ctxt


def gate_g1(wm, wrapper) -> None:
    """G1: an identity-initialised chart must change NOTHING.

    Rewritten (the previous version tested nothing it claimed to): it built a
    Chart, never applied it, never called the model, and compared two env
    rollouts driven by SEPARATE unseeded action_space.sample() draws -- so it
    could only ever have compared different action sequences against each
    other. It also used gymnasium's reset(seed=)/5-tuple step() against this
    legacy-gym env.

    What identity actually means here, and what is checked below:
      1. apply_() of a freshly-cloned chart (LoRA B=0, LN at pretrained values)
         leaves the predictor's OUTPUT bit-identical -- torch.equal, not
         allclose. This is CLAUDE.md 1.4's non-negotiable.
      2. restore_() puts every predictor tensor back bit-identically, so charts
         cannot contaminate each other (the leak that invalidated E0's first
         chart set).
    Runs headless on synthetic latents -- no env needed, so it executes under
    --all instead of being permanently skipped.
    """
    print("G1: identity chart bit-identity check...", end=" ")
    predictor = wm.predictor
    grid = wm.grid_size
    D = 384
    N = grid * grid
    device = next(predictor.parameters()).device

    pristine = {k: v.detach().clone() for k, v in predictor.state_dict().items()}

    # Fixed synthetic context + action: identical inputs across all three
    # forwards below, so any output difference is attributable to the chart.
    torch.manual_seed(0)
    z = torch.randn(1, 1, 1, grid, grid, D, device=device)
    a_raw = torch.randn(1, 1, wm.action_dim, device=device)

    def _forward():
        with torch.no_grad():
            act_feats = wm.encode_act(a_raw)
            prop_feats = None
            if getattr(wm, "proprio_encoder", None) is not None:
                prop_dim = wm.proprio_encoder.embed_dim
                prop_feats = torch.zeros(1, 1, 1, prop_dim, device=device)
                if getattr(wm, "proprio_encoding", None) == "feature":
                    prop_feats = prop_feats.repeat(1, 1, grid * grid, 1)
            pred_vis, _, _ = wm.forward_pred(z, act_feats, prop_feats)
        return pred_vis.reshape(N, D)

    out_frozen = _forward()

    for kind in ("ln_act", "lora4", "full"):
        c0 = Chart(predictor, kind=kind)
        c0.apply_(predictor)
        out_c0 = _forward()
        if not torch.equal(out_frozen, out_c0):
            max_abs = (out_frozen - out_c0).abs().max().item()
            c0.restore_(predictor)
            raise AssertionError(
                f"G1 FAILED: kind={kind} identity chart changed the predictor "
                f"output (max |diff| = {max_abs:.3e}, required exactly 0). "
                "A freshly-cloned chart must be a no-op -- check Chart.apply_() "
                "and the kind's identity initialisation (LoRA B=0, LN at "
                "pretrained values)."
            )
        c0.restore_(predictor)

        after = predictor.state_dict()
        for k, v0 in pristine.items():
            if not torch.equal(v0, after[k]):
                raise AssertionError(
                    f"G1 FAILED: kind={kind} restore_() did not return "
                    f"predictor tensor {k!r} to its pretrained value -- charts "
                    "would contaminate each other."
                )

        out_restored = _forward()
        if not torch.equal(out_frozen, out_restored):
            raise AssertionError(
                f"G1 FAILED: kind={kind} predictor output changed after "
                "restore_() despite an identical state_dict."
            )

        # ── FIX_SPEC.md C3: extend past identity-only charts ────────────────
        # The block above only ever tested a chart that had NEVER been
        # refined, where restore_() and apply_() are trivially identical
        # (restore_() just re-applies the same untouched _params -- see
        # FIX_SPEC.md C4). Genuinely exercise a REFINED chart: fit it for
        # real via _open_loop_rollout (the production mechanism, same as
        # gate_g2/atlas/expand.py::_fit_candidate), confirm it actually
        # moved the predictor's output away from frozen (sanity that
        # refinement did something), then confirm the NEW
        # Chart.restore_pretrained_() (C4) puts the predictor back to
        # EXACT pretrained weights and output -- what plain restore_()
        # cannot do for ln_act/full (see chart.py's restore_pretrained_
        # docstring).
        from atlas.score import _open_loop_rollout, _make_z_ctxt
        import torch.optim as optim
        T = 3
        act_dim = wm.action_dim
        proprio_ctxt = _make_synthetic_proprio_ctxt(wm, grid, device)
        torch.manual_seed(7)
        enc = torch.randn(T + 1, N, D, device=device)
        actions = torch.randn(T, act_dim, device=device)

        cr = Chart(predictor, kind=kind)
        cr.apply_(predictor)
        if kind == "lora4":
            params = [p for n, p in predictor.named_parameters()
                       if "lora_A" in n or "lora_B" in n]
        else:
            params = [p for n, p in predictor.named_parameters() if n in cr._param_names]
        opt = optim.Adam(params, lr=1e-2)
        z_ctxt = _make_z_ctxt(wrapper, enc[0], proprio_ctxt)
        for _ in range(20):
            opt.zero_grad()
            z_hat = _open_loop_rollout(wrapper, z_ctxt, actions)
            loss = (z_hat - enc[1:]).pow(2).mean()
            loss.backward()
            opt.step()
        cr.update_from_predictor_(predictor)

        out_refined = _forward()
        if torch.equal(out_frozen, out_refined):
            raise AssertionError(
                f"G1 FAILED: kind={kind} chart refined for 20 real gradient "
                "steps produced a bit-identical predictor output to frozen "
                "-- refinement did not do anything measurable, so the "
                "restore_pretrained_() check below would be vacuous."
            )

        cr.restore_pretrained_(predictor, pristine)
        after_refined_restore = predictor.state_dict()
        for k, v0 in pristine.items():
            if not torch.equal(v0, after_refined_restore[k]):
                raise AssertionError(
                    f"G1 FAILED: kind={kind} restore_pretrained_() did not "
                    f"return REFINED predictor tensor {k!r} to its "
                    "pretrained value."
                )
        out_after_pretrained_restore = _forward()
        if not torch.equal(out_frozen, out_after_pretrained_restore):
            raise AssertionError(
                f"G1 FAILED: kind={kind} predictor output differs from "
                "frozen after restore_pretrained_() on a REFINED chart, "
                "despite an identical state_dict."
            )

    print("PASSED")


def _g2_make_regime_chunk(wm, wrapper, c0, regime, predictor, grid, D, N, T,
                           act_dim, device):
    """One structured, learnable chunk (same construction as gate_g3a's
    make_regime_chunk): roll out under a fixed perturbation of the
    predictor's own weights, then restore c0 (baseline) before returning, so
    the predictor is always left in a known state between calls."""
    regime.apply_(predictor)
    z0 = torch.randn(N, D, device=device)
    actions = torch.randn(T, act_dim, device=device)
    z = z0
    frames = [z0]
    for t in range(T):
        z = _predict_one_step(wm, z, actions[t])
        frames.append(z)
    c0.apply_(predictor)
    return torch.stack(frames, dim=0), actions


def gate_g2(wm, wrapper) -> None:
    """G2: over-refine chart X on window W; the score that decides whether X
    "wins" must come from the NEXT, genuinely held-out window W' -- never
    from W itself. Catches scoring/refinement leakage (CLAUDE.md Sec1.6).

    Rewritten (the previous version built W/W' from i.i.d. torch.randn --
    structureless, nothing to over-refine on -- over-refined via a
    hand-rolled forward_pred loop that bypassed _open_loop_rollout entirely,
    computed both UMFs, and asserted nothing: `if ... is not None and ... is
    not None: pass`. It printed PASSED unconditionally.)

    This version:
      1. Builds W and W' from two INDEPENDENT structured "regimes" (small,
         learnable perturbations of the predictor's own weights, same
         mechanism as gate_g3a) -- so W' is a genuinely different, unrelated
         distribution from W, not just a second i.i.d. draw.
      2. Over-refines cx on W ONLY, for real, via _open_loop_rollout (the
         production rollout, atlas/expand.py::_fit_candidate's exact
         mechanism) run to convergence (300 Adam steps on one small chunk --
         enough to memorize W's specific instance).
      3. Computes THREE numbers: cx's UMF on its own training window W
         (`umf_cx_W` -- what a LEAKED scorer would report), cx's UMF on the
         genuinely held-out, differently-regimed W' (`umf_cx_Wprime` --
         what the correct prequential scorer reports), and baseline c0's UMF
         on W' for reference.
      4. Asserts the property a leaked scorer would violate: the genuinely
         held-out score must not be reported as at least as good as the
         leaked training-window score. After 300 steps of overfitting a
         high-capacity ("full") chart to one small chunk, umf_cx_W is driven
         near zero; a chart that trained only on W cannot legitimately have
         an equally-good or better score on the disjoint, differently-
         regimed W' -- if it does, whatever produced that W'-score is not
         actually looking at held-out data.
    """
    print("G2: prequential ordering check...", end=" ")
    predictor = wm.predictor
    c0 = Chart(predictor, kind="full")
    cx = c0.clone()

    grid = wm.grid_size
    D, T = 384, 5
    N = grid * grid
    act_dim = wm.action_dim
    device = next(predictor.parameters()).device
    proprio_ctxt = _make_synthetic_proprio_ctxt(wm, grid, device)

    # Two INDEPENDENT structured regimes -> W and W' are genuinely different
    # distributions (same mechanism as gate_g3a's REL_SCALE perturbation,
    # drawn twice, independently, so W' is not just fresh noise from the same
    # regime W was fit on).
    REL_SCALE = 0.3
    regime_W = c0.clone()
    for name, value in regime_W._params.items():
        regime_W._params[name] = value + torch.randn_like(value) * value.std() * REL_SCALE
    regime_Wp = c0.clone()
    for name, value in regime_Wp._params.items():
        regime_Wp._params[name] = value + torch.randn_like(value) * value.std() * REL_SCALE

    torch.manual_seed(2)
    W_enc, W_act = _g2_make_regime_chunk(wm, wrapper, c0, regime_W, predictor,
                                          grid, D, N, T, act_dim, device)
    Wp_enc, Wp_act = _g2_make_regime_chunk(wm, wrapper, c0, regime_Wp, predictor,
                                            grid, D, N, T, act_dim, device)

    # Over-refine cx on W ONLY, via the production rollout mechanism
    # (_open_loop_rollout / _make_z_ctxt -- exactly atlas/expand.py::
    # _fit_candidate's code path), run to convergence on a single chunk.
    from atlas.score import _open_loop_rollout, _make_z_ctxt
    import torch.optim as optim
    cx.apply_(predictor)
    params = [p for n, p in predictor.named_parameters() if n in cx._param_names]
    opt = optim.Adam(params, lr=5e-3)
    z_ctxt = _make_z_ctxt(wrapper, W_enc[0], proprio_ctxt)
    for _ in range(300):
        opt.zero_grad()
        z_hat = _open_loop_rollout(wrapper, z_ctxt, W_act)
        loss = (z_hat - W_enc[1:]).pow(2).mean()
        loss.backward()
        opt.step()
    cx.update_from_predictor_(predictor)
    cx.restore_(predictor)

    umf_c0_Wp = compute_umf(c0, wrapper, Wp_enc, Wp_act, proprio_ctxt=proprio_ctxt)
    umf_cx_Wp = compute_umf(cx, wrapper, Wp_enc, Wp_act, proprio_ctxt=proprio_ctxt)
    # Reference-only value: what a scorer that (bug) reused the training
    # window W instead of the held-out W' would report. Never fed into a
    # decision here -- computed purely so the assertion below can detect the
    # leak signature (a suspiciously good "held-out" score).
    umf_cx_W = compute_umf(cx, wrapper, W_enc, W_act, proprio_ctxt=proprio_ctxt)

    if umf_cx_Wp is None or umf_cx_W is None or umf_c0_Wp is None:
        raise AssertionError("G2 FAILED: compute_umf returned None on a "
                              "genuinely moving synthetic chunk -- motion "
                              "gate or denominator guard misfiring.")

    if umf_cx_Wp <= umf_cx_W:
        raise AssertionError(
            f"G2 FAILED (leakage signature): the over-refined chart's score "
            f"on the genuinely held-out window W' ({umf_cx_Wp:.4f}) is <= "
            f"its score on its OWN training window W ({umf_cx_W:.4f}). A "
            "chart driven to near-zero loss on one small chunk cannot "
            "legitimately score at least as well on a disjoint, "
            "differently-regimed window -- whatever produced the W' number "
            "is not looking at held-out data. This is exactly the "
            "scoring/refinement leakage G2 exists to catch."
        )
    print(f"PASSED  (umf_cx on training W={umf_cx_W:.4f} [leaked, "
          f"reference-only], umf_cx on held-out W'={umf_cx_Wp:.4f}, "
          f"umf_c0 on held-out W'={umf_c0_Wp:.4f})")


def _predict_one_step(wm, z_t: torch.Tensor, a_t_raw: torch.Tensor) -> torch.Tensor:
    """One-step latent prediction via VideoWM.forward_pred(), no grad.

    Same low-level call convention as expand.py::_fit_candidate() and
    gate_g2's _one_step_loss -- used here to construct synthetic ground-truth
    chunks for G3a/G3b, not to compute a training loss.
    z_t: [N, D] current latent. a_t_raw: [action_dim] one raw action.
    Returns: [N, D] predicted next latent.
    """
    grid = wm.grid_size
    D = z_t.shape[-1]
    N = grid * grid
    with torch.no_grad():
        z_cur = z_t.reshape(1, 1, 1, grid, grid, D)
        act_feats = wm.encode_act(a_t_raw.reshape(1, 1, -1))
        prop_feats = None
        if getattr(wm, "proprio_encoder", None) is not None:
            prop_dim = wm.proprio_encoder.embed_dim
            prop_feat = torch.zeros(1, 1, 1, prop_dim, device=z_cur.device)
            if getattr(wm, "proprio_encoding", None) == "feature":
                prop_feat = prop_feat.repeat(1, 1, grid * grid, 1)
            prop_feats = prop_feat
        pred_vis, _, _ = wm.forward_pred(z_cur, act_feats, prop_feats)
        return pred_vis.reshape(N, D)


def gate_g3a(wm, wrapper) -> None:
    """G3a: genuinely new, learnable regime shift -> probe fires and commits.

    Tests Expander's coded logic only (record()/maybe_expand() called
    directly, bypassing the separately-broken atlas_step()/route() wiring --
    see CLAUDE.md Sec 0.1). Does NOT test whether UMF-based verification would
    catch a chart that improves UMF while still hurting real CEM planning
    (see the E0 CEM-cost diagnostic finding) -- that's a different, harder,
    not-yet-gated question. compute_umf()/maybe_expand() now need the
    EncPredWM WRAPPER (torch.hub.load's return value, not .model) -- see
    E0_IMPLEMENTATION_PLAN.md T1/T2.
    """
    print("G3a: probe fires on a genuinely new regime...", end=" ")
    predictor = wm.predictor
    c0 = Chart(predictor, kind="full")
    library = Library(c0, max_size=5)
    cfg = ExpansionConfig(kind="full")
    expander = Expander(cfg)

    grid = wm.grid_size
    D, T = 384, 5
    N = grid * grid
    act_dim = wm.action_dim
    device = next(predictor.parameters()).device
    proprio_ctxt = _make_synthetic_proprio_ctxt(wm, grid, device)

    # "Regime shift" = a small perturbation to the predictor's OWN weights, not
    # an external latent-space bias (empirically, an additive bias applied
    # outside forward_pred() barely moves after 200 fitting steps -- adding it
    # inside the same parametric family the candidate can also reach makes
    # this provably fittable via gradient descent, not just hoped-for).
    # REL_SCALE=0.3 validated: robust across seeds, baseline UMF ~0.8-0.95 on
    # deficit chunks, candidate UMF ~0.03-0.10 after n_probe=20 steps.
    REL_SCALE = 0.3
    regime = c0.clone()
    for name, value in regime._params.items():
        regime._params[name] = value + torch.randn_like(value) * value.std() * REL_SCALE

    def make_regime_chunk():
        # Ground truth generated by rolling out under the SHIFTED weights,
        # then restoring baseline before returning -- c0/predictor must be
        # left in baseline state for compute_umf()/record() to score correctly.
        regime.apply_(predictor)
        z0 = torch.randn(N, D, device=device)
        actions = torch.randn(T, act_dim, device=device)
        z = z0
        frames = [z0]
        for t in range(T):
            z = _predict_one_step(wm, z, actions[t])
            frames.append(z)
        c0.apply_(predictor)
        return torch.stack(frames, dim=0), actions

    for _ in range(cfg.q):
        enc, actions = make_regime_chunk()
        best_umf = compute_umf(c0, wrapper, enc, actions, proprio_ctxt=proprio_ctxt)
        expander.record(best_umf, enc, actions, proprio_ctxt=proprio_ctxt)

    if expander._strikes < cfg.q:
        raise AssertionError(
            f"G3a FAILED: only {expander._strikes}/{cfg.q} strikes recorded -- "
            "the regime-shift scenario isn't producing deficit chunks. "
            "Try increasing REL_SCALE."
        )

    next_enc, next_actions = make_regime_chunk()
    outcome = expander.maybe_expand(library, wrapper, next_enc, next_actions, motion_gate=None,
                                     next_proprio_ctxt=proprio_ctxt)

    if outcome != "committed":
        raise AssertionError(
            f"G3a FAILED: expected 'committed', got {outcome!r}. "
            "Either the candidate isn't fitting the regime shift within n_probe steps "
            "(try increasing REL_SCALE or n_probe), or the probe's "
            "commit condition is broken."
        )
    print(f"PASSED  (outcome={outcome}, strikes={cfg.q})")


def gate_g3b(wm, wrapper) -> None:
    """G3b: unfixable, structureless noise -> probe rejects, nothing commits.

    Deliberately the crudest possible unfixable case (pure i.i.d. noise, zero
    learnable structure) -- see plan's Scope boundary note. This does not
    validate that verification catches a chart which merely improves UMF
    while still being bad for real planning; it only confirms the probe
    isn't vacuous (doesn't say yes to everything). compute_umf()/maybe_expand()
    now need the EncPredWM WRAPPER, not .model -- see
    E0_IMPLEMENTATION_PLAN.md T1/T2.
    """
    print("G3b: probe rejects unfixable noise...", end=" ")
    predictor = wm.predictor
    c0 = Chart(predictor, kind="full")
    library = Library(c0, max_size=5)
    cfg = ExpansionConfig(kind="full")
    expander = Expander(cfg)

    grid = wm.grid_size
    D, T = 384, 5
    N = grid * grid
    act_dim = wm.action_dim
    device = next(predictor.parameters()).device
    proprio_ctxt = _make_synthetic_proprio_ctxt(wm, grid, device)

    def make_noise_chunk():
        encoder_output = torch.randn(T + 1, N, D, device=device)
        actions = torch.randn(T, act_dim, device=device)
        return encoder_output, actions

    for _ in range(cfg.q):
        enc, actions = make_noise_chunk()
        best_umf = compute_umf(c0, wrapper, enc, actions, proprio_ctxt=proprio_ctxt)
        expander.record(best_umf, enc, actions, proprio_ctxt=proprio_ctxt)

    if expander._strikes < cfg.q:
        raise AssertionError(
            f"G3b FAILED: only {expander._strikes}/{cfg.q} strikes recorded -- "
            "pure noise chunks should reliably read as deficits. Check compute_umf/record()."
        )

    next_enc, next_actions = make_noise_chunk()
    outcome = expander.maybe_expand(library, wrapper, next_enc, next_actions, motion_gate=None,
                                     next_proprio_ctxt=proprio_ctxt)

    if outcome != "rejected_score":
        raise AssertionError(
            f"G3b FAILED: expected 'rejected_score', got {outcome!r}. "
            "If 'committed', the probe is vacuous -- it accepted an unfixable "
            "candidate. Check the passes_tau/beats_best logic in maybe_expand()."
        )
    print(f"PASSED  (outcome={outcome})")


# ── G4: regimes real ──────────────────────────────────────────────────────────
# Rewritten per IMPLEMENTATION_PLAN_V3 §9. The old version compared the mean
# pixel value of the whole flattened frame, averaged over random-action
# rollouts, with a 1e-6 threshold (CHECK 4.1) -- Push-T frames are ~97% white,
# so the mean image is near-identical for any regime and the test could neither
# pass meaningfully nor ever fail (on a real OR a fake regime). It was also
# hard-skipped in headless mode and had never actually run.
#
# New design (system-identification logic: hold the input constant, test if the
# output distribution differs). Four choices, each pinned rather than left open:
#   1. Fixed IDENTICAL aimed-walk actions across regimes (persistent random
#      target near the block + proportional aim), NOT a planner -- a planner
#      adapts and would compensate the shift away. Contact is REQUIRED, not
#      hoped for: raw random actions make contact ~15% of the time and the
#      regimes act only on contact / post-contact dynamics, so a no-contact
#      rollout carries no signal. G4 aborts if the R0 contact rate < 0.5 and
#      warns (low power) below 0.8.
#   2. n = 120 sequences per regime. This gate is CPU-only and free -- there is
#      no cost reason to run it underpowered.
#   3. Negative control with REAL sampling variance: R0 seeds [0,n) vs R0 seeds
#      [NULL_OFFSET, NULL_OFFSET+n) -- two independent R0 samples, different
#      initial states, SAME physics. Its bootstrap CI on the mean difference
#      must INCLUDE zero. (Comparing R0 to itself on the same seeds is
#      deterministic -- d_i == 0 exactly -- and tests nothing.)
#   4. ONE pre-registered primary statistic: combined block POSE change
#      sqrt(dx^2 + dy^2 + (G4_RAD_TO_PX * dtheta)^2) from start to end, where
#      G4_RAD_TO_PX = 30 px/rad ~ the T-block's moment arm (so a rotation and a
#      translation that move the block's extremity equally count equally).
#      Friction (R1) changes the rotation/translation split of a push, not
#      necessarily the translation magnitude, so a translation-only statistic
#      would be blind to it by construction -- the combined pose covers both.
#      The (translation, rotation) decomposition and a KS test are reported as
#      secondary, not decisive.
#
# PASS iff: (a) the null CI includes 0 (the procedure does not false-positive on
# noise), AND (b) for every shifted regime the paired-bootstrap CI on the mean
# displacement difference vs R0 excludes 0 AND the effect magnitude exceeds the
# null noise band (max |bound| of the null CI). "Materially large" is thus
# measured against real noise, not an arbitrary pixel count.

_G4_NULL_OFFSET = 100_000
_G4_ACTION_GAIN = 0.25  # matches run_e0.py's scripted collector (calibrated to
                        # the checkpoint's own action_std ~0.20)
_G4_RAD_TO_PX = 30.0    # T-block moment arm; folds rotation into the pose stat


def _g4_rollout(regime: str, seed: int, n_steps: int, track: bool = False) -> tuple[float, float, float, int]:
    """One aimed-walk rollout in the REAL Push-T sim under `regime`. Returns
    (combined pose change, translation magnitude, |Δangle| in rad, n_contacts).
    No world model, no renderer dependency -- reads info['block_pose']."""
    import numpy as np
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
    from atlas.regimes import PhysicsRegime

    rs = np.random.RandomState(seed)
    base_env = PushTEnv(render_size=96, with_velocity=True)
    env = PhysicsRegime(base_env, regime)
    env.seed(seed)
    obs, state = env.reset()

    block_xy0 = np.array([state[2], state[3]], dtype=np.float64)
    angle0 = float(state[4])
    agent_xy = np.asarray(obs["proprio"][:2], dtype=np.float64)
    action_scale = env.action_scale

    # Fixed push direction (seeded) + a fixed target point near the initial
    # block position. In `track` mode the aim point follows the block's current
    # position offset along that direction, so contact is SUSTAINED for the
    # whole rollout -- the right probe for "does the friction effect compound
    # over a long push?" (the fixed-target walk stops pushing once the block
    # slides off the target).
    push_dir = rs.uniform(-1.0, 1.0, size=(2,))
    push_dir /= (np.linalg.norm(push_dir) + 1e-9)
    target = block_xy0 + rs.uniform(-25.0, 25.0, size=(2,))

    n_contacts = 0
    block_xy = block_xy0.copy()
    last_pose = np.array([block_xy0[0], block_xy0[1], angle0])
    for _ in range(n_steps):
        aim = (block_xy - 35.0 * push_dir) if track else target
        direction = (aim - agent_xy) / action_scale
        noise = rs.normal(0.0, 0.15, size=(2,))
        act = np.clip((direction + noise) * _G4_ACTION_GAIN, -1.0, 1.0)
        obs, _, done, info = env.step(act)
        agent_xy = np.asarray(obs["proprio"][:2], dtype=np.float64)
        n_contacts += int(info["n_contacts"])
        last_pose = np.asarray(info["block_pose"], dtype=np.float64)
        block_xy = last_pose[:2]
        if done:
            break
    env.close()

    d_xy = last_pose[:2] - block_xy0
    d_theta = float(last_pose[2] - angle0)
    # wrap Δangle to [-pi, pi] so a near-2pi spin isn't read as a huge change
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    translation = float(np.linalg.norm(d_xy))
    pose = float(np.sqrt(d_xy[0] ** 2 + d_xy[1] ** 2 + (_G4_RAD_TO_PX * d_theta) ** 2))
    return pose, translation, abs(d_theta), n_contacts


def gate_g4(regimes: list[str], n: int = 120, n_steps: int = 40,
            selftest: bool = False) -> None:
    """G4: fixed identical actions under each regime => block trajectory
    statistics differ, paired, beyond a measured noise band. See the block
    comment above for the full pre-registered design.

    selftest=True replaces every shifted regime with R0 (a deliberately fake
    "shift") -- G4 must then FAIL criterion (b), proving it is not vacuous.
    """
    import numpy as np
    from atlas.stats import paired_bootstrap
    try:
        from scipy.stats import ks_2samp
    except Exception:
        ks_2samp = None

    r0 = regimes[0]
    shifted = [(r0 if selftest else r) for r in regimes[1:]]
    label = f"{regimes} (SELFTEST: shifts faked as {r0})" if selftest else f"{regimes}"
    print(f"G4: regime reality check {label}, n={n} x {n_steps} steps...")

    def batch(regime, seed_start):
        rows = [_g4_rollout(regime, seed_start + i, n_steps) for i in range(n)]
        return {k: np.array([x[i] for x in rows])
                for i, k in enumerate(("pose", "trans", "angle", "contacts"))}

    b0 = batch(r0, 0)
    b0b = batch(r0, _G4_NULL_OFFSET)
    pose0 = b0["pose"]

    contact_rate = float((b0["contacts"] > 0).mean())
    print(f"  R0 contact rate = {contact_rate:.2f}  "
          f"(pose Δ mean {pose0.mean():.1f} px, sd {pose0.std():.1f})")
    if contact_rate < 0.5:
        raise AssertionError(
            f"G4 FAILED: only {contact_rate:.0%} of R0 rollouts made contact -- "
            "the aimed-walk actions are not engaging the block, so no regime "
            "test has power. Fix the action generator before trusting G4.")
    if contact_rate < 0.8:
        print(f"  [WARN] R0 contact rate {contact_rate:.0%} < 0.8 -- reduced power.")

    # ── negative control: two independent R0 samples, same physics ────────────
    null_mean, (null_lo, null_hi) = paired_bootstrap(b0b["pose"], pose0)
    null_band = max(abs(null_lo), abs(null_hi))
    null_ok = null_lo <= 0.0 <= null_hi
    print(f"  NULL  R0[{_G4_NULL_OFFSET}:] - R0[0:]  mean {null_mean:+.2f} px  "
          f"CI [{null_lo:+.2f}, {null_hi:+.2f}]  "
          f"{'OK (contains 0)' if null_ok else 'BROKEN (excludes 0!)'}")
    if not null_ok:
        raise AssertionError(
            "G4 FAILED: the negative control (R0 vs R0, different seeds) has a "
            f"CI that excludes zero ({null_lo:+.2f}, {null_hi:+.2f}) -- the test "
            "procedure false-positives on pure sampling noise and cannot be "
            "trusted when it fires on a real regime.")

    # ── each shifted regime vs R0, paired by seed ────────────────────────────
    failures = []
    for name, actual in zip(regimes[1:], shifted):
        bS = batch(actual, 0)
        mean_d, (lo, hi) = paired_bootstrap(bS["pose"], pose0)   # same seeds => paired
        excludes0 = not (lo <= 0.0 <= hi)
        beats_noise = abs(mean_d) > null_band
        ks_p = (ks_2samp(bS["pose"], pose0).pvalue if ks_2samp is not None else float("nan"))
        d_trans = float(bS["trans"].mean() - b0["trans"].mean())
        d_ang = float(bS["angle"].mean() - b0["angle"].mean())
        verdict = "REAL" if (excludes0 and beats_noise) else "NOT DISTINGUISHABLE"
        print(f"  {name:>3} vs {r0}:  Δpose {mean_d:+.2f} px  CI [{lo:+.2f}, {hi:+.2f}]"
              f"  (Δtrans {d_trans:+.2f} px, Δ|angle| {d_ang:+.4f} rad; KS p={ks_p:.1e})"
              f"  -> {verdict}")
        if not (excludes0 and beats_noise):
            failures.append(name)

    if selftest:
        if failures == regimes[1:]:
            print("G4 SELFTEST PASSED  (faked shifts correctly reported NOT distinguishable)")
            return
        raise AssertionError(
            f"G4 SELFTEST FAILED: faked shifts {set(regimes[1:]) - set(failures)} "
            "were reported REAL -- the gate is vacuous (fires on no shift at all).")

    if failures:
        raise AssertionError(
            f"G4 FAILED: regime(s) {failures} produce block trajectories "
            f"statistically indistinguishable from {r0} (or within the noise "
            "band) -- the physics modification is not real enough to build on.")
    print(f"G4 PASSED  (all shifts real: null band ±{null_band:.2f} px)")


def g4_duration_sweep(regimes: list[str], n: int = 120,
                      steps_grid=(40, 80, 120, 160, 200)) -> None:
    """Diagnostic (not a pass/fail gate): does a regime's trajectory-level
    signature GROW with contact duration? Runs R0-vs-each-shift at several
    n_steps, in both `track=False` (fixed target) and `track=True` (aim point
    follows the block -> sustained push) modes, and prints Δpose against the
    R0-vs-R0 null band at each point. If Δpose climbs above the null band as
    n_steps grows, the effect compounds (usable in a full closed-loop episode
    even if invisible in a short probe); if it stays flat and inside the band,
    the shift is structurally undetectable this way.
    """
    import numpy as np
    from atlas.stats import paired_bootstrap

    r0 = regimes[0]
    for track in (False, True):
        mode = "block-tracking (sustained push)" if track else "fixed-target"
        print(f"\n=== G4 duration sweep — {mode}, n={n} ===")
        print(f"{'steps':>6} {'R0 contact':>11} {'null ±':>8}  " +
              "  ".join(f"{name}:Δpose[CI]" for name in regimes[1:]))
        for ns in steps_grid:
            def batch(regime, s0):
                rows = [_g4_rollout(regime, s0 + i, ns, track) for i in range(n)]
                return (np.array([x[0] for x in rows]), np.array([x[3] for x in rows]))
            pose0, con0 = batch(r0, 0)
            pose0b, _ = batch(r0, _G4_NULL_OFFSET)
            _, (nlo, nhi) = paired_bootstrap(pose0b, pose0)
            band = max(abs(nlo), abs(nhi))
            cols = []
            for name in regimes[1:]:
                poseS, _ = batch(name, 0)
                md, (lo, hi) = paired_bootstrap(poseS, pose0)
                flag = "  REAL" if (not (lo <= 0 <= hi)) and abs(md) > band else "  --"
                cols.append(f"{name} {md:+6.1f} [{lo:+.1f},{hi:+.1f}]{flag}")
            print(f"{ns:>6} {(con0 > 0).mean():>11.2f} {band:>8.2f}  " + "   ".join(cols))


def _g5_build_and_reset(seed: int):
    """Construct a real PushTEnv, seed it, reset it, and return the raw init
    state vector + goal_pose. Used both by gate_g5 (same-seed case, must
    match) and by scratchpad/g2_g5_demo.py (mismatched-seed case, must NOT
    match) -- see FIX_SPEC C2."""
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
    env = PushTEnv(render_size=224, with_velocity=True)
    env.seed(seed)
    obs, state = env.reset()
    goal = env.goal_pose.copy()
    env.close()
    return obs["visual"].copy(), state.copy(), goal


def gate_g5() -> None:
    """G5: Two arms, same seed -> genuinely identical initial env state + goal.

    Rewritten (the previous version only checked that paired_seed() ignores
    its `arm` argument -- true by inspection, since `arm` is never
    referenced in the function body; it built no env and sampled nothing).
    This version constructs two REAL PushTEnv instances at the seed
    paired_seed() produces, resets both, and asserts the raw init state
    vector, the rendered visual observation, and the goal_pose are all
    bit-identical -- what "same seed -> identical initial states/goals"
    (CLAUDE.md G5 definition) actually means for the real env, not just for
    an integer.
    """
    print("G5: paired seeding check (real env)...", end=" ")
    import numpy as np
    from atlas.streams import paired_seed

    s1 = paired_seed(0, 0, arm="atlas")
    s2 = paired_seed(0, 0, arm="frozen")
    if s1 != s2:
        raise AssertionError(
            f"G5 FAILED: paired_seed() depends on the 'arm' argument "
            f"(atlas={s1}, frozen={s2})."
        )

    vis_a, state_a, goal_a = _g5_build_and_reset(s1)
    vis_b, state_b, goal_b = _g5_build_and_reset(s2)

    if not np.array_equal(state_a, state_b):
        raise AssertionError(
            f"G5 FAILED: same seed {s1} produced different init states "
            f"({state_a} vs {state_b}) across two independently constructed "
            "envs -- two arms would not be paired."
        )
    if not np.array_equal(vis_a, vis_b):
        raise AssertionError(
            "G5 FAILED: same seed produced different initial visual "
            "observations across two independently constructed envs."
        )
    if not np.array_equal(goal_a, goal_b):
        raise AssertionError(
            f"G5 FAILED: same seed produced different goals ({goal_a} vs "
            f"{goal_b})."
        )
    print(f"PASSED  (seed={s1}, init_state={state_a}, goal={goal_a})")


def gate_g6(wm, wrapper) -> None:
    """G6: Static chunk → compute_umf returns None (motion_gate or zero denominator).

    compute_umf() now needs the EncPredWM WRAPPER, not .model -- see
    E0_IMPLEMENTATION_PLAN.md T1/T2.
    """
    print("G6: denominator / static-chunk gate check...", end=" ")
    predictor = wm.predictor
    c0 = Chart(predictor, "ln_act")
    grid = wm.grid_size
    act_dim = wm.action_dim   # model action dim = raw_dim * frameskip
    N, D, T = grid * grid, 384, 5
    device = next(predictor.parameters()).device
    # Static chunk: all frames identical -> displacement = 0 -> denominator = 0.
    z0 = torch.randn(N, D, device=device)
    encoder_output = z0.unsqueeze(0).expand(T + 1, -1, -1).clone()
    actions = torch.zeros(T, act_dim, device=device)

    result = compute_umf(c0, wrapper, encoder_output, actions, motion_gate=0.0)
    if result is not None:
        raise AssertionError(
            f"G6 FAILED: expected None for static chunk but got UMF = {result:.4f}. "
            "Check the denominator guard in score.umf()."
        )
    print("PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ATLAS correctness gates.")
    parser.add_argument("--all", action="store_true", help="Run all gates.")
    parser.add_argument("--gate", choices=["G1", "G2", "G3a", "G3b", "G4", "G5", "G6"],
                        help="Run a specific gate.")
    parser.add_argument("--regime", default="R1", help="Regime for G4 (default R1)")
    parser.add_argument("--g4-regimes", default="R0,R1,R2",
                        help="G4: comma-separated, first is the reference (default R0,R1,R2)")
    parser.add_argument("--g4-n", type=int, default=120,
                        help="G4: rollouts per regime (default 120)")
    parser.add_argument("--g4-selftest", action="store_true",
                        help="G4: fake every shift as the reference regime; the gate MUST fail")
    parser.add_argument("--g4-sweep", action="store_true",
                        help="G4: run the contact-duration diagnostic sweep instead of the gate")
    args = parser.parse_args()

    if not args.all and args.gate is None:
        parser.print_help()
        sys.exit(1)

    run = args.gate
    run_all = args.all

    failed = []

    def run_gate(name, fn, *fn_args):
        if run_all or run == name:
            try:
                fn(*fn_args)
            except Exception as e:
                print(f"  [FAIL] {e}")
                failed.append(name)

    # G5 does not require loading the model checkpoint
    run_gate("G5", gate_g5)

    if run_all or run in ("G1", "G2", "G3a", "G3b", "G6"):
        import torch
        print("Loading dino_wm_pusht...")
        # FIX_SPEC.md C8: this used to omit source="local", so torch.hub
        # resolved "facebookresearch/jepa-wms" against the REMOTE repo spec
        # (github.com/facebookresearch/jepa-wms) rather than the patched
        # local checkout at HUB_PATH (this file's own top-of-file sys.path
        # insertion, same as run_e0_planning.py/diagnose_cem_costs.py) --
        # production never uses the remote copy, so these gates could pass
        # or fail against code nothing else in this project runs.
        model, prep = torch.hub.load(
            _HUB_PATH, "dino_wm_pusht", source="local",
            force_reload=False, trust_repo=True,
        )
        # torch.hub returns the EncPredWM wrapper; underlying VideoWM is at
        # .model. compute_umf()/route()/maybe_expand() need the WRAPPER
        # (T1/T2); predictor state-dict ops and direct forward_pred/encode_act
        # calls in this file's own synthetic-data helpers still need `wm`.
        wrapper = model
        wm = model.model if hasattr(model, "model") else model
        for p in wm.encoder.parameters():
            p.requires_grad_(False)

        run_gate("G1", gate_g1, wm, wrapper)
        run_gate("G2", gate_g2, wm, wrapper)
        run_gate("G3a", gate_g3a, wm, wrapper)
        run_gate("G3b", gate_g3b, wm, wrapper)
        run_gate("G6", gate_g6, wm, wrapper)

    if run_all or run == "G4":
        # G4 runs the real pymunk Push-T sim (CPU, no model, no GPU) -- it is
        # no longer headless-skipped (IMPLEMENTATION_PLAN_V3 §9). --regimes lets
        # the caller pick which pair(s); default R0 vs R1 vs R2.
        g4_regimes = args.g4_regimes.split(",")
        if args.g4_sweep:
            g4_duration_sweep(g4_regimes, args.g4_n)
        else:
            run_gate("G4", gate_g4, g4_regimes, args.g4_n, 40, args.g4_selftest)

    if failed:
        print(f"\n{'='*40}")
        print(f"FAILED gates: {failed}")
        sys.exit(1)
    else:
        print("\nAll available gates PASSED.")


if __name__ == "__main__":
    main()
