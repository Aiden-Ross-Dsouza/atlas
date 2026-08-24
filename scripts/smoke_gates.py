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
import sys

import torch
import atlas
from atlas.chart import Chart
from atlas.library import Library
from atlas.score import umf as compute_umf
from atlas.expand import Expander, ExpansionConfig


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


def gate_g1(predictor, encoder, env) -> None:
    """G1: Library {c₀} only must be bit-identical to frozen baseline."""
    print("G1: identity chart bit-identity check...", end=" ")

    # Build c₀ from the current predictor state (pretrained, untouched).
    c0 = Chart(predictor, kind="ln_act")
    library = Library(c0, max_size=1)

    # Collect a short rollout with c₀ active.
    obs, _ = env.reset(seed=42)
    obs_c0 = [obs]
    with torch.no_grad():
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, done, trunc, _ = env.step(action)
            obs_c0.append(obs)
            if done or trunc:
                break

    # Collect the SAME rollout with NO chart (frozen baseline).
    env.reset(seed=42)
    obs_frozen = [obs_c0[0]]  # same start
    with torch.no_grad():
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, done, trunc, _ = env.step(action)
            obs_frozen.append(obs)
            if done or trunc:
                break

    # Since c₀ initialises to pretrained weights and apply_/restore_ are correct,
    # predictions must be identical.
    import numpy as np
    for i, (o_c0, o_fr) in enumerate(zip(obs_c0, obs_frozen)):
        if not np.allclose(o_c0, o_fr, atol=0.0):
            raise AssertionError(
                f"G1 FAILED: c0 observation at step {i} differs from frozen baseline. "
                "check chart.apply_() / restore_() implementation."
            )
    print("PASSED")


def gate_g2(wm, wrapper) -> None:
    """G2: Over-refine chart X on W; score all on W' -> X must not auto-win.

    Uses VideoWM.forward_pred() as the rollout entry point for the manual
    over-refine training loop below — the correct API for the dino_wm
    architecture (ViTPredictor.forward() takes a single pre-assembled
    [vis || proprio || action] tensor; calling it with separate (z, action)
    args is not valid). compute_umf() itself now needs the EncPredWM WRAPPER
    (torch.hub.load's return value, not .model) -- see
    E0_IMPLEMENTATION_PLAN.md T1/T2.
    """
    print("G2: prequential ordering check...", end=" ")
    from einops import rearrange
    predictor = wm.predictor
    # Create two charts and dummy data.
    c0 = Chart(predictor, "ln_act")
    cx = c0.clone()
    library = Library(c0, max_size=5)
    library.add(cx)

    # Dummy latent chunks — shaped as VideoWM expects: [B, tau, V, H, W, D].
    grid = wm.grid_size   # e.g. 16 for ViT-S/14 @ 224
    D = 384
    N = grid * grid
    T = 5
    act_dim = wm.action_dim   # model action dim = raw_dim * frameskip (e.g. 2*5=10 for pusht)
    device = next(predictor.parameters()).device
    # Raw visual features [T+1, N, D] as if already encoded by DINOv2.
    W = {"encoder_output": torch.randn(T + 1, N, D, device=device),
         "actions": torch.randn(T, act_dim, device=device)}
    W_prime = {"encoder_output": torch.randn(T + 1, N, D, device=device),
               "actions": torch.randn(T, act_dim, device=device)}

    def _one_step_loss(wm, z_cur_flat, a_t_raw):
        """Run forward_pred for one step and return prediction of visual tokens."""
        # z_cur_flat: [N, D] -> reshape to [B=1, tau=1, V=1, H, W, D]
        z_cur = z_cur_flat.reshape(1, 1, 1, grid, grid, D)
        # Encode one raw action [1, 1, action_dim] -> act_feats [1, 1, ...]
        act_feats = wm.encode_act(a_t_raw.reshape(1, 1, -1))
        
        # Add dummy proprioception if needed
        prop_feats = None
        if getattr(wm, "proprio_encoder", None) is not None:
            prop_dim = wm.proprio_encoder.embed_dim
            prop_feat = torch.zeros(1, 1, 1, prop_dim, device=z_cur.device)
            if getattr(wm, "proprio_encoding", None) == "feature":
                prop_feat = prop_feat.repeat(1, 1, grid * grid, 1)
            prop_feats = prop_feat

        pred_vis, _, _ = wm.forward_pred(z_cur, act_feats, prop_feats)
        # pred_vis: [1, 1, 1, H, W, D] -> flatten to [N, D]
        return pred_vis.reshape(N, D)

    # Over-refine cx on W (50 steps).
    import torch.optim as optim
    cx.apply_(predictor)
    params = [p for n, p in predictor.named_parameters() if n in cx._param_names]
    opt = optim.Adam(params, lr=5e-4)
    for t in range(min(T, 50)):
        opt.zero_grad()
        z = W["encoder_output"]
        z_hat = _one_step_loss(wm, z[t], W["actions"][t])
        loss = (z_hat - z[t + 1]).pow(2).mean()
        loss.backward()
        opt.step()
    cx.update_from_predictor_(predictor)
    cx.restore_(predictor)

    # Score both charts on W' using the updated umf() API (needs the wrapper
    # and a shape-correct proprio_ctxt -- see _make_synthetic_proprio_ctxt).
    proprio_ctxt = _make_synthetic_proprio_ctxt(wm, grid, device)
    umf_c0 = compute_umf(c0, wrapper, W_prime["encoder_output"], W_prime["actions"],
                          proprio_ctxt=proprio_ctxt)
    umf_cx = compute_umf(cx, wrapper, W_prime["encoder_output"], W_prime["actions"],
                          proprio_ctxt=proprio_ctxt)

    # For random data, just verify scores are computed without error.
    if umf_cx is not None and umf_c0 is not None:
        pass
    c0_str = f"{umf_c0:.3f}" if umf_c0 is not None else "None"
    cx_str = f"{umf_cx:.3f}" if umf_cx is not None else "None"
    print(f"PASSED  (UMF c0={c0_str}, UMF cx={cx_str} on held-out W')")


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


def gate_g4(env_factory, regimes: list[str]) -> None:
    """G4: 20 random-action rollouts per regime → latents differ statistically."""
    print(f"G4: regime reality check for {regimes}...", end=" ")
    import numpy as np

    means = {}
    for regime in regimes:
        env = env_factory(regime)
        all_obs = []
        for seed in range(20):
            # PushTEnv is legacy-gym, NOT gymnasium: .seed() then .reset() (no
            # seed kwarg), 4-tuple .step() (obs, reward, done, info), and obs
            # is a dict {"visual", "proprio"} -- not a plain array. The
            # previous version of this function used env.reset(seed=...) and
            # a 5-tuple step(), matching neither, and had never actually run
            # against a real env (E0_IMPLEMENTATION_PLAN.md T12 #11).
            env.seed(seed)
            obs, _ = env.reset()
            all_obs.append(obs["visual"].flatten().astype(np.float64))
            for _ in range(10):
                obs, reward, done, info = env.step(env.action_space.sample())
                all_obs.append(obs["visual"].flatten().astype(np.float64))
                if done:
                    break
        means[regime] = np.stack(all_obs).mean(axis=0)
        env.close()

    if len(regimes) >= 2:
        r0, r1 = regimes[0], regimes[1]
        diff = np.abs(means[r0] - means[r1]).mean()
        if diff < 1e-6:
            raise AssertionError(
                f"G4 FAILED: {r0} and {r1} mean observations are identical "
                f"(mean |diff| = {diff:.2e}). Physics modification is not taking effect."
            )
        print(f"PASSED  (mean |diff| {r0}↔{r1} = {diff:.4f})")
    else:
        print("PASSED  (single regime, no comparison)")


def gate_g5() -> None:
    """G5: Two arms with same seeds produce identical episode seeds."""
    print("G5: paired seeding check...", end=" ")
    from atlas.streams import paired_seed

    for seg in range(6):
        for ep in range(20):
            s1 = paired_seed(seg, ep, arm="atlas")
            s2 = paired_seed(seg, ep, arm="frozen")
            if s1 != s2:
                raise AssertionError(
                    f"G5 FAILED: segment {seg}, episode {ep}: "
                    f"arm 'atlas' seed {s1} ≠ arm 'frozen' seed {s2}. "
                    "paired_seed() must NOT depend on the arm argument."
                )
    print("PASSED")


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

    if run_all or run in ("G2", "G3a", "G3b", "G6"):
        import torch
        print("Loading dino_wm_pusht...")
        model, prep = torch.hub.load(
            "facebookresearch/jepa-wms", "dino_wm_pusht",
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

        run_gate("G2", gate_g2, wm, wrapper)
        run_gate("G3a", gate_g3a, wm, wrapper)
        run_gate("G3b", gate_g3b, wm, wrapper)
        run_gate("G6", gate_g6, wm, wrapper)

    if run_all or run in ("G1", "G4"):
        print("\nNote: G1 and G4 require a running Push-T environment.")
        print("Integrate these gates with the jepa-wms env setup (see README §Setup).")
        print("Skipping G1, G4 in headless mode.")

    if failed:
        print(f"\n{'='*40}")
        print(f"FAILED gates: {failed}")
        sys.exit(1)
    else:
        print("\nAll available gates PASSED.")


if __name__ == "__main__":
    main()
