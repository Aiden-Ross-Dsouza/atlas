"""
loop.py — atlas_step(): the ATLAS prequential controller (C1 + C2).

Called once per replan, BEFORE planning. The upstream planning code calls:

    chart_idx = atlas_step(state, library, expander, chunk, cfg)
    world_model.apply_chart(library[chart_idx])
    plan = cem_planner(world_model, ...)
    world_model.restore_chart(library.c0)   # restore baseline
    world_model.apply_chart(library[chart_idx])
    [execute 5 actions, collect new chunk]
    atlas_refine(library[chart_idx], predictor, new_chunk, cfg)

Step order (strict — never re-order):
  1. SCORE    UMF(c; Q) for all c
  2. SELECT   c* = argmin UMF, with hysteresis margin m
  3. EXPAND   if strikes ≥ q, fire probe on NEXT chunk (see expand.py)
  4. (caller) EXECUTE plan under c*
  5. REFINE   1 SGD step on c*      ← atlas_refine(), called by caller AFTER step 4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from atlas.library import Library
from atlas.router import route, RouterKind
from atlas.expand import Expander, ExpansionConfig, ProbeOutcome
from atlas.score import umf as compute_umf


@dataclass
class ATLASConfig:
    router: RouterKind = "umf"
    tau: float = 0.5
    q: int = 3
    hysteresis: float = 0.05
    lr: float = 5e-4
    n_probe: int = 20
    motion_gate: float | None = None   # computed from training data; set before use
    k_max: int = 10

    # For oracle router only:
    label_to_chart: dict[int, int] | None = None

    # Expansion mode: 'atlas' | 'detect_only' | 'fixed' | 'none'
    expansion_mode: Literal["atlas", "detect_only", "fixed", "none"] = "atlas"


@dataclass
class StepInfo:
    selected_idx: int
    scores: list[float | None]
    gated: bool
    probe_outcome: ProbeOutcome
    strikes: int


def atlas_step(
    library: Library,
    expander: Expander,
    world_model,
    encoder_output: torch.Tensor,
    actions: torch.Tensor,
    current_idx: int,
    cfg: ATLASConfig,
    *,
    regime_label: int | None = None,
    # next_chunk is only needed for full ATLAS expansion; pass None otherwise
    next_encoder_output: torch.Tensor | None = None,
    next_actions: torch.Tensor | None = None,
) -> StepInfo:
    """
    Execute one ATLAS decision step (SCORE + SELECT + EXPAND).
    Refinement (REFINE) is NOT done here — call atlas_refine() after planning.

    Args:
        library:              Current chart library.
        expander:             Stateful expander (strike counter).
        world_model:          EncPredWM instance (the object torch.hub.load
                              returns — NOT .model). Passed through to
                              route()/expander.maybe_expand(), both of which
                              need the wrapper (see E0_IMPLEMENTATION_PLAN.md T1/T2).
        encoder_output:       Current chunk [T+1, N, D].
        actions:              Executed actions [T, action_dim].
        current_idx:          Currently active chart index.
        cfg:                  ATLAS hyperparameters.
        regime_label:         True regime label (oracle router only).
        next_encoder_output:  Next chunk for expansion verification (ATLAS mode).
        next_actions:         Next chunk actions (ATLAS mode).

    Returns:
        StepInfo with selected chart index and diagnostics.
    """
    # ── 1. SCORE + 2. SELECT ─────────────────────────────────────────────────
    selected_idx, route_info = route(
        kind=cfg.router,
        library=library,
        world_model=world_model,
        encoder_output=encoder_output,
        actions=actions,
        current_idx=current_idx,
        motion_gate=cfg.motion_gate,
        hysteresis=cfg.hysteresis,
        regime_label=regime_label,
        label_to_chart=cfg.label_to_chart,
    )

    scores = route_info["scores"]
    gated = route_info["gated"]

    # ── Best UMF for strike counter ───────────────────────────────────────────
    best_umf: float | None = None
    if not gated and scores:
        valid = [s for s in scores if s is not None]
        if valid:
            best_umf = min(valid)

    # ── 3. EXPAND ─────────────────────────────────────────────────────────────
    probe_outcome: ProbeOutcome = "not_ready"

    if cfg.expansion_mode == "atlas":
        expander.record(best_umf, encoder_output, actions)
        if (
            next_encoder_output is not None
            and next_actions is not None
            and expander._strikes >= cfg.q
        ):
            probe_outcome = expander.maybe_expand(
                library, world_model, next_encoder_output, next_actions, cfg.motion_gate
            )

    elif cfg.expansion_mode == "detect_only":
        # Detect-and-spawn: commit immediately when strikes ≥ q, no verification.
        expander.record(best_umf, encoder_output, actions)
        if expander._strikes >= cfg.q and not library.is_full():
            best_idx = selected_idx
            new_chart = library.clone_from(best_idx)
            library.add(new_chart)
            selected_idx = len(library) - 1
            expander._strikes = 0
            expander._deficit_chunks.clear()
            expander._n_committed += 1
            expander._n_probes_fired += 1
            probe_outcome = "committed"

    # 'fixed' and 'none' modes: no expansion logic.

    return StepInfo(
        selected_idx=selected_idx,
        scores=scores,
        gated=gated,
        probe_outcome=probe_outcome,
        strikes=expander._strikes,
    )


def atlas_refine(
    chart,
    world_model,
    encoder_output: torch.Tensor,
    actions: torch.Tensor,
    lr: float = 5e-4,
) -> float:
    """
    One AdaJEPA gradient step on the selected chart.
    Must be called AFTER scoring and planning (step 5 in the prequential order).

    Rolls out via _open_loop_rollout() -- the same EncPredWM.unroll()-based
    function score.umf() and harness.run_e0_finetune() use, rather than the
    bare predictor(z_cur, a_t) call this used to make, which is not a valid
    call signature for this ViTPredictor (see E0_DIAGNOSIS_AND_PLAN.md).

    Args:
        chart:          The selected chart (c*).
        world_model:    EncPredWM instance (the object torch.hub.load
                        returns — NOT .model). Predictor reached via
                        world_model.model.predictor.
        encoder_output: Current chunk [T+1, N, D].
        actions:        Executed actions [T, action_dim].
        lr:             Learning rate.

    Returns:
        Scalar loss value for logging.
    """
    import torch.optim as optim
    from atlas.score import _open_loop_rollout

    predictor = world_model.model.predictor
    chart.apply_(predictor)
    params = [p for n, p in predictor.named_parameters() if n in chart._param_names]
    optimizer = optim.Adam(params, lr=lr)

    optimizer.zero_grad()
    z_vis = encoder_output[0]
    z_preds = _open_loop_rollout(world_model, z_vis, actions)  # [T, N, D]
    loss = (z_preds - encoder_output[1:]).pow(2).mean(dim=-1).mean()
    loss.backward()
    optimizer.step()

    scalar_loss = loss.item()

    # [WandB Logging] Log to active WandB run if initialized
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "loop/probe_refine_loss": scalar_loss,
                "kind": chart.kind,
            })
    except ImportError:
        pass

    chart.update_from_predictor_(predictor)
    chart.restore_(predictor)  # restore predictor to chart's baseline weights
    return scalar_loss
