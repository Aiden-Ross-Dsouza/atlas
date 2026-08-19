"""
expand.py — Verification-gated expansion (C2).

Rules:
  1. One LIBRARY-LEVEL strike counter: the library is inadequate iff its
     BEST chart UMF > τ for q consecutive informative chunks.
  2. When strikes == q: fit a CANDIDATE chart on the DEFICIT chunks only
     (not the full recent window — that would span regimes).
  3. VERIFY on the NEXT unseen chunk: commit only if the candidate
     beats BOTH τ AND the current best chart.
  4. Probe rejected → strikes reset, no new chart.
  5. Library cap (K_max) is enforced: if the library is full, the probe
     fires but cannot commit — logged as 'rejected_full'.

The expansion module is stateful (strike counter + pending candidate).
One Expander instance lives per ATLAS loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.optim as optim

from atlas.chart import Chart, ChartKind
from atlas.library import Library
from atlas.score import umf as compute_umf


ProbeOutcome = Literal["committed", "rejected_score", "rejected_full", "not_ready"]


@dataclass
class ExpansionConfig:
    tau: float = 0.5          # UMF adequacy threshold
    q: int = 3                # strikes before arming probe
    n_probe: int = 20         # gradient steps to fit the candidate
    lr: float = 5e-4          # AdaJEPA's learning rate
    kind: ChartKind = "ln_act"  # set to winner from E0


class Expander:
    """
    Tracks the library-level strike counter and manages the fixability probe.
    """

    def __init__(self, cfg: ExpansionConfig) -> None:
        self.cfg = cfg
        self._strikes: int = 0
        self._deficit_chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._candidate: Chart | None = None
        self._n_probes_fired: int = 0
        self._n_probes_rejected: int = 0
        self._n_committed: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        best_umf: float | None,
        encoder_output: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """
        Called every informative replan with the library's best UMF.

        Args:
            best_umf:       UMF of the best chart this replan (None if gated).
            encoder_output: Current chunk's encoded states [T+1, N, D].
            actions:        Executed actions [T, action_dim].
        """
        if best_umf is None:
            return  # uninformative chunk — do not count as a strike

        if best_umf > self.cfg.tau:
            self._strikes += 1
            self._deficit_chunks.append((encoder_output.detach(), actions.detach()))
        else:
            self._strikes = 0
            self._deficit_chunks.clear()
            self._candidate = None

    def maybe_expand(
        self,
        library: Library,
        world_model,
        next_encoder_output: torch.Tensor,
        next_actions: torch.Tensor,
        motion_gate: float | None,
    ) -> ProbeOutcome:
        """
        If strikes >= q: fit a candidate and verify on the next unseen chunk.

        Args:
            library:             The current chart library (may be modified).
            world_model:         VideoWM instance (EncPredWM.model from torch.hub).
            next_encoder_output: The NEXT chunk (held-out from candidate training).
            next_actions:        Actions for the next chunk.
            motion_gate:         Informative-chunk gate threshold.

        Returns:
            ProbeOutcome string.
        """
        if self._strikes < self.cfg.q:
            return "not_ready"

        self._n_probes_fired += 1

        if library.is_full():
            # Cannot commit even if the probe passes.
            self._strikes = 0
            self._deficit_chunks.clear()
            self._candidate = None
            self._n_probes_rejected += 1
            return "rejected_full"

        # ── Fit candidate on deficit chunks ───────────────────────────────────
        best_chart = library[_argmin_umf(library, world_model,
                                         next_encoder_output, next_actions, motion_gate)]
        candidate = library.clone_from(library._charts.index(best_chart))
        _fit_candidate(candidate, world_model, self._deficit_chunks,
                       self.cfg.n_probe, self.cfg.lr)

        # ── Verify on the next unseen chunk ───────────────────────────────────
        cand_umf = compute_umf(candidate, world_model, next_encoder_output,
                                next_actions, motion_gate)
        best_umf = compute_umf(best_chart, world_model, next_encoder_output,
                                next_actions, motion_gate)

        passes_tau = (cand_umf is not None) and (cand_umf < self.cfg.tau)
        beats_best = (cand_umf is not None) and (best_umf is None or cand_umf < best_umf)

        if passes_tau and beats_best:
            library.add(candidate)
            self._n_committed += 1
            self._strikes = 0
            self._deficit_chunks.clear()
            self._candidate = None
            return "committed"
        else:
            self._strikes = 0
            self._deficit_chunks.clear()
            self._candidate = None
            self._n_probes_rejected += 1
            return "rejected_score"

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "strikes": self._strikes,
            "probes_fired": self._n_probes_fired,
            "probes_rejected": self._n_probes_rejected,
            "charts_committed": self._n_committed,
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fit_candidate(
    candidate: Chart,
    world_model,
    deficit_chunks: list[tuple[torch.Tensor, torch.Tensor]],
    n_steps: int,
    lr: float,
) -> None:
    """
    Fit *candidate* on the deficit chunks with n_steps gradient steps.
    Uses AdaJEPA's exact loss: mean L2 in latent space.
    Calls VideoWM.forward_pred() for rollouts — the correct API for dino_wm.
    The ViTPredictor is temporarily modified in-place via chart.apply_.
    """
    predictor = world_model.predictor
    grid = world_model.grid_size
    candidate.apply_(predictor)
    params = [p for n, p in predictor.named_parameters() if n in candidate._param_names]
    optimizer = optim.Adam(params, lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=next(iter(params)).device)
        for enc_out, actions in deficit_chunks:
            T = actions.shape[0]
            D = enc_out.shape[-1]
            z = enc_out  # [T+1, N, D]
            # Encode all actions for this chunk at once
            act_feats = world_model.encode_act(actions.unsqueeze(0))  # [1, T, ...]
            
            # Setup dummy proprioception if needed
            proprio_buf = None
            if getattr(world_model, "proprio_encoder", None) is not None:
                prop_dim = world_model.proprio_encoder.embed_dim
                prop_feat = torch.zeros(1, 1, 1, prop_dim, device=z.device)
                if world_model.proprio_encoding == "feature":
                    prop_feat = prop_feat.repeat(1, 1, grid * grid, 1)
                proprio_buf = prop_feat

            loss = torch.tensor(0.0, device=z.device)
            for t in range(T):
                z_cur = z[t].reshape(1, 1, 1, grid, grid, D)  # [1, 1, 1, H, W, D]
                a_t = act_feats[:, t : t + 1]                  # [1, 1, ...]
                pred_vis, _, _ = world_model.forward_pred(
                    z_cur, a_t, proprio_buf
                )
                z_next_hat = pred_vis.reshape(grid * grid, D)   # [N, D]
                loss = loss + (z_next_hat - z[t + 1]).pow(2).mean()
            total_loss = total_loss + loss / T
        (total_loss / len(deficit_chunks)).backward()
        optimizer.step()

    # Pull updated weights back into the chart, then restore predictor.
    candidate.update_from_predictor_(predictor)
    candidate.restore_(predictor)  # no-op net after update_from_predictor_


def _argmin_umf(
    library: Library,
    world_model,
    encoder_output: torch.Tensor,
    actions: torch.Tensor,
    motion_gate: float | None,
) -> int:
    """Return index of library chart with lowest UMF. Falls back to 0."""
    best_idx, best_score = 0, float("inf")
    for i, chart in enumerate(library):
        s = compute_umf(chart, world_model, encoder_output, actions, motion_gate)
        if s is not None and s < best_score:
            best_score, best_idx = s, i
    return best_idx
