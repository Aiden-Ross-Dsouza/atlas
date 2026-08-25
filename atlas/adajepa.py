"""
adajepa.py — AdaJEPA reimplemented inside the jepa-wms substrate.

Implements two variants:
  AdaJEPA               — per-episode re-init from pretrained weights; 1 SGD
                          step per replan on 5 most-recent transitions.
  (variant="persistent") — same, but NO per-episode re-init (cross-episode
                          retention). Labelled as our modification in the paper.

Both use:
  - predictor lr 5e-4, Adam (AdaJEPA published hyperparameters)
  - buffer of 5 most recent (encoder_output, actions, proprio_ctxt) transitions
  - encoder FROZEN
  - loss: mean L2 latent prediction error (AdaJEPA Eq. 1), computed via
    _open_loop_rollout()/_make_z_ctxt() -- the same EncPredWM.unroll()-based
    path score.umf() and atlas.loop.atlas_refine() use, NOT the old bare
    predictor(z_cur, a_t) call (not a valid ViTPredictor call signature; see
    E0_DIAGNOSIS_AND_PLAN.md).
  - exactly 1 gradient step per replan (one backward per buffer item, summed
    grads -- O(1) peak memory, same P2a fix as harness.py's run_e0_finetune)

Prequential invariant: refinement ALWAYS occurs AFTER scoring.
Callers are responsible for calling refine() after score/select.

E3_E4_IMPLEMENTATION_PLAN.md §0.3: this file is owned by the E3/E4 agent;
the E0/E1 agent never imports it.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor

AdaVariant = Literal["adajepa", "persistent"]


class AdaJEPA:
    """
    AdaJEPA adaptation controller.

    Usage (per-replan, in order — NEVER reverse):
        1. caller scores/selects/plans using other machinery (AdaJEPA arms do
           not route -- there is nothing to score against)
        2. push(encoder_output, actions, proprio_ctxt) with the just-executed chunk
        3. refine()   — 1 SGD step on adapted params, using the buffer

    For AdaJEPA (not persistent): call adapter.reset() at episode start.
    """

    BUFFER_SIZE = 5

    def __init__(
        self,
        world_model,
        param_names: list[str],
        variant: AdaVariant = "adajepa",
        lr: float = 5e-4,
    ) -> None:
        """
        Args:
            world_model: EncPredWM instance (the object torch.hub.load
                         returns — NOT .model). _open_loop_rollout needs the
                         wrapper's canonical unroll(); the predictor being
                         adapted is reached via world_model.model.predictor.
            param_names: exactly Chart(predictor, kind)._param_names for the
                         E0 winner kind, so arms 2/3 adapt the SAME parameter
                         surface arms 4/5/6 do (plan §7.6: same loss, lr,
                         optimiser, buffer size; only library/routing/
                         expansion differ).
            variant:     'adajepa' resets per-episode; 'persistent' does not.
            lr:          Learning rate (AdaJEPA default 5e-4).
        """
        self.world_model = world_model
        self.predictor = world_model.model.predictor
        self.param_names = list(param_names)
        self.variant = variant
        self.lr = lr
        self.pretrained_state = {
            k: v.detach().clone()
            for k, v in self.predictor.state_dict().items()
            if k in self.param_names
        }
        self._buffer: deque[tuple[Tensor, Tensor, Tensor | None]] = deque(maxlen=self.BUFFER_SIZE)
        self._params = [p for n, p in self.predictor.named_parameters() if n in self.param_names]
        for p in self._params:
            p.requires_grad_(True)
        self._optimizer = optim.Adam(self._params, lr=lr)

    def reset(self) -> None:
        """
        Called at the start of each episode.
        For 'adajepa': resets predictor to pretrained weights, clears buffer,
        and re-creates the optimizer (fresh Adam moment state).
        For 'persistent': no-op (cross-episode retention is the point).
        """
        if self.variant == "adajepa":
            self.predictor.load_state_dict(self.pretrained_state, strict=False)
            self._buffer.clear()
            self._optimizer = optim.Adam(self._params, lr=self.lr)

    def push(
        self,
        encoder_output: Tensor,
        actions: Tensor,
        proprio_ctxt: Tensor | None = None,
    ) -> None:
        """Add a (encoder_output, actions, proprio_ctxt) chunk to the rolling buffer."""
        self._buffer.append((
            encoder_output.detach(),
            actions.detach(),
            None if proprio_ctxt is None else proprio_ctxt.detach(),
        ))

    def refine(self) -> float:
        """
        One AdaJEPA gradient step on the buffer contents.
        Must be called AFTER scoring and planning.

        Returns:
            The mean scalar loss value across buffer items, for logging.
        """
        from atlas.score import _open_loop_rollout, _make_z_ctxt

        if not self._buffer:
            return 0.0

        self._optimizer.zero_grad()
        total = 0.0
        for enc_out, actions, proprio_ctxt in self._buffer:
            z_ctxt = _make_z_ctxt(self.world_model, enc_out[0], proprio_ctxt)
            z_preds = _open_loop_rollout(self.world_model, z_ctxt, actions)
            loss = (z_preds - enc_out[1:]).pow(2).mean(dim=-1).mean()
            (loss / len(self._buffer)).backward()   # per-item backward = O(1) memory,
            total += loss.item()                    # same as harness.py's P2a fix
        self._optimizer.step()
        avg_loss = total / len(self._buffer)

        # [WandB Logging] Log to active WandB run if initialized
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"adajepa/adapt_loss": avg_loss, "adajepa/variant": self.variant})
        except ImportError:
            pass

        return avg_loss
