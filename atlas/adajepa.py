"""
adajepa.py — AdaJEPA reimplemented inside the jepa-wms substrate.

Implements two variants:
  AdaJEPA              — per-episode re-init from pretrained weights; 1 SGD
                         step per replan on 5 most-recent transitions.
  PersistentAdaJEPA    — same, but NO per-episode re-init (cross-episode
                         retention). Labelled as our modification in the paper.

Both use:
  - predictor lr 5e-4, Adam (AdaJEPA published hyperparameters)
  - buffer of 5 most recent (encoder_output, actions) transitions
  - encoder FROZEN (= predlast+encfrozen variant)
  - loss: mean L2 latent prediction error (AdaJEPA Eq. 1)
  - exactly 1 gradient step per replan

Prequential invariant: refinement ALWAYS occurs AFTER scoring.
Callers are responsible for calling refine() after score/select.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import torch
import torch.optim as optim
import torch.nn as nn

AdaVariant = Literal["adajepa", "persistent"]


class AdaJEPA:
    """
    AdaJEPA adaptation controller.

    Usage (per-replan, in order — NEVER reverse):
        1. adapter.score(chunk)    → used by caller for routing / expansion
        2. caller selects and plans
        3. adapter.refine(chunk)   → 1 SGD step on current adapter state

    For AdaJEPA (not persistent): call adapter.reset() at episode start.
    """

    BUFFER_SIZE = 5

    def __init__(
        self,
        predictor: nn.Module,
        pretrained_state: dict[str, torch.Tensor],
        variant: AdaVariant = "adajepa",
        lr: float = 5e-4,
    ) -> None:
        """
        Args:
            predictor:        The JEPA predictor (encoder must already be frozen).
            pretrained_state: state_dict of the predictor at initialisation.
                              AdaJEPA resets to this on episode start.
            variant:          'adajepa' resets per-episode; 'persistent' does not.
            lr:               Learning rate (AdaJEPA default 5e-4).
        """
        self.predictor = predictor
        self.pretrained_state = {k: v.clone() for k, v in pretrained_state.items()}
        self.variant = variant
        self.lr = lr
        self._buffer: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=self.BUFFER_SIZE)
        self._optimizer = optim.Adam(
            [p for p in predictor.parameters() if p.requires_grad], lr=lr
        )

    def reset(self) -> None:
        """
        Called at the start of each episode.
        For 'adajepa': resets predictor to pretrained weights and clears buffer.
        For 'persistent': no-op (cross-episode retention is the point).
        """
        if self.variant == "adajepa":
            self.predictor.load_state_dict(self.pretrained_state, strict=False)
            self._buffer.clear()
            self._optimizer = optim.Adam(
                [p for p in self.predictor.parameters() if p.requires_grad], lr=self.lr
            )

    def push(self, encoder_output: torch.Tensor, actions: torch.Tensor) -> None:
        """Add a (encoder_output, actions) chunk to the rolling buffer."""
        self._buffer.append((encoder_output.detach(), actions.detach()))

    def refine(self) -> float:
        """
        One AdaJEPA gradient step on the buffer contents.
        Must be called AFTER scoring and planning.

        Returns:
            The scalar loss value for logging.
        """
        if not self._buffer:
            return 0.0

        self._optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for enc_out, actions in self._buffer:
            T = actions.shape[0]
            z = enc_out
            z_cur = z[0].unsqueeze(0)
            loss = torch.tensor(0.0, device=z_cur.device)
            for t in range(T):
                a_t = actions[t].unsqueeze(0)
                z_next_hat = self.predictor(z_cur, a_t)
                loss = loss + (z_next_hat - z[t + 1].unsqueeze(0)).pow(2).mean()
                z_cur = z_next_hat.detach()
            total_loss = total_loss + loss / T

        total_loss = total_loss / len(self._buffer)
        total_loss.backward()
        self._optimizer.step()
        loss_val = total_loss.item()

        # [WandB Logging] Log to active WandB run if initialized
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"adajepa/adapt_loss": loss_val})
        except ImportError:
            pass

        return loss_val
