"""
regimes.py — Physics regime wrappers and visual corruption wrappers for Push-T.

Three dynamics regimes (primary, matched appearance):
  R0  default — shipped parameters
  R1  light block — T-block mass × 0.2 (mass, moment both scaled)
  R2  high damping — space.damping decreased (pymunk damping is a RETENTION factor)

Visual corruptions (E2 only — appearance differs, dynamics unchanged):
  blur, salt_pepper, dark, colour_change
  Ported from AdaJEPA (github.com/agentic-learning-ai-lab/adajepa).

CRITICAL: _apply_physics() is called inside reset() because many pymunk envs
rebuild the space on reset, which resets physics parameters.
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np

RegimeName = Literal["R0", "R1", "R2"]

# Physics parameter values — exact values from the plan.
REGIME_CONFIGS: dict[str, dict] = {
    "R0": {},                              # default: no modifications
    "R1": {"mass_scale": 0.2},            # light block: mass × 0.2
    "R2": {"damping": 0.3},               # high damping: space.damping = 0.3 (from 0.9 default)
}


class PhysicsRegime(gym.Wrapper):
    """
    Gymnasium wrapper that applies physics modifications to a pymunk-based env.

    Args:
        env:          The base Push-T environment.
        regime:       One of 'R0', 'R1', 'R2'.

    Raises:
        ValueError:   If regime is not recognised.
        RuntimeError: If the expected pymunk attributes are not found on the env
                      (catches wrong env or changed internals early).
    """

    def __init__(self, env: gym.Env, regime: RegimeName) -> None:
        super().__init__(env)
        if regime not in REGIME_CONFIGS:
            raise ValueError(
                f"Unknown regime {regime!r}. Valid options: {list(REGIME_CONFIGS.keys())}"
            )
        self.regime = regime
        self._cfg = REGIME_CONFIGS[regime]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_physics()
        return obs, info

    def _apply_physics(self) -> None:
        """Apply physics modifications. Called after every reset."""
        if not self._cfg:
            return  # R0: no modifications

        # Locate the pymunk space. Try common attribute paths used in DINO-WM / Push-T.
        space = self._get_space()

        if "mass_scale" in self._cfg:
            block = self._get_block()
            original_mass = getattr(block, "_original_mass", None)
            original_moment = getattr(block, "_original_moment", None)
            if original_mass is None:
                # First time: cache the original values.
                block._original_mass = block.mass
                block._original_moment = block.moment
                original_mass = block._original_mass
                original_moment = block._original_moment
            block.mass = original_mass * self._cfg["mass_scale"]
            block.moment = original_moment * self._cfg["mass_scale"]

        if "damping" in self._cfg:
            space.damping = self._cfg["damping"]

    def _get_space(self):
        """Locate the pymunk space object through common env attribute paths."""
        for attr_path in ("env.space", "space", "env.env.space", "unwrapped.space"):
            obj = self.env
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        raise RuntimeError(
            "Could not locate pymunk Space on the environment. "
            "Check the env's attribute structure with `vars(env.unwrapped)` and "
            "update _get_space() in regimes.py."
        )

    def _get_block(self):
        """Locate the T-block body."""
        for attr_path in ("env.block", "block", "env.env.block",
                          "unwrapped.block", "env.tee", "unwrapped.tee"):
            obj = self.env
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        raise RuntimeError(
            "Could not locate T-block body on the environment. "
            "Check the env's attribute structure with `vars(env.unwrapped)` and "
            "update _get_block() in regimes.py."
        )


# ── Visual corruptions (E2 only) ───────────────────────────────────────────────

CorruptionKind = Literal["blur", "salt_pepper", "dark", "colour_change", "none"]


class VisualCorruption(gym.ObservationWrapper):
    """
    Observation wrapper that applies a fixed visual corruption while leaving
    physics completely unchanged (appearance differs, dynamics same — E2 Cell C/D).

    Args:
        env:       Base environment.
        kind:      Type of corruption to apply.
        severity:  Strength parameter (interpretation depends on kind).
    """

    def __init__(self, env: gym.Env, kind: CorruptionKind, severity: float = 0.5) -> None:
        super().__init__(env)
        if kind not in ("blur", "salt_pepper", "dark", "colour_change", "none"):
            raise ValueError(f"Unknown corruption kind: {kind!r}")
        self.kind = kind
        self.severity = severity

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.kind == "none":
            return obs
        if obs.ndim == 1:
            # Flattened; return as-is (not an image).
            return obs
        return _corrupt(obs, self.kind, self.severity)


def _corrupt(img: np.ndarray, kind: CorruptionKind, severity: float) -> np.ndarray:
    """Apply a visual corruption to a uint8 image array [..., H, W, C] or [H, W, C]."""
    import cv2

    img = img.copy()
    if kind == "blur":
        ksize = max(1, int(severity * 20) | 1)  # must be odd
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    elif kind == "salt_pepper":
        rng = np.random.default_rng()
        mask = rng.random(img.shape[:2])
        img[mask < severity / 2] = 0
        img[mask > 1 - severity / 2] = 255

    elif kind == "dark":
        img = (img * (1.0 - severity * 0.8)).clip(0, 255).astype(np.uint8)

    elif kind == "colour_change":
        # Hue shift via HSV.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(severity * 90)) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return img


def build_env(
    base_env: gym.Env,
    regime: RegimeName = "R0",
    corruption: CorruptionKind = "none",
    corruption_severity: float = 0.5,
) -> gym.Env:
    """
    Compose a Push-T environment with the specified regime and visual corruption.

    Args:
        base_env:             The raw Push-T gym environment (already instantiated).
        regime:               Physics regime to apply.
        corruption:           Visual corruption (E2 only; use 'none' otherwise).
        corruption_severity:  Severity parameter for the corruption.

    Returns:
        Wrapped environment ready for episodes.
    """
    env = PhysicsRegime(base_env, regime)
    if corruption != "none":
        env = VisualCorruption(env, corruption, corruption_severity)
    return env
