"""
modal/modal_app.py — Modal GPU image and experiment stubs for ATLAS.

Usage (Modal contributors):
    modal run modal/modal_app.py::run_e0
    modal run modal/modal_app.py::run_e1 --charts atlas_out/e0/ln_act
    modal run modal/modal_app.py::run_e4 --arms frozen adajepa atlas
    modal deploy modal/modal_app.py    # for persistent volumes

Architecture:
  - One Modal Volume mounts to /atlas_root, providing persistent storage for
    checkpoints, datasets, logs, and outputs.
  - The GPU image installs all deps from pyproject.toml.
  - Each experiment is a @app.function wrapping the corresponding script's main().
  - ATLAS_HOME is set to /atlas_root inside the container.
"""

from __future__ import annotations

import modal

# ── Persistent storage ────────────────────────────────────────────────────────
# One volume for all ATLAS data (checkpoints, dataset, logs, outputs).
# Create it with: modal volume create atlas-data
atlas_volume = modal.Volume.from_name("atlas-data", create_if_missing=True)

ATLAS_MOUNT_PATH = "/atlas_root"

# ── GPU image ─────────────────────────────────────────────────────────────────
# Install from pyproject.toml so the Modal image is always in sync with local.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1")
    .pip_install("uv")
    .run_commands(
        # Install atlas and all deps (including jepa-wms git dep) via uv.
        "pip install uv && "
        "uv pip install --system "
        "  torch torchvision --index-url https://download.pytorch.org/whl/cu121 && "
        "uv pip install --system 'atlas-wm @ git+https://github.com/Aiden-Ross-Dsouza/atlas.git'"
    )
    .env({
        "ATLAS_HOME": ATLAS_MOUNT_PATH,
        "JEPAWM_DSET":  f"{ATLAS_MOUNT_PATH}/data",
        "JEPAWM_LOGS":  f"{ATLAS_MOUNT_PATH}/logs",
        "JEPAWM_CKPT":  f"{ATLAS_MOUNT_PATH}/ckpts",
        "ATLAS_OUT":    f"{ATLAS_MOUNT_PATH}/atlas_out",
        "TORCH_HOME":   f"{ATLAS_MOUNT_PATH}/hub",
        "MUJOCO_GL":    "egl",
        "PYOPENGL_PLATFORM": "egl",
    })
)

app = modal.App("atlas-wm", image=image)

VOLUME_MOUNTS = {ATLAS_MOUNT_PATH: atlas_volume}


# ── Experiment stubs ──────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=3600 * 4,   # 4 hours (~3 GPU-h with headroom)
)
def run_e0(
    kinds: list[str] = ("ln_act", "lora4", "full"),
    regimes: list[str] = ("R1", "R2"),
    steps: int = 2000,
) -> None:
    """E0: Adapter capacity. ~3 GPU-h on A10G."""
    import subprocess, sys
    subprocess.run(
        [sys.executable, "scripts/run_e0.py",
         "--kinds", *kinds,
         "--regimes", *regimes,
         "--steps", str(steps)],
        check=True,
        cwd="/atlas_root/src",   # adjust to wherever the package is installed
    )
    atlas_volume.commit()


@app.function(
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=3600 * 6,   # 6 hours (~4 GPU-h with headroom)
)
def run_e1(
    charts_dir: str = f"{ATLAS_MOUNT_PATH}/atlas_out/e0/ln_act",
    routers: list[str] = ("umf", "e1", "sdyn", "random", "oracle_id"),
    episodes: int = 60,
    seeds: int = 3,
) -> None:
    """E1: Fitness routing — THE GATE. ~4 GPU-h on A10G."""
    import subprocess, sys
    subprocess.run(
        [sys.executable, "scripts/run_e1.py",
         "--charts", charts_dir,
         "--routers", *routers,
         "--episodes", str(episodes),
         "--seeds", str(seeds)],
        check=True,
    )
    atlas_volume.commit()


@app.function(
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=3600 * 30,  # 30 hours (~21 GPU-h with headroom)
)
def run_e4(
    arms: list[str] | None = None,
    episodes: int = 20,
    seeds: int = 3,
) -> None:
    """E4+E3: Continual stream + expansion ablation. ~21 GPU-h on A10G."""
    import subprocess, sys
    arm_args = arms or [
        "frozen", "adajepa", "adajepa_persist",
        "atlas_fixed", "atlas_detect", "atlas", "oracle_id",
    ]
    subprocess.run(
        [sys.executable, "scripts/run_e4.py",
         "--arms", *arm_args,
         "--episodes", str(episodes),
         "--seeds", str(seeds)],
        check=True,
    )
    atlas_volume.commit()


@app.function(
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=3600 * 8,   # 8 hours (~6 GPU-h with headroom)
)
def run_e2(
    cells: list[str] = ("A", "B", "C", "D"),
    routers: list[str] = ("umf", "sdyn"),
    episodes: int = 40,
    seeds: int = 3,
) -> None:
    """E2: Appearance vs dynamics 2×2. ~6 GPU-h on A10G."""
    import subprocess, sys
    subprocess.run(
        [sys.executable, "scripts/run_e2.py",
         "--cells", *cells,
         "--routers", *routers,
         "--episodes", str(episodes),
         "--seeds", str(seeds)],
        check=True,
    )
    atlas_volume.commit()


@app.function(
    gpu="A10G",
    volumes=VOLUME_MOUNTS,
    timeout=3600 * 5,
)
def download_data() -> None:
    """Download checkpoints and Push-T dataset into the Modal volume."""
    import subprocess, sys
    subprocess.run([sys.executable, "scripts/download_data.py"], check=True)
    atlas_volume.commit()


@app.local_entrypoint()
def main() -> None:
    print("ATLAS Modal App — available functions:")
    print("  modal run modal/modal_app.py::download_data")
    print("  modal run modal/modal_app.py::run_e0")
    print("  modal run modal/modal_app.py::run_e1 --charts <path>")
    print("  modal run modal/modal_app.py::run_e2")
    print("  modal run modal/modal_app.py::run_e4")
