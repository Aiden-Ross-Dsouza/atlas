"""
modal/modal_e0_planning.py — Modal GPU runner for scripts/run_e0_planning.py.

Separate from modal_app.py because that file's image installs atlas-wm from
a stale GitHub remote with none of this session's fixes (code-review.md Bugs
#1-#7) or run_e0_planning.py itself. This image builds from the local repo.

Usage:
    modal volume create atlas-data   # if not already created
    modal volume put atlas-data atlas_out/e0 /atlas_root/atlas_out/e0  # upload local charts
    modal run modal/modal_e0_planning.py --kind ln_act --regime R1 --episodes 10

Needs the checkpoint/dataset in the volume already (see modal_app.py::download_data),
plus the E0 charts uploaded as above (they were trained locally, not on Modal).
"""

from __future__ import annotations

from pathlib import Path

import modal

atlas_volume = modal.Volume.from_name("atlas-data", create_if_missing=True)
ATLAS_MOUNT_PATH = "/atlas_root"
REPO_ROOT = Path(__file__).parent.parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1")
    .pip_install("uv")
    .add_local_dir(
        # /src, NOT under /atlas_root: that's the volume mount point, and Modal
        # refuses to mount a volume onto a path that already has image content.
        str(REPO_ROOT), remote_path="/src", copy=True,
        ignore=[".venv", ".git", "data", "hub", "atlas_out", "graphify-out", "__pycache__", "logs"],
    )
    .run_commands(
        # vendor/jepa-wms installed explicitly first: pyproject.toml deliberately
        # does not list it as a dependency (uv can't resolve a relative `file:`
        # path for a dependency of this project's own wheel, and a hardcoded
        # absolute path isn't portable across machines/containers).
        "cd /src && uv pip install --system -e vendor/jepa-wms && "
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu121 && "
        "uv pip install --system -e ."
    )
    .env({
        "ATLAS_HOME": ATLAS_MOUNT_PATH,
        "JEPAWM_DSET": f"{ATLAS_MOUNT_PATH}/data",
        "JEPAWM_LOGS": f"{ATLAS_MOUNT_PATH}/logs",
        "JEPAWM_CKPT": f"{ATLAS_MOUNT_PATH}/ckpts",
        "ATLAS_OUT": f"{ATLAS_MOUNT_PATH}/atlas_out",
        "TORCH_HOME": f"{ATLAS_MOUNT_PATH}/hub",
    })
)

app = modal.App("atlas-e0-planning", image=image)


@app.function(
    gpu="L4",  # 24GB @ $0.80/h -- best ROI: comfortable headroom over the ~13.5GB
               # measured need, well under L40S ($1.95/h)/A100 pricing. Avoid "T4"
               # (16GB, too close to the measured requirement to risk OOM).
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=3600 * 6,
)
def run_e0_planning(
    kind: str = "ln_act",
    regime: str = "R1",
    episodes: int = 10,
    num_samples: int = 300,
    iterations: int = 30,
    horizon: int = 6,
) -> None:
    """~14.4 min/episode extrapolated from a 6GB local GPU at reduced
    settings -- untested at full spec, may be faster on L40S/A100."""
    import subprocess
    import sys
    # Volumes are eventually consistent -- reload() picks up commits made by
    # a separate `modal volume put` process before this container started.
    atlas_volume.reload()
    subprocess.run(
        [sys.executable, "scripts/run_e0_planning.py",
         "--kind", kind,
         "--regime", regime,
         "--episodes", str(episodes),
         "--num-samples", str(num_samples),
         "--iterations", str(iterations),
         "--horizon", str(horizon),
         "--charts-dir", f"{ATLAS_MOUNT_PATH}/atlas_out/e0"],
        check=True,
        cwd="/src",
    )
    atlas_volume.commit()


@app.local_entrypoint()
def main(kind: str = "ln_act", regime: str = "R1", episodes: int = 10,
          num_samples: int = 300, iterations: int = 30, horizon: int = 6) -> None:
    run_e0_planning.remote(kind=kind, regime=regime, episodes=episodes,
                            num_samples=num_samples, iterations=iterations, horizon=horizon)
