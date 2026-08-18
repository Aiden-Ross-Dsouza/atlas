"""
scripts/download_data.py — Download DINO-WM Push-T checkpoint and dataset.

Downloads:
  1. dino_wm_pusht checkpoint via torch.hub.load (stored in TORCH_HOME/hub)
  2. Push-T dataset via the jepa-wms download utility (stored in JEPAWM_DSET)

No HF login required. Checkpoints are public via GitHub + fbaipublicfiles CDN.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --skip-dataset     # checkpoint only
    python scripts/download_data.py --skip-checkpoint  # dataset only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import atlas  # sets TORCH_HOME and creates directories


def download_checkpoint() -> None:
    """Load dino_wm_pusht via torch.hub — downloads to TORCH_HOME if not cached."""
    import torch

    print("Downloading dino_wm_pusht checkpoint via torch.hub...")
    print(f"  TORCH_HOME = {atlas.ATLAS_HOME / 'hub'}")
    model, prep = torch.hub.load(
        "facebookresearch/jepa-wms",
        "dino_wm_pusht",
        force_reload=False,
        trust_repo=True,
    )
    print(f"  OK — encoder: {type(model.encoder).__name__}, "
          f"predictor: {type(model.predictor).__name__}")

    # Save a local copy for faster re-loading without going through TorchHub.
    ckpt_path = atlas.CKPT_DIR / "dino_wm_pusht.pth.tar"
    if not ckpt_path.exists():
        import torch
        torch.save({
            "encoder": model.encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
        }, ckpt_path)
        print(f"  Saved local copy: {ckpt_path}")
    else:
        print(f"  Local copy already exists: {ckpt_path}")


def download_dataset() -> None:
    """Download the Push-T dataset using the jepa-wms download utility."""
    import subprocess

    dataset_dir = atlas.DATA_DIR / "pusht_noise"
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir} — skipping download.")
        return

    print(f"Downloading Push-T dataset to {atlas.DATA_DIR}...")
    # jepa-wms ships a download script; invoke it with the correct destination.
    try:
        import jepa_wms  # noqa: F401
        jepa_wms_root = Path(jepa_wms.__file__).parent.parent
        dl_script = jepa_wms_root / "src" / "scripts" / "download_data.py"
        if not dl_script.exists():
            raise FileNotFoundError(f"jepa-wms download script not found at {dl_script}")
        subprocess.run(
            [sys.executable, str(dl_script), "--dataset", "pusht",
             "--out", str(atlas.DATA_DIR)],
            check=True,
        )
    except ImportError:
        raise ImportError(
            "jepa-wms is not installed. Run: uv pip install -e . "
            "to install it from pyproject.toml."
        )
    print(f"  Dataset saved to {atlas.DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ATLAS checkpoints and data.")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()

    if not args.skip_checkpoint:
        download_checkpoint()
    if not args.skip_dataset:
        download_dataset()

    print("\nDownload complete.")
    print(f"  Checkpoints : {atlas.CKPT_DIR}")
    print(f"  Dataset     : {atlas.DATA_DIR}")
    print(f"  Hub cache   : {atlas.ATLAS_HOME / 'hub'}")


if __name__ == "__main__":
    main()
