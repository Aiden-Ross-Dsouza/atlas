"""
scripts/modal_run_e0.py — Launch E0 experiment on Modal GPU.

Usage:
    modal run scripts/modal_run_e0.py
    modal run scripts/modal_run_e0.py --steps 500
"""

from pathlib import Path
import sys
import os

# Bypass local folder shadowing of the installed 'modal' package
pkg_dir = Path(__file__).parent.resolve()
parent_dir = pkg_dir.parent.resolve()

cleaned_sys_path = []
for p in sys.path:
    if not p or p == ".":
        continue
    try:
        resolved = Path(p).resolve()
        if resolved in (parent_dir, pkg_dir):
            continue
    except Exception:
        pass
    cleaned_sys_path.append(p)

sys.path = cleaned_sys_path
import modal

sys.path.insert(0, str(parent_dir))

app = modal.App("atlas-e0")

local_atlas_dir = parent_dir

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "numpy>=1.26",
        "scipy>=1.13",
        "pandas>=2.2",
        "statsmodels>=0.14",
        "omegaconf>=2.3",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "matplotlib>=3.9",
        "gymnasium>=0.29",
        "gym",
        "pymunk>=6.8",
        "imageio>=2.34",
        "opencv-python-headless>=4.9",
        "einops",
        "decord",
        "httpx",
        "huggingface_hub",
        "jepa-wms @ git+https://github.com/facebookresearch/jepa-wms.git",
    )
    .add_local_python_source("atlas", copy=True)
    .add_local_dir(
        str(local_atlas_dir / "scripts"),
        remote_path="/root/atlas/scripts",
        ignore=["__pycache__", "*.pyc"],
        copy=True,
    )
    .add_local_dir(
        str(local_atlas_dir / "configs"),
        remote_path="/root/atlas/configs",
        ignore=["__pycache__"],
        copy=True,
    )
    .add_local_dir(
        str(local_atlas_dir / "hub"),
        remote_path="/root/atlas/hub",
        ignore=["__pycache__", "*.pth.tar", "*.pth"],
        copy=True,
    )
)


volume = modal.Volume.from_name("atlas-e0-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={"/root/atlas/atlas_out/e0": volume},
)
def run_e0_remote(kinds: list[str], regimes: list[str], steps: int, use_wandb: bool = False, wandb_api_key: str | None = None):
    import os
    import sys

    os.chdir("/root/atlas")
    if "/root/atlas" not in sys.path:
        sys.path.insert(0, "/root/atlas")

    import subprocess
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key

    # [Debug print statement] Unbuffered Python execution for real-time Modal logging
    cmd = [
        sys.executable,
        "-u",
        "-X", "utf8",
        "scripts/run_e0.py",
        "--kinds", *kinds,
        "--regimes", *regimes,
        "--steps", str(steps),
    ]
    if use_wandb:
        cmd.append("--wandb")

    print(f"Executing: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        # Always commit completed output files to the persistent volume
        volume.commit()

    # Return generated files (text and binary plots) back to local machine
    results_dir = Path("/root/atlas/atlas_out/e0")
    files = {}
    if results_dir.exists():
        for p in results_dir.glob("*"):
            if p.is_file():
                if p.suffix in [".json", ".md"]:
                    files[p.name] = ("text", p.read_text(encoding="utf-8"))
                elif p.suffix in [".png", ".pdf", ".pt"]:
                    import base64
                    files[p.name] = ("binary", base64.b64encode(p.read_bytes()).decode("ascii"))
    return files


@app.local_entrypoint()
def main(steps: int = 2000, wandb: bool = False):
    import os
    import netrc
    import base64

    wandb_key = os.environ.get("WANDB_API_KEY", None)
    if not wandb_key and wandb:
        try:
            for p in [Path.home() / "_netrc", Path.home() / ".netrc"]:
                if p.exists():
                    auth = netrc.netrc(str(p)).authenticators("api.wandb.ai")
                    if auth:
                        wandb_key = auth[2]
                        break
        except Exception:
            pass

    print(f"🚀 Launching E0 experiment on Modal (T4 GPU, steps={steps}, wandb={wandb})...")
    remote_files = run_e0_remote.remote(
        kinds=["ln_act", "lora4", "full"],
        regimes=["R1", "R2"],
        steps=steps,
        use_wandb=wandb,
        wandb_api_key=wandb_key,
    )
    
    # Save the files locally
    local_out = Path(__file__).parent.parent / "atlas_out" / "e0"
    local_out.mkdir(parents=True, exist_ok=True)
    
    if remote_files:
        for filename, (file_type, data) in remote_files.items():
            out_file = local_out / filename
            if file_type == "text":
                out_file.write_text(data, encoding="utf-8")
            else:
                out_file.write_bytes(base64.b64decode(data))
            print(f"💾 Saved {filename} to {out_file}")
        
        if "results.md" in remote_files:
            print("\n📊 E0 Results (Markdown):")
            print(remote_files["results.md"][1])
    
    print("\n✅ Modal E0 run completed successfully!")
