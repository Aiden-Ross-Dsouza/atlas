"""
modal/modal_e4.py — Modal GPU runner for scripts/run_e4.py.

Structurally mirrors modal_e0_planning.py: same volume (atlas-data), same
/atlas_root mount, same /src local-dir image, same
`uv pip install -e vendor/jepa-wms` -> torch cu121 -> `-e .` chain, same
.env({...}) block, gpu="L4", timeout=3600*6.

`arm` and `seed_run` are exposed as SINGLE-VALUED args (not lists) so the
7-arm x N-seed grid fans out across separate containers -- this is
E3_E4_IMPLEMENTATION_PLAN.md Phase 0.2's cut #4 (7-21x wall-clock reduction
at the SAME total GPU-h) and the only reason the full run finishes in a
reasonable wall-clock time. Modal's CLI can't pass Python lists directly, so
--arms/--seeds here are comma-separated strings, split before dispatch --
same pattern as modal_e0_planning.py's run_e0_train (kinds/regimes).

Usage:
    modal volume create atlas-data   # if not already created
    modal run --detach modal/modal_e4.py --arm frozen --seed-run 0 \\
        --episodes 10 --iterations 10 --segment-regimes R0,R2 \\
        --charts-subdir e0_v3_dataset --kind ln_act --out-subdir e4_v1

    # fan out all 7 arms, one container each:
    for arm in frozen adajepa adajepa_persist atlas_fixed atlas_detect atlas oracle_id; do
        modal run --detach modal/modal_e4.py --arm $arm --seed-run 0 \\
            --episodes 10 --iterations 10 --segment-regimes R0,R2 \\
            --charts-subdir e0_v3_dataset --kind ln_act --out-subdir e4_v1
    done

    # after all containers finish, merge their per-(arm,seed_run) JSONL files:
    modal run modal/modal_e4.py --merge --out-subdir e4_v1

Needs the checkpoint/dataset in the volume already (see modal_app.py::download_data),
plus the E0 charts uploaded (they are trained locally/on a separate Modal run,
not by this file) -- see modal_e0_planning.py's run_e0_train.
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
        str(REPO_ROOT), remote_path="/src", copy=True,
        ignore=[".venv", ".git", "data", "hub", "atlas_out", "graphify-out", "__pycache__", "logs"],
    )
    .run_commands(
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

app = modal.App("atlas-e4", image=image)


@app.function(
    gpu="L4",
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=3600 * 6,
)
def run_e4(
    arm: str = "frozen",
    seed_run: int = 0,
    episodes: int = 10,
    num_samples: int = 300,
    iterations: int = 10,
    horizon: int = 6,
    num_act_stepped: int = 1,
    max_mpc_steps: int = 30,
    segment_regimes: str = "R0,R2",
    charts_subdir: str = "e0",
    kind: str = "ln_act",
    expansion_start_library: str = "full",
    out_subdir: str = "e4",
) -> None:
    """Runs ONE (arm, seed_run) pair -- writes episodes_{arm}_{seed_run}.jsonl
    into out_subdir so N parallel containers (one per arm) never write-race
    the same file. Merge with `--merge` after all containers finish."""
    import subprocess
    import sys

    atlas_volume.reload()
    regime_a, regime_b = segment_regimes.split(",")
    out_dir = f"{ATLAS_MOUNT_PATH}/atlas_out/{out_subdir}"
    # run_e4.py always writes/reads episodes.jsonl in --out; point each
    # container at its OWN subdirectory, then move the file to the shared
    # out_subdir under a per-(arm,seed_run) name so containers never collide.
    container_out = f"{out_dir}/_containers/{arm}_{seed_run}"
    cmd = [sys.executable, "scripts/run_e4.py",
           "--arms", arm,
           "--seeds", "1",   # this container runs exactly ONE seed_run
           # FIX_SPEC.md B12: previously run_e4.py always ran LOCAL
           # seed_run=0 regardless of which seed_run this container was
           # launched for (get_stream(..., seeds=1) only ever generated the
           # seed_run=0 episode/init/goal stream), and the seed_run field
           # was only relabelled AFTER THE FACT below -- a "3-seed" sweep
           # produced bit-identical episode data under 3 different labels.
           # --seed-run-offset makes run_e4.py generate and run the REAL
           # seed_run's stream (and seed the CEM planner's own local_seed
           # from it too), so the relabelling below is now confirmatory,
           # not load-bearing.
           "--seed-run-offset", str(seed_run),
           "--episodes", str(episodes),
           "--num-samples", str(num_samples),
           "--iterations", str(iterations),
           "--horizon", str(horizon),
           "--num-act-stepped", str(num_act_stepped),
           "--max-mpc-steps", str(max_mpc_steps),
           "--segment-regimes", regime_a, regime_b,
           "--charts", f"{ATLAS_MOUNT_PATH}/atlas_out/{charts_subdir}",
           "--kind", kind,
           "--expansion-start-library", expansion_start_library,
           "--out", container_out]
    subprocess.run(cmd, check=True, cwd="/src")

    # Copy this container's episodes.jsonl into the shared out_subdir under a
    # (arm, seed_run)-qualified name -- NOTE: run_e4.py's --seeds 1 always
    # writes seed_run=0 internally; relabel to the REAL seed_run requested
    # here so merged records carry the correct seed_run.
    import json
    src = Path(container_out) / "episodes.jsonl"
    dst = Path(out_dir) / f"episodes_{arm}_{seed_run}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        with open(src) as f_in, open(dst, "w") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["seed_run"] = seed_run
                f_out.write(json.dumps(rec) + "\n")
    atlas_volume.commit()


@app.local_entrypoint()
def main(arm: str = "frozen", seed_run: int = 0, episodes: int = 10,
          num_samples: int = 300, iterations: int = 10, horizon: int = 6,
          num_act_stepped: int = 1, max_mpc_steps: int = 30,
          segment_regimes: str = "R0,R2", charts_subdir: str = "e0",
          kind: str = "ln_act", expansion_start_library: str = "full",
          out_subdir: str = "e4", merge: bool = False) -> None:
    if merge:
        merge_episodes.remote(out_subdir=out_subdir)
        return
    run_e4.remote(arm=arm, seed_run=seed_run, episodes=episodes,
                   num_samples=num_samples, iterations=iterations, horizon=horizon,
                   num_act_stepped=num_act_stepped, max_mpc_steps=max_mpc_steps,
                   segment_regimes=segment_regimes, charts_subdir=charts_subdir,
                   kind=kind, expansion_start_library=expansion_start_library,
                   out_subdir=out_subdir)


@app.function(volumes={ATLAS_MOUNT_PATH: atlas_volume}, timeout=600)
def merge_episodes(out_subdir: str = "e4") -> None:
    """Concatenates every episodes_{arm}_{seed_run}.jsonl in out_subdir into
    one episodes.jsonl -- the file scripts/make_tables.py / make_figures.py
    expect."""
    atlas_volume.reload()
    out_dir = Path(f"{ATLAS_MOUNT_PATH}/atlas_out/{out_subdir}")
    parts = sorted(out_dir.glob("episodes_*.jsonl"))
    merged_path = out_dir / "episodes.jsonl"
    n = 0
    with open(merged_path, "w") as out_f:
        for part in parts:
            with open(part) as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)
                        n += 1
    print(f"Merged {len(parts)} file(s), {n} record(s) -> {merged_path}")
    atlas_volume.commit()
