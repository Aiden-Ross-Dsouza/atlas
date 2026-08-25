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
    regime_config: str | None = None,
    episodes: int = 10,
    num_samples: int = 300,
    iterations: int = 30,
    horizon: int = 6,
    num_act_stepped: int = 6,
    charts_subdir: str = "e0",
    out_subdir: str = "e0_planning",
    log_planner_diagnostics: bool = False,
    min_block_pos_diff: float = 40.0,
) -> None:
    """Defaults = the SUBSTRATE's own validated Push-T config (CEM 300x30,
    horizon 6, num_act_stepped 6 -> 30 raw steps/episode, 1 replan), the
    config dino_wm_pusht reports ~90% SR under -- see
    run_e0_planning.py's module docstring (E0_IMPLEMENTATION_PLAN.md T6).
    charts_subdir/out_subdir let a corrected-config re-run write to e.g.
    atlas_out/e0_planning_v2/ without overwriting earlier (superseded-config,
    kept for the record) results. min_block_pos_diff: minimum required block
    displacement between a sampled real init/goal pair -- unfiltered sampling
    let some pairs draw an already-near-goal state, making success trivial
    without any real pushing (confirmed empirically: 5/10 baseline R0
    episodes finished in <=8 raw steps under the old unfiltered sampler).
    Runs with this filter active must use a NEW out_subdir (e.g.
    e0_planning_v2) -- old results under the same seeds were sampled
    differently and are not resume-compatible with filtered runs."""
    import subprocess
    import sys
    # Volumes are eventually consistent -- reload() picks up commits made by
    # a separate `modal volume put` process before this container started.
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/run_e0_planning.py",
           "--kind", kind,
           "--regime", regime,
           "--episodes", str(episodes),
           "--num-samples", str(num_samples),
           "--iterations", str(iterations),
           "--horizon", str(horizon),
           "--num-act-stepped", str(num_act_stepped),
           "--charts-dir", f"{ATLAS_MOUNT_PATH}/atlas_out/{charts_subdir}",
           "--out-dir", f"{ATLAS_MOUNT_PATH}/atlas_out/{out_subdir}",
           "--min-block-pos-diff", str(min_block_pos_diff)]
    if regime_config is not None:
        cmd += ["--regime-config", regime_config]
    if log_planner_diagnostics:
        cmd.append("--log-planner-diagnostics")
    subprocess.run(cmd, check=True, cwd="/src")
    atlas_volume.commit()


@app.local_entrypoint()
def main(kind: str = "ln_act", regime: str = "R1", regime_config: str | None = None,
          episodes: int = 10,
          num_samples: int = 300, iterations: int = 30, horizon: int = 6,
          num_act_stepped: int = 6, charts_subdir: str = "e0", out_subdir: str = "e0_planning",
          log_planner_diagnostics: bool = False, min_block_pos_diff: float = 40.0) -> None:
    run_e0_planning.remote(kind=kind, regime=regime, regime_config=regime_config, episodes=episodes,
                            num_samples=num_samples, iterations=iterations, horizon=horizon,
                            min_block_pos_diff=min_block_pos_diff,
                            num_act_stepped=num_act_stepped, charts_subdir=charts_subdir,
                            out_subdir=out_subdir, log_planner_diagnostics=log_planner_diagnostics)


@app.function(
    gpu="L4",
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=3600,
)
def diagnose_cem_costs(
    kind: str = "baseline",
    regime: str = "R1",
    seed: int = 0,
    num_samples: int = 300,
    iterations: int = 30,
    horizon: int = 6,
    num_act_stepped: int = 6,
) -> None:
    """scripts/diagnose_cem_costs.py -- captures CEM's per-candidate costs at
    iteration 0 (same RNG seed -> same candidates regardless of chart) and the
    final iteration, to check whether a chart distorts CEM's cost ranking."""
    import subprocess
    import sys
    atlas_volume.reload()
    subprocess.run(
        [sys.executable, "scripts/diagnose_cem_costs.py",
         "--kind", kind,
         "--regime", regime,
         "--seed", str(seed),
         "--num-samples", str(num_samples),
         "--iterations", str(iterations),
         "--horizon", str(horizon),
         "--num-act-stepped", str(num_act_stepped),
         "--charts-dir", f"{ATLAS_MOUNT_PATH}/atlas_out/e0",
         "--out-dir", f"{ATLAS_MOUNT_PATH}/atlas_out/e0_planning/cem_diagnostics"],
        check=True,
        cwd="/src",
    )
    atlas_volume.commit()


@app.local_entrypoint(name="diagnose_cem_costs")
def diagnose_cem_costs_entrypoint(kind: str = "baseline", regime: str = "R1", seed: int = 0,
                                    num_samples: int = 300, iterations: int = 30, horizon: int = 6,
                                    num_act_stepped: int = 6) -> None:
    diagnose_cem_costs.remote(kind=kind, regime=regime, seed=seed, num_samples=num_samples,
                               iterations=iterations, horizon=horizon, num_act_stepped=num_act_stepped)


@app.function(
    gpu="L4",
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=3600 * 6,
)
def run_e0_train(
    kinds: str = "ln_act",
    regimes: str = "R1",
    regime_config: str | None = None,
    steps: int = 2000,
    num_train_trajs: int = 20,
    train_traj_len: int = 25,
    num_val_trajs: int = 8,
    eval_traj_len: int = 50,
    eval_every: int = 25,
    patience: int = 5,
    data_source: str = "dataset",
    data_split: str = "train",
    out_subdir: str = "e0_v2",
) -> None:
    """scripts/run_e0.py -- offline chart fine-tuning, T9: real-data replay
    (data_source='dataset') + early stopping (eval_every/patience) on a held-
    out val split (num_val_trajs), sized for Modal's 24GB L4 (not the 6GB
    local card). out_subdir defaults to e0_v2 so this doesn't overwrite the
    original (pre-T1-fix, invalidated) atlas_out/e0/ charts. regime_config: JSON
    string applied to EVERY regime in `regimes` via set_regime_config -- pass
    the SAME calibrated config used at eval time (run_e0_planning's own
    regime_config), or a chart trains under one physics and gets evaluated
    under another with no error raised anywhere (E0_RECOVERY_PLAN.md P3)."""
    import subprocess
    import sys
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/run_e0.py",
           "--kinds", *kinds.split(","),
           "--regimes", *regimes.split(","),
           "--steps", str(steps),
           "--num-train-trajs", str(num_train_trajs),
           "--train-traj-len", str(train_traj_len),
           "--num-val-trajs", str(num_val_trajs),
           "--eval-traj-len", str(eval_traj_len),
           "--eval-every", str(eval_every),
           "--patience", str(patience),
           "--data-source", data_source,
           "--data-split", data_split,
           "--out", f"{ATLAS_MOUNT_PATH}/atlas_out/{out_subdir}"]
    if regime_config is not None:
        cmd += ["--regime-config", regime_config]
    subprocess.run(cmd, check=True, cwd="/src")
    atlas_volume.commit()


@app.local_entrypoint(name="run_e0_train")
def run_e0_train_entrypoint(kinds: str = "ln_act", regimes: str = "R1",
                              regime_config: str | None = None, steps: int = 2000,
                              num_train_trajs: int = 20, train_traj_len: int = 25,
                              num_val_trajs: int = 8, eval_traj_len: int = 50,
                              eval_every: int = 25, patience: int = 5,
                              data_source: str = "dataset", data_split: str = "train",
                              out_subdir: str = "e0_v2") -> None:
    run_e0_train.remote(kinds=kinds, regimes=regimes, regime_config=regime_config, steps=steps,
                         num_train_trajs=num_train_trajs, train_traj_len=train_traj_len,
                         num_val_trajs=num_val_trajs, eval_traj_len=eval_traj_len,
                         eval_every=eval_every, patience=patience,
                         data_source=data_source, data_split=data_split,
                         out_subdir=out_subdir)
