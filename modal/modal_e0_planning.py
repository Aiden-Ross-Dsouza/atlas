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

from tqdm import tqdm

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
    max_agent_block_dist: float | None = None,
    episode_start: int = 0,
    out_suffix: str = "",
    log_umf: bool = True,
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
    differently and are not resume-compatible with filtered runs.
    max_agent_block_dist (P2d, E0_RECOVERY_PLAN.md): rejects init states the
    agent cannot plausibly reach within the step budget -- None (default)
    defers to run_e0_planning.py's own DEFAULT_MAX_AGENT_BLOCK_DIST rather
    than duplicating that data-derived constant here.
    episode_start/out_suffix: for splitting one N-episode request across
    multiple concurrent Modal calls (e.g. episode_start=0/episodes=50 and
    episode_start=50/episodes=100, out_suffix='_shard0'/'_shard1') -- see
    scripts/merge_planning_shards.py to combine them afterward. log_umf:
    on by default -- logs per-replan UMF of the executed chunk alongside
    success, giving an episode-level (UMF, success) pair for free."""
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
           "--min-block-pos-diff", str(min_block_pos_diff),
           "--episode-start", str(episode_start),
           "--out-suffix", out_suffix]
    if regime_config is not None:
        cmd += ["--regime-config", regime_config]
    if max_agent_block_dist is not None:
        cmd += ["--max-agent-block-dist", str(max_agent_block_dist)]
    if log_planner_diagnostics:
        cmd.append("--log-planner-diagnostics")
    if not log_umf:
        cmd.append("--no-log-umf")
    subprocess.run(cmd, check=True, cwd="/src")
    atlas_volume.commit()


@app.function(
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=600,
)
def merge_shards(kind: str, regime: str, out_subdir: str, shards: list[str]) -> None:
    """Runs scripts/merge_planning_shards.py inside a container with the
    volume mounted, so combining shard outputs needs no local download."""
    import subprocess
    import sys
    atlas_volume.reload()
    subprocess.run(
        [sys.executable, "scripts/merge_planning_shards.py",
         "--kind", kind, "--regime", regime,
         "--out-dir", f"{ATLAS_MOUNT_PATH}/atlas_out/{out_subdir}",
         "--shards", *shards],
        check=True, cwd="/src",
    )
    atlas_volume.commit()


@app.local_entrypoint()
def main(kind: str = "ln_act", regime: str = "R1", regime_config: str | None = None,
          episodes: int = 10,
          num_samples: int = 300, iterations: int = 30, horizon: int = 6,
          num_act_stepped: int = 6, charts_subdir: str = "e0", out_subdir: str = "e0_planning",
          log_planner_diagnostics: bool = False, min_block_pos_diff: float = 40.0,
          max_agent_block_dist: float | None = None,
          episode_start: int = 0, out_suffix: str = "", log_umf: bool = True,
          num_shards: int = 1) -> None:
    """num_shards > 1: splits [episode_start, episodes) into that many
    contiguous, near-equal ranges, launches each as its own CONCURRENT Modal
    container (via .spawn(), not sequential .remote() calls), waits for all
    to finish, then merges them into the canonical {kind}_{regime}.jsonl via
    merge_shards(). This is the actual wall-clock lever on a deadline -- the
    workload is a sequential per-episode CEM loop on a small model (not
    GPU-flop-bound at this batch size), so N containers in parallel beats a
    single faster GPU. E.g. --episodes 100 --num-shards 4 runs four L4s
    concurrently instead of one L4 for 4x as long, for the same total cost."""
    if num_shards <= 1:
        run_e0_planning.remote(kind=kind, regime=regime, regime_config=regime_config, episodes=episodes,
                                num_samples=num_samples, iterations=iterations, horizon=horizon,
                                min_block_pos_diff=min_block_pos_diff,
                                max_agent_block_dist=max_agent_block_dist,
                                num_act_stepped=num_act_stepped, charts_subdir=charts_subdir,
                                out_subdir=out_subdir, log_planner_diagnostics=log_planner_diagnostics,
                                episode_start=episode_start, out_suffix=out_suffix, log_umf=log_umf)
        return

    total = episodes - episode_start
    if total <= 0:
        raise ValueError(f"episodes ({episodes}) must exceed episode_start ({episode_start}).")
    base, rem = divmod(total, num_shards)
    bounds = []
    start = episode_start
    for i in range(num_shards):
        size = base + (1 if i < rem else 0)  # first `rem` shards get one extra episode
        if size == 0:
            continue  # more shards requested than episodes to cover
        bounds.append((start, start + size))
        start += size

    print(f"Splitting episodes [{episode_start},{episodes}) into {len(bounds)} shard(s): {bounds}")
    shard_suffixes = [f"_shard{i}" for i in range(len(bounds))]
    calls = [
        run_e0_planning.spawn(
            kind=kind, regime=regime, regime_config=regime_config, episodes=e,
            num_samples=num_samples, iterations=iterations, horizon=horizon,
            min_block_pos_diff=min_block_pos_diff, max_agent_block_dist=max_agent_block_dist,
            num_act_stepped=num_act_stepped, charts_subdir=charts_subdir, out_subdir=out_subdir,
            log_planner_diagnostics=log_planner_diagnostics,
            episode_start=s, out_suffix=suffix, log_umf=log_umf,
        )
        for (s, e), suffix in zip(bounds, shard_suffixes)
    ]
    # Containers run concurrently (already launched via .spawn() above); this
    # loop only blocks LOCALLY waiting for results, one shard at a time, in
    # whatever order they were spawned -- not the order they actually finish
    # in. That's fine for a progress bar (all N will complete regardless),
    # just don't read bar order as "shard i finished i-th".
    for call in tqdm(calls, desc=f"{kind}_{regime} shards", unit="shard"):
        call.get()
    print("All shards complete -- merging into the canonical file...")
    merge_shards.remote(kind=kind, regime=regime, out_subdir=out_subdir, shards=shard_suffixes)
    print(f"Merged: atlas_out/{out_subdir}/{kind}_{regime}.jsonl")


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
    collect_num_samples: int = 100,
    collect_iterations: int = 10,
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
    if data_source == "closed_loop":
        cmd += ["--collect-num-samples", str(collect_num_samples),
                "--collect-iterations", str(collect_iterations)]
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
                              out_subdir: str = "e0_v2", collect_num_samples: int = 100,
                              collect_iterations: int = 10) -> None:
    run_e0_train.remote(kinds=kinds, regimes=regimes, regime_config=regime_config, steps=steps,
                         num_train_trajs=num_train_trajs, train_traj_len=train_traj_len,
                         num_val_trajs=num_val_trajs, eval_traj_len=eval_traj_len,
                         eval_every=eval_every, patience=patience,
                         data_source=data_source, data_split=data_split,
                         out_subdir=out_subdir, collect_num_samples=collect_num_samples,
                         collect_iterations=collect_iterations)


@app.function(
    gpu="L4",
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=3600 * 4,
)
def run_e2(
    cells: str = "A,B,C,D",
    routers: str = "umf,sdyn",
    episodes: int = 40,
    seeds: int = 3,
    traj_len: int = 50,
    kind: str = "ln_act",
    chart_regime: str = "R1",
    corruption: str = "dark",
    corruption_severity: float = 0.5,
    dynamics_regime: str = "R1",
    probe_q: int = 3,
    probe_tau: float = 0.5,
    charts_subdir: str = "e0_v6_R1",
    out_subdir: str = "e2",
) -> None:
    """scripts/run_e2.py -- E2's 2x2 routing-accuracy grid.

    No CEM planner: routing accuracy is a property of UMF scoring on an observed
    chunk, so this is collection + scoring only and costs ~1 GPU-h rather than
    plan 7.3's 6. See run_e2.py's module docstring for that deviation.

    corruption defaults to 'dark', NOT plan 6.3's colour: colour_change was
    measured to alter only ~5.6% of pixels on this env (Push-T renders are ~97%
    white and an HSV hue rotation is a no-op on desaturated pixels), which would
    let Cell C's committed==0 pass vacuously. 'dark' changes 100% -- the
    conservative direction. run_e2.py reports the measured magnitude either way.

    charts_subdir defaults to e0_v6_R1, the post-rollout-fix R1 charts; the
    charts in atlas_out/e0 are the INVALIDATED pre-fix set.
    """
    import subprocess
    import sys
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/run_e2.py",
           "--cells", *cells.split(","),
           "--routers", *routers.split(","),
           "--episodes", str(episodes),
           "--seeds", str(seeds),
           "--traj-len", str(traj_len),
           "--kind", kind,
           "--chart-regime", chart_regime,
           "--corruption", corruption,
           "--corruption-severity", str(corruption_severity),
           "--dynamics-regime", dynamics_regime,
           "--probe-q", str(probe_q),
           "--probe-tau", str(probe_tau),
           "--charts-dir", f"{ATLAS_MOUNT_PATH}/atlas_out/{charts_subdir}",
           "--out", f"{ATLAS_MOUNT_PATH}/atlas_out/{out_subdir}"]
    subprocess.run(cmd, check=True, cwd="/src")
    atlas_volume.commit()


@app.local_entrypoint(name="run_e2")
def run_e2_entrypoint(cells: str = "A,B,C,D", routers: str = "umf,sdyn",
                       episodes: int = 40, seeds: int = 3, traj_len: int = 50,
                       kind: str = "ln_act", chart_regime: str = "R1",
                       corruption: str = "dark", corruption_severity: float = 0.5,
                       dynamics_regime: str = "R1", probe_q: int = 3, probe_tau: float = 0.5,
                       charts_subdir: str = "e0_v6_R1", out_subdir: str = "e2") -> None:
    run_e2.remote(cells=cells, routers=routers, episodes=episodes, seeds=seeds,
                   traj_len=traj_len, kind=kind, chart_regime=chart_regime,
                   corruption=corruption, corruption_severity=corruption_severity,
                   dynamics_regime=dynamics_regime, probe_q=probe_q, probe_tau=probe_tau,
                   charts_subdir=charts_subdir, out_subdir=out_subdir)
