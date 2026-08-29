"""
modal/modal_phase0.py — Modal GPU runner for scripts/phase0_measure.py
(IMPLEMENTATION_PLAN_V3 §11.1 Phase 0, gates P0-A/B/D/E).

Forward-only, no CEM planner -> T4 is plenty. Writes under the volume's
phase0_v3/ (NOT atlas_out/). Same local-repo image as modal_e0_planning.py.

    modal run --detach modal/modal_phase0.py::main --num-trajs 80
"""
from __future__ import annotations

from pathlib import Path

import modal

atlas_volume = modal.Volume.from_name("atlas-data", create_if_missing=True)
MOUNT = "/atlas_root"
REPO_ROOT = Path(__file__).parent.parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1")
    .pip_install("uv")
    .add_local_dir(
        str(REPO_ROOT), remote_path="/src", copy=True,
        ignore=[".venv", ".git", "data", "hub", "atlas_out", "graphify-out",
                "__pycache__", "logs", "phase0_v3"],
    )
    # §7-B1: modal_phase0.py imports _p0g_spec at module load; in the remote
    # container the entrypoint is at /root/ (not /root/modal/), so put the spec
    # there too. (The sys.path /src/scripts fallback below covers it as well.)
    .add_local_file(str(REPO_ROOT / "scripts" / "_p0g_spec.py"),
                    remote_path="/root/_p0g_spec.py", copy=True)
    .run_commands(
        "cd /src && uv pip install --system -e vendor/jepa-wms && "
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu121 && "
        "uv pip install --system -e ."
    )
    .env({
        "ATLAS_HOME": MOUNT,
        "JEPAWM_DSET": f"{MOUNT}/data",
        "JEPAWM_LOGS": f"{MOUNT}/logs",
        "JEPAWM_CKPT": f"{MOUNT}/ckpts",
        "ATLAS_OUT": f"{MOUNT}/atlas_out",
        "TORCH_HOME": f"{MOUNT}/hub",
    })
)

app = modal.App("atlas-phase0", image=image)


def _local_git_sha() -> str:
    """P20: read the SHA on the CLIENT (the image drops .git, so the remote
    `git rev-parse` returns 'unknown')."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)).strip()
    except Exception:
        return "unknown"


def _run(cmd: list[str], git_sha: str) -> None:
    import os
    import subprocess
    env = {**os.environ, "ATLAS_GIT_SHA": git_sha}
    subprocess.run(cmd, check=True, cwd="/src", env=env)


@app.function(gpu="T4", volumes={MOUNT: atlas_volume}, timeout=3600 * 3)
def phase0(num_trajs: int = 80, traj_len: int = 60, num_act_stepped: int = 2,
           regimes: str = "R0,R1,R2", git_sha: str = "unknown") -> None:
    import sys
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/phase0_measure.py",
           "--out", f"{MOUNT}/phase0_v3",
           "--regimes", *regimes.split(","),
           "--num-trajs", str(num_trajs),
           "--traj-len", str(traj_len),
           "--num-act-stepped", str(num_act_stepped)]
    _run(cmd, git_sha)
    atlas_volume.commit()


# P0-G is SPLIT into collection and fine-tune (P2 / v3 §5): the realistic
# single-container total for collect+finetune of both regimes is ~13.6 h — over
# the old timeout=3600*8. Splitting also means a fine-tune failure no longer
# destroys the ~4 h of collection, and the fine-tune becomes independently
# re-runnable (needed for §4.1's determinism-asymmetry check). Run ONE regime
# per call (P1: a combined R0,R2 call collects R2 under an R0-adapted predictor;
# also isolates CEM generator state). Cost line: the old SMOKE_SUMMARY.md
# "$3.6 / 4.5 h" is SUPERSEDED (§7-C-2) — §3.1 roughly doubles CEM compute per
# trajectory (plan_length 9→18 summed), so ~135 s/traj is the first-order
# estimate (NOT measured): ~4.3 h collection + ~5.8 h fine-tune per regime vs the
# 6 h / 10 h timeouts here. EVIDENCE_LEDGER §5 budgets P0-G at ~$15. Re-measure
# on the next smoke and rewrite this + SMOKE_SUMMARY.md + plan §11.2.
# R0 is a SEPARATE `--regime R0` call (τ / σ_r are defined over R0 chunks) —
# nothing here guards against forgetting it (§7-C-3).

# §7-B1: the guard-relevant flag spec lives in scripts/_p0g_spec.py (no `modal`
# import there) so tests/test_p0g_guard.py can use it without a Modal runtime.
# This module is imported both on the client (during `modal run`, where the repo
# is at REPO_ROOT) AND inside the remote container (where the entrypoint file is
# copied to /root/ and the repo image lives at /src — REPO_ROOT is then "/" and
# useless). Insert both candidate paths.
import sys as _sys  # noqa: E402
for _p in (str(REPO_ROOT / "scripts"), "/src/scripts"):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from _p0g_spec import _P0G_COMMON, _P0G_DEFAULTS, _p0g_flags  # noqa: E402,F401


@app.function(gpu="L4", volumes={MOUNT: atlas_volume}, timeout=3600 * 6)
def p0g_collect(regime: str = "R2",
                num_trajs: int = _P0G_DEFAULTS["num_trajs"],
                traj_len: int = _P0G_DEFAULTS["traj_len"],
                nas: int = _P0G_DEFAULTS["nas"],
                num_samples: int = _P0G_DEFAULTS["num_samples"],
                iterations: int = _P0G_DEFAULTS["iterations"],
                num_val_trajs: int = _P0G_DEFAULTS["num_val_trajs"],
                num_test_trajs: int = _P0G_DEFAULTS["num_test_trajs"],
                eval_traj_len: int = _P0G_DEFAULTS["eval_traj_len"],
                out_subdir: str = "p0g_onpolicy", git_sha: str = "unknown") -> None:
    """P0-G on-policy chart-training-data COLLECTION only (v3 §5.2). ONE regime
    per call. closed_loop collector, all v3 fixes: contact filter OFF,
    --collect-num-act-stepped FUNCTIONAL, N=300/it=10/nas=2, eval-matched
    lookahead + goal separation (§3.1/§3.2), determinism on. Persists
    trajs_{regime}.pt + chunks_{regime}.jsonl, then exits.
    Smoke: --num-trajs 5 --num-val-trajs 2 --num-test-trajs 2."""
    import sys
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/run_e0.py", *_P0G_COMMON, "--collect-only",
           "--regimes", regime,
           *_p0g_flags(traj_len, eval_traj_len, num_trajs, num_val_trajs,
                       num_test_trajs, num_samples, iterations, nas),
           "--out", f"{MOUNT}/phase0_v3/{out_subdir}"]
    _run(cmd, git_sha)
    atlas_volume.commit()


@app.function(gpu="L4", volumes={MOUNT: atlas_volume}, timeout=3600 * 10)
def p0g_finetune(regime: str = "R2", steps: int = 2000,
                 out_subdir: str = "p0g_onpolicy", load_subdir: str | None = None,
                 git_sha: str = "unknown",
                 num_trajs: int = _P0G_DEFAULTS["num_trajs"],
                 traj_len: int = _P0G_DEFAULTS["traj_len"],
                 nas: int = _P0G_DEFAULTS["nas"],
                 num_samples: int = _P0G_DEFAULTS["num_samples"],
                 iterations: int = _P0G_DEFAULTS["iterations"],
                 num_val_trajs: int = _P0G_DEFAULTS["num_val_trajs"],
                 num_test_trajs: int = _P0G_DEFAULTS["num_test_trajs"],
                 eval_traj_len: int = _P0G_DEFAULTS["eval_traj_len"]) -> None:
    """P0-G ln_act fine-tune from cached trajectories (P2). Requires a prior
    p0g_collect (load_subdir, defaults to out_subdir) for this regime — loads
    trajs_{regime}.pt via --load-trajs, so ZERO CEM searches happen here.
    Independently re-runnable.

    The eight collection-defining params default off _P0G_DEFAULTS and are
    emitted via _p0g_flags — identical to p0g_collect — so run_e0.py's
    --load-trajs guard matches (§7-B1). If a collection used non-default values,
    pass the SAME values here.

    §4.1 (P16) determinism-asymmetry check: run this TWICE with the same
    load_subdir but different out_subdir (e.g. det_run1 / det_run2) per regime,
    then diff val_loss_ln_act_{regime}.json stopped_early_at_step and
    results.json eval_umf across the two. NEVER reuse an out_subdir (§1.7)."""
    import sys
    atlas_volume.reload()
    out = f"{MOUNT}/phase0_v3/{out_subdir}"
    src = f"{MOUNT}/phase0_v3/{load_subdir or out_subdir}"
    cmd = [sys.executable, "scripts/run_e0.py", *_P0G_COMMON,
           "--regimes", regime,
           "--load-trajs", src,
           "--steps", str(steps),
           *_p0g_flags(traj_len, eval_traj_len, num_trajs, num_val_trajs,
                       num_test_trajs, num_samples, iterations, nas),
           "--out", out]
    _run(cmd, git_sha)
    atlas_volume.commit()


@app.local_entrypoint()
def main(num_trajs: int = 80, traj_len: int = 60, num_act_stepped: int = 2,
         regimes: str = "R0,R1,R2") -> None:
    phase0.remote(num_trajs=num_trajs, traj_len=traj_len,
                  num_act_stepped=num_act_stepped, regimes=regimes,
                  git_sha=_local_git_sha())


@app.local_entrypoint(name="p0g-collect")
def p0g_collect_entry(regime: str = "R2",
                      num_trajs: int = _P0G_DEFAULTS["num_trajs"],
                      traj_len: int = _P0G_DEFAULTS["traj_len"],
                      nas: int = _P0G_DEFAULTS["nas"],
                      num_samples: int = _P0G_DEFAULTS["num_samples"],
                      iterations: int = _P0G_DEFAULTS["iterations"],
                      num_val_trajs: int = _P0G_DEFAULTS["num_val_trajs"],
                      num_test_trajs: int = _P0G_DEFAULTS["num_test_trajs"],
                      eval_traj_len: int = _P0G_DEFAULTS["eval_traj_len"],
                      out_subdir: str = "p0g_onpolicy") -> None:
    """ONE regime per call (P1). **Run BOTH: --regime R2 (charts) AND --regime R0**
    — τ = P95 UMF(c₀) over R0 chunks and σ_r over the R0 informative set (§6.1,
    §6.3), so R0 collection is required even though no R0 chart is used downstream
    (§7-C-3). Smoke: --num-trajs 5 --num-val-trajs 2 --num-test-trajs 2 (§7-C-1)."""
    p0g_collect.remote(regime=regime, num_trajs=num_trajs, traj_len=traj_len,
                       nas=nas, num_samples=num_samples, iterations=iterations,
                       num_val_trajs=num_val_trajs, num_test_trajs=num_test_trajs,
                       eval_traj_len=eval_traj_len, out_subdir=out_subdir,
                       git_sha=_local_git_sha())


@app.local_entrypoint(name="p0g-finetune")
def p0g_finetune_entry(regime: str = "R2", steps: int = 2000,
                       out_subdir: str = "p0g_onpolicy", load_subdir: str = "",
                       num_trajs: int = _P0G_DEFAULTS["num_trajs"],
                       traj_len: int = _P0G_DEFAULTS["traj_len"],
                       nas: int = _P0G_DEFAULTS["nas"],
                       num_samples: int = _P0G_DEFAULTS["num_samples"],
                       iterations: int = _P0G_DEFAULTS["iterations"],
                       num_val_trajs: int = _P0G_DEFAULTS["num_val_trajs"],
                       num_test_trajs: int = _P0G_DEFAULTS["num_test_trajs"],
                       eval_traj_len: int = _P0G_DEFAULTS["eval_traj_len"]) -> None:
    """Collection-defining params must MATCH the p0g_collect run being loaded
    (they default identically off _P0G_DEFAULTS; override all of them together
    if the collection was non-default). §7-B1."""
    p0g_finetune.remote(regime=regime, steps=steps, out_subdir=out_subdir,
                        load_subdir=load_subdir or None, git_sha=_local_git_sha(),
                        num_trajs=num_trajs, traj_len=traj_len, nas=nas,
                        num_samples=num_samples, iterations=iterations,
                        num_val_trajs=num_val_trajs, num_test_trajs=num_test_trajs,
                        eval_traj_len=eval_traj_len)
