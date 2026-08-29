"""
scripts/run_e4.py — E4 + E3: Continual stream (S2) with 7-arm ablation ladder.

E4 (RQ4): persistent chart library delivers recall on A→B→A.
E3 (RQ3): expansion arms 4/5/6 form the expansion ablation inside the same run.

Arms (in order of the ablation ladder):
  1. frozen
  2. adajepa
  3. adajepa_persist
  4. atlas_fixed      (expansion_mode: fixed)
  5. atlas_detect     (expansion_mode: detect_only)
  6. atlas            (expansion_mode: atlas, full verification)
  7. oracle_id

Does NOT use jepa-wms's own eval.py/PlanEvaluator -- same reasoning as
run_e0_planning.py/run_e1.py (confirmed broken for Push-T, see code-review.md
Bug #7). Reuses GC_Agent/CEMPlanner directly plus E0's corrected init/goal
sampler and success metric (atlas.harness_e4.run_e4_episode).

Usage:
    python scripts/run_e4.py --arms frozen adajepa atlas --episodes 20 --seeds 3
    python scripts/run_e4.py --profile --episodes 3  # budget calibration
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

import torch
from omegaconf import OmegaConf

import atlas
from atlas.loop import ATLASConfig
from atlas.score import compute_motion_gate
from atlas.streams import get_stream
from atlas.harness_e4 import build_arm_state, run_e4_episode

# Raw ATLAS_HOME env var, NOT atlas.ATLAS_HOME (.resolve()'d -- breaks on the
# Modal volume mount; see run_e0_planning.py's identical HUB_PATH comment).
_atlas_home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
HUB_PATH = str(Path(_atlas_home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
if HUB_PATH not in sys.path:
    sys.path.insert(0, HUB_PATH)

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402

from atlas.regimes import PhysicsRegime, REGIME_CONFIGS  # noqa: E402
from scripts.run_e0 import load_regime_trajectories  # noqa: E402
from scripts.run_e0_planning import load_dataset_states  # noqa: E402

ArmName = Literal[
    "frozen", "adajepa", "adajepa_persist",
    "atlas_fixed", "atlas_detect", "atlas", "oracle_id"
]

ALL_ARMS: list[ArmName] = [
    "frozen", "adajepa", "adajepa_persist",
    "atlas_fixed", "atlas_detect", "atlas", "oracle_id",
]

FRAMESKIP = 5
CEM_NUM_SAMPLES = 300
CEM_ITERATIONS = 30          # cut to 10 per Phase 0.2's ladder, after profiling
CEM_HORIZON = 6
# E4 DEVIATION: nas=6 (the substrate's own validated config, used by
# run_e0_planning.py/run_e1.py) gives ONE replan per 30-raw-step episode,
# which structurally cannot exercise routing, refinement, or Expander's
# next-chunk verification. nas=1 -> 5 raw actions/replan -> 6 replans in 30
# raw steps. Same reasoning as E0_RECOVERY_PLAN.md P5's fix for E1; documented
# per ATLAS_implementation_plan_v2.md §7.0a's convention.
CEM_NUM_ACT_STEPPED = 1
MAX_MPC_STEPS = 30
HYSTERESIS = 0.05
REGIME_LABELS = {"R0": 0, "R1": 1, "R2": 2}


def build_planner_cfg(num_samples: int, iterations: int, horizon: int, num_act_stepped: int,
                       local_seed: int = 0) -> OmegaConf:
    return OmegaConf.create({
        "local_seed": local_seed,
        "task_specification": {"obs": "rgb_state"},
        "planner": {
            "planner_name": "cem",
            "iterations": iterations,
            "num_samples": num_samples,
            "num_elites": min(10, num_samples),
            "horizon": horizon,
            "var_scale": 1.0,
            "num_act_stepped": num_act_stepped,
            "repeat_actskip": False,
            "decode_each_iteration": False,
            "distribute_planner": False,
            "planning_objective": {"objective_type": "L2", "sum_all_diffs": False, "alpha": 0.1},
        },
    })


def load_completed_keys(jsonl_path: Path) -> set[tuple[str, int, int]]:
    """Resume support: (arm, seed_run, global_episode_idx) triples already on disk."""
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec["arm"], rec["seed_run"], rec["global_episode_idx"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="E4+E3: Continual stream S2.")
    parser.add_argument("--stream", default="s2", choices=["s2"])
    parser.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    parser.add_argument("--episodes", type=int, default=20, help="Episodes PER SEGMENT.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-run-offset", type=int, default=0,
                        help="FIX_SPEC.md B12: base seed_run index for THIS process. "
                             "modal_e4.py always requests --seeds 1 (one seed_run per "
                             "container) but previously always ran local seed_run=0 "
                             "regardless of which seed_run the container was launched "
                             "for -- get_stream(...) was built with seeds=1, so "
                             "streams[0] (the seed_run=0 episode/init/goal set) was the "
                             "only stream ever generated, and modal_e4.py's post-hoc "
                             "rec['seed_run']=seed_run relabelling made 'different "
                             "seeds' silently mean 'bit-identical episodes under "
                             "different labels'. With --seed-run-offset=K, this process "
                             "generates and runs streams[K:K+seeds] (the REAL seed_run "
                             "K..K+seeds-1 episode sets) and seeds the CEM planner's own "
                             "local_seed from K too, so distinct containers launched at "
                             "distinct offsets genuinely diverge in both init/goal "
                             "sampling and planner sampling.")
    parser.add_argument("--segment-regimes", nargs=2, default=["R0", "R2"],
                        metavar=("REGIME_A", "REGIME_B"),
                        help="Alternating S2 regime pair (A,B). Default R0/R2 -- see "
                             "atlas.streams.stream_s2's docstring for why R2, not R1.")
    parser.add_argument("--charts", type=Path, default=atlas.OUT_DIR / "e0_v3_dataset",
                        help="Directory with chart_{kind}_{regime}.pt files from E0. "
                             "Placeholder default per plan Phase 5 -- confirm with the "
                             "user before a real run.")
    parser.add_argument("--kind", default="ln_act", choices=["ln_act", "lora4", "full"])
    parser.add_argument("--expansion-start-library", default="full", choices=["full", "c0_only"],
                        help="'full' (default): atlas_fixed/detect/atlas start from "
                             "{c0, chart_B} -- monotone ladder, correct ATLAS = 0 commits. "
                             "'c0_only': start from {c0} only, requiring discovery. "
                             "See E3_E4_IMPLEMENTATION_PLAN.md §2b -- ask the user which "
                             "to report before the real run.")
    parser.add_argument("--num-samples", type=int, default=CEM_NUM_SAMPLES)
    parser.add_argument("--iterations", type=int, default=CEM_ITERATIONS)
    parser.add_argument("--horizon", type=int, default=CEM_HORIZON)
    parser.add_argument("--num-act-stepped", type=int, default=CEM_NUM_ACT_STEPPED)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e4")
    parser.add_argument("--profile", action="store_true",
                        help="Profile mode: run ONLY the 'atlas' arm for --episodes "
                             "episodes on segment 0, print timing + extrapolated "
                             "GPU-h for the full grid, then exit without running anything else.")
    args = parser.parse_args()

    # FIX_SPEC.md B10: was `max_mpc_steps // num_act_stepped` -- a raw-step
    # budget (max_mpc_steps) divided by a MODEL-step count
    # (num_act_stepped), a unit mismatch (num_act_stepped counts model
    # chunks, each frameskip=5 raw steps long). At nas=1 this computed
    # n_replans_target=30 instead of the correct 6 -- numerically INERT for
    # episode length only because harness_e4.py's replan loop separately
    # breaks once `elapsed >= max_raw_steps`, so real episodes still ran 6
    # replans regardless. But n_replans_target also feeds
    # `steps_left_model = (n_replans_target - replan_idx) * num_act_stepped`
    # (harness_e4.py), which tells the CEM planner how many MODEL steps
    # remain so it can shorten its plan/horizon near the end of the
    # episode -- with the wrong (30 instead of 6) n_replans_target,
    # steps_left_model was always far larger than the horizon (6), so that
    # shortening logic never actually engaged. Divide by RAW steps per
    # replan (num_act_stepped * FRAMESKIP), matching max_mpc_steps' own
    # raw-step units.
    n_replans_target = max(args.max_mpc_steps // (args.num_act_stepped * FRAMESKIP), 1)
    regime_a, regime_b = args.segment_regimes

    print("Loading dino_wm_pusht from local hub...")
    model, prep = torch.hub.load(
        HUB_PATH, "dino_wm_pusht", source="local", force_reload=False, trust_repo=True,
    )
    wm = model.model if hasattr(model, "model") else model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = wm.to(device)
    model = model.to(device)
    for p in wm.encoder.parameters():
        p.requires_grad_(False)
    for p in wm.predictor.parameters():
        p.requires_grad_(False)

    # T7 throughput fixes -- see run_e0_planning.py's identical block for detail.
    for m in wm.predictor.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    torch.set_float32_matmul_precision("high")

    pristine_predictor_state = copy.deepcopy(wm.predictor.state_dict())

    # Motion gate: computed ONCE from the A-segment regime (R0) and held fixed
    # for the whole stream (plan §3 -- a per-regime gate would shift the
    # informative-chunk definition underneath the strike counter, exactly
    # what G6 exists to prevent).
    #
    # FIX_SPEC.md B3 (SUBMISSION_PLAN.md Part B2 B3, confirmed empirically by
    # the Phase 3 Step 0 measurement -- research_audit/FIXLOG.md): the OLD
    # calibration took the 10th percentile of WHOLE-"traj_len=10" (== 2
    # model-step, since frameskip=5) displacement from only 3 trajectories,
    # then applied that threshold to SINGLE-model-step (nas=1) chunks at
    # runtime -- a quantity that is structurally larger than what it gates.
    # Measured: old gate=317.77 vs. real nas=1 chunk displacement max
    # 278.55 (R0) / 331.57 (R2) -- 100.00% / 99.44% of real nas=1 chunks
    # gated to None. Recalibrated here at the EXACT chunk granularity
    # scored at runtime (frameskip * num_act_stepped raw steps -> ONE
    # model-step chunk when num_act_stepped=1, matching harness_e4.py's
    # own per-replan re-encoding window), over >=30 trajectories (was 3).
    gate_traj_len = FRAMESKIP * args.num_act_stepped
    gate_num_trajs = 30
    print(f"Computing motion_gate from {gate_num_trajs} fresh {regime_a} trajectories "
          f"at the runtime-scored chunk size (traj_len={gate_traj_len} = "
          f"frameskip*num_act_stepped)...")
    # P7: `source` was IMPLICIT (signature default "scripted", the retired
    # goal-free walk). Explicit now; NOT changed to "closed_loop" (a scope
    # decision needing sign-off — FIXLOG V3-8 / P0G_FIX_PLAN §4.3).
    GATE_SOURCE = "scripted"
    gate_trajectories = load_regime_trajectories(
        model, prep, regime_a, num_trajs=gate_num_trajs, traj_len=gate_traj_len,
        device=device, seed_offset=20_000, source=GATE_SOURCE)
    gate_displacements = torch.tensor([
        (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
        for t in gate_trajectories
    ])
    motion_gate = compute_motion_gate(gate_displacements)
    print(f"  motion_gate = {motion_gate:.4f}  "
          f"(from {len(gate_displacements)} single-chunk displacements, "
          f"range [{gate_displacements.min():.2f}, {gate_displacements.max():.2f}])")

    # FIX_SPEC.md B12: local_seed now varies with --seed-run-offset so the
    # CEM planner's own internal sampling genuinely differs across
    # containers launched at distinct offsets, not just the init/goal draw
    # below.
    cfg = build_planner_cfg(args.num_samples, args.iterations, args.horizon,
                            args.num_act_stepped, local_seed=args.seed_run_offset)
    agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
    agent.device = device

    base_env = PushTEnv(render_size=224, with_velocity=True)
    segment_regimes_used = {regime_a, regime_b}
    regimes = {r: PhysicsRegime(base_env, r) for r in segment_regimes_used}

    dataset_states, dataset_seq_lengths = load_dataset_states()

    atlas_cfg = ATLASConfig(
        router="umf", tau=0.5, q=3, hysteresis=HYSTERESIS, lr=5e-4, n_probe=20,
        motion_gate=motion_gate, k_max=10,
    )

    chart_b_path = args.charts / f"chart_{args.kind}_{regime_b}.pt"

    if args.profile:
        args.arms = ["atlas"]
        print(f"\n[PROFILE MODE] arm=atlas, segment 0 only, {args.episodes} episode(s)")

    # FIX_SPEC.md B12: request enough streams to cover
    # [seed_run_offset, seed_run_offset + seeds), not just [0, seeds) --
    # get_stream/stream_s2 always generate sequentially from seed_run=0, so
    # to genuinely produce the seed_run=K..K+seeds-1 episode sets this
    # process must ask for seed_run_offset+seeds streams total and use the
    # tail slice. (Cheap: stream_s2 is pure seed arithmetic, no GPU/env
    # cost per extra stream.)
    streams_all = get_stream(args.stream, args.episodes, args.seed_run_offset + args.seeds,
                              regimes=(regime_a, regime_b))
    streams = streams_all[args.seed_run_offset:args.seed_run_offset + args.seeds]
    print(f"\nStream {args.stream}: {len(streams)} seed(s) (seed_run "
          f"{args.seed_run_offset}..{args.seed_run_offset + args.seeds - 1}) x "
          f"{len(streams[0])} episodes each (regimes={regime_a}/{regime_b})")
    print(f"Arms: {args.arms}")
    print(f"Output: {args.out}")

    args.out.mkdir(parents=True, exist_ok=True)
    episodes_jsonl = args.out / "episodes.jsonl"
    if not args.profile and episodes_jsonl.exists():
        already_done = load_completed_keys(episodes_jsonl)
        print(f"Resume: {len(already_done)} episode(s) already logged in {episodes_jsonl}")
    else:
        already_done = set()
        if args.profile and episodes_jsonl.exists():
            pass  # profile mode never writes to the real episodes.jsonl below

    profile_times: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    profile_seeds = 1 if args.profile else args.seeds
    for arm in args.arms:
        # FIX_SPEC.md B12: seed_run is now the REAL global seed_run index
        # (offset-shifted); `streams` was already sliced to
        # [seed_run_offset, seed_run_offset+seeds), so index into it with
        # the LOCAL position `local_i`, not the global `seed_run`.
        for local_i in range(profile_seeds):
            seed_run = args.seed_run_offset + local_i
            wm.predictor.load_state_dict(pristine_predictor_state)
            state = build_arm_state(
                arm=arm, predictor=wm.predictor, world_model=model, kind=args.kind,
                chart_b_path=chart_b_path, cfg=atlas_cfg,
                expansion_start_library=args.expansion_start_library,
            )
            specs = streams[local_i] if not args.profile else \
                [s for s in streams[local_i] if s.segment_idx == 0][:args.episodes]

            for spec in specs:
                key = (arm, seed_run, spec.global_episode_idx)
                if key in already_done:
                    continue

                import random as _random
                router_rng_seed = _random.Random(spec.seed).randrange(2**31) if arm != "frozen" else None

                t0 = time.time()
                record = run_e4_episode(
                    state=state, agent=agent, world_model=model, base_env=base_env,
                    regimes=regimes, spec=spec,
                    dataset_states=dataset_states, dataset_seq_lengths=dataset_seq_lengths,
                    n_replans_target=n_replans_target, frameskip=FRAMESKIP,
                    num_act_stepped=args.num_act_stepped, max_raw_steps=args.max_mpc_steps,
                    motion_gate=motion_gate, out_dir=args.out if not args.profile else (args.out / "_profile_tmp"),
                    seed_run=seed_run, router_rng_seed=router_rng_seed,
                )
                dt = time.time() - t0
                profile_times.append(dt)
                print(f"  [{arm}] seed={seed_run} seg={spec.segment_idx} ep={spec.episode_idx} "
                      f"({spec.regime}): success={record['success']} replans={record['n_replans']} "
                      f"steps={record['elapsed_raw_steps']} wall={dt:.1f}s")

    if args.profile:
        import shutil
        tmp_dir = args.out / "_profile_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        n = len(profile_times)
        mean_ep_s = sum(profile_times) / n if n else float("nan")
        mean_replan_s = mean_ep_s / n_replans_target if n_replans_target else float("nan")
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        total_episodes_full_grid = len(ALL_ARMS) * args.seeds * 6 * args.episodes
        extrapolated_gpu_h = mean_ep_s * total_episodes_full_grid / 3600
        print(f"\n[PROFILE] n_replans_target={n_replans_target}")
        print(f"[PROFILE] mean sec/episode = {mean_ep_s:.1f}")
        print(f"[PROFILE] mean sec/replan  = {mean_replan_s:.1f}")
        if peak_mem_gb is not None:
            print(f"[PROFILE] peak GPU memory  = {peak_mem_gb:.2f} GB")
        print(f"[PROFILE] extrapolated full-grid GPU-h "
              f"({len(ALL_ARMS)} arms x {args.seeds} seeds x 6 segments x {args.episodes} eps/seg "
              f"= {total_episodes_full_grid} episodes) = {extrapolated_gpu_h:.1f} GPU-h")
        return

    # ── Summary JSON ─────────────────────────────────────────────────────────
    all_records = []
    if episodes_jsonl.exists():
        with open(episodes_jsonl) as f:
            all_records = [json.loads(line) for line in f if line.strip()]

    summary = {
        "arms": args.arms, "stream": args.stream,
        "segment_regimes": [regime_a, regime_b],
        "episodes_per_segment": args.episodes, "seeds": args.seeds,
        "num_samples": args.num_samples, "iterations": args.iterations,
        "horizon": args.horizon, "num_act_stepped": args.num_act_stepped,
        "max_mpc_steps": args.max_mpc_steps, "hysteresis": HYSTERESIS,
        "expansion_start_library": args.expansion_start_library,
        "motion_gate": motion_gate,
        "gate_source": GATE_SOURCE,  # P7
        "regime_configs": {r: dict(REGIME_CONFIGS.get(r, {})) for r in segment_regimes_used},
        "chart_kind": args.kind, "charts_dir": str(args.charts),
        "per_arm_success_rate": {
            arm: (sum(r["success"] for r in all_records if r["arm"] == arm) /
                  max(1, sum(1 for r in all_records if r["arm"] == arm)))
            for arm in args.arms
        },
        "mean_wall_time_s": (sum(r["wall_time"] for r in all_records) / len(all_records)
                              if all_records else None),
        "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9
                                if torch.cuda.is_available() else None),
    }
    (args.out / "e4_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {args.out / 'e4_summary.json'}")
    print(f"Episodes JSONL: {episodes_jsonl}")


if __name__ == "__main__":
    main()
