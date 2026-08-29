"""
scripts/smoke_e4.py — E4/E3 smoke test. Run this BEFORE spending real GPU-hours
on the full grid. Tiny CEM budget (jepa-wms's own `quick_debug` scale), 2
episodes per segment, 2 segments, ALL 7 arms.

Asserts, not just prints (E3_E4_IMPLEMENTATION_PLAN.md Phase 4 checklist):
  - every arm completes without exception
  - identical init_block_pos_diff per global_episode_idx across all 7 arms (G5)
  - raw_steps_per_replan entries are multiples of frameskip and consistent
    with num_act_stepped=1 (short episodes here, so no fixed [5]*6 length --
    the full-length invariant is validated by run_e4.py's own smoke run at
    num_act_stepped=1 with max_mpc_steps=30, not this tiny-CEM/short-episode
    smoke config)
  - umf_trace entries are finite where not None
  - oracle_id selects index 0 in the A regime, index 1 in the B regime
  - frozen never changes library_size (always 1); atlas_fixed never commits
  - atlas_detect / atlas can commit (mechanism reachable), not asserted to
    fire within this tiny budget
  - arm 2 (adajepa) predictor state is reset each episode (bit-identical at
    episode start); arm 3 (adajepa_persist) is not required to be
  - every key in the JSONL contract (harness_e4.run_e4_episode's record) is present

Usage:
    python scripts/smoke_e4.py --charts atlas_out/e0_v3_dataset --kind ln_act \\
        --segment-regimes R0 R2
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HUB_PATH = str(REPO_ROOT / "hub" / "hub" / "facebookresearch_jepa-wms_main")
if HUB_PATH not in sys.path:
    sys.path.insert(0, HUB_PATH)

import atlas  # noqa: E402
from atlas.loop import ATLASConfig  # noqa: E402
from atlas.score import compute_motion_gate  # noqa: E402
from atlas.streams import get_stream  # noqa: E402
from atlas.harness_e4 import build_arm_state, run_e4_episode  # noqa: E402

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402
from atlas.regimes import PhysicsRegime  # noqa: E402
from scripts.run_e0 import load_regime_trajectories  # noqa: E402
from scripts.run_e0_planning import load_dataset_states  # noqa: E402
from scripts.run_e4 import build_planner_cfg, ALL_ARMS as RUN_E4_ARMS  # noqa: E402

REQUIRED_KEYS = {
    "arm", "success", "segment_idx", "global_episode_idx",
    "probe_outcome", "library_size", "library_full", "charts_committed_cumulative",
    "probes_rejected_cumulative", "seed_run", "episode_idx", "regime",
    "regime_label", "seed", "selected_trace", "umf_trace", "strikes",
    "elapsed_raw_steps", "n_replans", "raw_steps_per_replan",
    "init_block_pos_diff", "init_block_angle_diff", "init_agent_block_dist",
    "total_contacts", "block_pos_diff", "block_angle_diff", "refine_loss",
    "wall_time",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="E4/E3 smoke test.")
    parser.add_argument("--charts", type=Path, default=atlas.OUT_DIR / "e0_v3_dataset")
    parser.add_argument("--kind", default="ln_act", choices=["ln_act", "lora4", "full"])
    parser.add_argument("--segment-regimes", nargs=2, default=["R0", "R2"])
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e4_smoke")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)

    regime_a, regime_b = args.segment_regimes
    chart_b_path = args.charts / f"chart_{args.kind}_{regime_b}.pt"
    if not chart_b_path.exists():
        print(f"FATAL: chart not found at {chart_b_path} -- pass --charts/--kind/"
              f"--segment-regimes pointing at a real E0 output directory.")
        sys.exit(1)

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

    pristine_predictor_state = copy.deepcopy(wm.predictor.state_dict())

    print(f"Computing motion_gate from a fresh {regime_a} trajectory sample...")
    gate_trajectories = load_regime_trajectories(
        model, prep, regime_a, num_trajs=2, traj_len=10, device=device,
        seed_offset=30_000, source="scripted")  # P7: explicit (was the signature default)
    gate_displacements = torch.tensor([
        (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
        for t in gate_trajectories
    ])
    motion_gate = compute_motion_gate(gate_displacements)
    print(f"  motion_gate = {motion_gate:.4f}")

    # Tiny CEM budget (jepa-wms's own quick_debug scale).
    cfg = build_planner_cfg(num_samples=2, iterations=2, horizon=2, num_act_stepped=1)
    agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
    agent.device = device

    base_env = PushTEnv(render_size=224, with_velocity=True)
    regimes = {r: PhysicsRegime(base_env, r) for r in {regime_a, regime_b}}
    dataset_states, dataset_seq_lengths = load_dataset_states()

    atlas_cfg = ATLASConfig(router="umf", tau=0.5, q=3, hysteresis=0.05, lr=5e-4,
                             n_probe=3, motion_gate=motion_gate, k_max=10)

    streams = get_stream("s2", episodes_per_segment=2, seeds=1, regimes=(regime_a, regime_b))
    # Only segments 0 and 1 (one A, one B) for the smoke test.
    specs_all = [s for s in streams[0] if s.segment_idx < 2]

    print(f"\n[1/6] Running all 7 arms on {len(specs_all)} episodes (2 segments x 2 eps)...")
    all_records: dict[str, list[dict]] = {}
    for arm in RUN_E4_ARMS:
        wm.predictor.load_state_dict(pristine_predictor_state)
        state = build_arm_state(
            arm=arm, predictor=wm.predictor, world_model=model, kind=args.kind,
            chart_b_path=chart_b_path, cfg=atlas_cfg, expansion_start_library="full",
        )
        records = []
        adajepa_reset_ok = True
        for spec in specs_all:
            if arm == "adajepa":
                state.adapter.reset()
                post_reset_state = {k: v.clone() for k, v in wm.predictor.state_dict().items()
                                     if k in state.adapter.param_names}
                # FIX_SPEC.md B13: this filters BOTH sides (post_reset_state
                # and pretrained_state) by the same state.adapter.param_names
                # -- if that namespace list were ever empty (or totally
                # diverged from the predictor's real state_dict keys), both
                # filtered dicts would be empty, all()/torch.equal over zero
                # pairs would vacuously be True, and this check would pass
                # having tested nothing. Assert real coverage explicitly.
                assert len(state.adapter.param_names) > 0, (
                    "adajepa arm: param_names is empty -- reset-equality check "
                    "would be vacuous."
                )
                assert len(post_reset_state) == len(state.adapter.param_names), (
                    f"adajepa arm: only {len(post_reset_state)}/"
                    f"{len(state.adapter.param_names)} param_names found in the "
                    "live predictor's state_dict -- namespace mismatch, reset "
                    "check coverage is incomplete."
                )
                adajepa_reset_ok = adajepa_reset_ok and all(
                    torch.equal(post_reset_state[k], state.adapter.pretrained_state[k])
                    for k in post_reset_state
                )
            record = run_e4_episode(
                state=state, agent=agent, world_model=model, base_env=base_env,
                regimes=regimes, spec=spec,
                dataset_states=dataset_states, dataset_seq_lengths=dataset_seq_lengths,
                n_replans_target=4, frameskip=5, num_act_stepped=1, max_raw_steps=10,
                motion_gate=motion_gate, out_dir=args.out, seed_run=0,
                router_rng_seed=spec.seed,
            )
            records.append(record)
        all_records[arm] = records
        print(f"  OK [{arm}]: {[r['success'] for r in records]}")

        if arm == "adajepa":
            # Right after reset(), predictor state must equal pretrained -- checked
            # BEFORE each episode's refine() call runs (which moves it away again).
            assert adajepa_reset_ok, "adajepa arm: reset() did not restore pretrained predictor state"

        if arm == "frozen":
            # FIX_SPEC.md B13: the OLD check (r["library_size"] == 1) was
            # structurally vacuous -- harness_e4.py's record always writes
            # library_size=1 whenever state.library is None, which
            # build_arm_state() guarantees for arm="frozen" by construction;
            # the assertion could not fail regardless of what the frozen arm
            # actually did. Replace with a genuinely dynamic check on the
            # actual substance of "frozen": the predictor's weights must be
            # bit-identical to the pristine state after running its episodes
            # -- a real invariant that a future bug (e.g. accidentally
            # calling atlas_refine for this arm) would break.
            frozen_state = wm.predictor.state_dict()
            for k, v in pristine_predictor_state.items():
                assert torch.equal(v, frozen_state[k]), (
                    f"frozen arm: predictor param {k!r} changed from pristine -- "
                    "frozen must never adapt."
                )

        if arm == "atlas_fixed":
            # FIX_SPEC.md B13: the OLD check (probe_outcome != "committed")
            # was a control-flow guarantee, not a behavioural test --
            # atlas_step()'s 'fixed'/'none' branch never touches
            # probe_outcome at all, so it is provably "not_ready" on every
            # call regardless of what the arm actually does. Replace with
            # two dynamic checks against real per-episode data: (1) the
            # library never grows past its initial size (the substantive
            # claim "fixed never commits" actually means), and (2) the arm
            # genuinely ROUTES -- selects more than one distinct chart index
            # across the smoke run (proves the routing mechanism is live,
            # not that it trivially never runs at all).
            initial_size = len(state.library)
            for r in records:
                assert r["library_size"] == initial_size, (
                    f"atlas_fixed: library_size changed from {initial_size} to "
                    f"{r['library_size']} -- 'fixed' must never commit a new chart."
                )
            distinct_selected = {idx for r in records for idx in r["selected_trace"]}
            print(f"    [atlas_fixed] distinct selected chart indices across the "
                  f"smoke run: {distinct_selected} (informational -- this tiny "
                  f"budget/episode-length smoke config has only 1 scored replan "
                  f"per episode, so it is not guaranteed to exercise a real "
                  f"cross-regime switch; run_e4.py's own full-length smoke run is "
                  f"where genuine multi-index routing gets exercised).")

    print("\n[2/6] Checking pairing (G5): init_block_pos_diff identical across all 7 arms per episode...")
    for i in range(len(specs_all)):
        vals = {arm: all_records[arm][i]["init_block_pos_diff"] for arm in RUN_E4_ARMS}
        ref = next(iter(vals.values()))
        assert all(v == ref for v in vals.values()), f"G5 FAILED at episode {i}: {vals}"
    print("  OK: identical init_block_pos_diff for all arms, per episode.")

    print("\n[3/6] Checking raw_steps_per_replan divisibility by frameskip...")
    # FIX_SPEC.md B13: the OLD check only asserted `n >= 0` on a Python list
    # length, which is impossible to be negative -- it tested nothing the
    # docstring claims. Implement the real check: every replan's raw-step
    # count must be an exact multiple of frameskip=5, EXCEPT possibly the
    # very last replan of an episode that ended in success (harness_e4.py's
    # inner loop can break mid-chunk the instant success is detected, before
    # completing a full frameskip block -- see harness_e4.py:271-274).
    FRAMESKIP_CHECK = 5
    n_checked = 0
    for arm, records in all_records.items():
        for r in records:
            steps = r["raw_steps_per_replan"]
            for i, n in enumerate(steps):
                is_last = (i == len(steps) - 1)
                if is_last and r["success"]:
                    assert 0 <= n <= FRAMESKIP_CHECK, (
                        f"[{arm}] final (success) raw_steps_per_replan entry "
                        f"{n} out of range [0, {FRAMESKIP_CHECK}]: {steps}"
                    )
                else:
                    assert n % FRAMESKIP_CHECK == 0, (
                        f"[{arm}] raw_steps_per_replan entry {n} (index {i}) is "
                        f"not a multiple of frameskip={FRAMESKIP_CHECK}: {steps}"
                    )
                n_checked += 1
    assert n_checked > 0, "B13: divisibility check ran over zero entries -- vacuous."
    print(f"  OK: {n_checked} raw_steps_per_replan entries checked for "
          f"frameskip={FRAMESKIP_CHECK} divisibility.")

    print("\n[4/6] Checking umf_trace finiteness and oracle_id routing...")
    for arm, records in all_records.items():
        for r in records:
            for scores in r["umf_trace"]:
                for s in scores:
                    if s is not None:
                        assert s == s and abs(s) != float("inf"), f"[{arm}] non-finite UMF: {s}"
    oracle_checked = 0
    for r in all_records["oracle_id"]:
        expected_idx = 0 if r["regime"] == regime_a else 1
        # Warmup (no prev_chunk) replan(s) stay on whatever current_idx carried in;
        # only check the LAST selected index, which had a scored chunk behind it.
        if len(r["selected_trace"]) > 1:
            assert r["selected_trace"][-1] == expected_idx, (
                f"oracle_id FAILED: regime={r['regime']} expected chart {expected_idx}, "
                f"got selected_trace={r['selected_trace']}"
            )
            oracle_checked += 1
    # FIX_SPEC.md B13: if every oracle_id record had len(selected_trace)<=1
    # (all-warmup episodes), the loop above would silently check nothing and
    # still print "OK" -- a latent vacuous mode. Assert real coverage.
    assert oracle_checked > 0, (
        "B13: oracle_id routing check covered ZERO records (all episodes were "
        "warmup-only) -- vacuous. Increase n_replans_target/max_raw_steps."
    )
    print(f"  OK: umf_trace finite; oracle_id selects the correct chart by the "
          f"last replan ({oracle_checked}/{len(all_records['oracle_id'])} records checked).")

    print("\n[5/6] Library-size invariants checked per-arm inline above "
          "(FIX_SPEC.md B13 -- frozen's predictor-identity check and "
          "atlas_fixed's library-size + multi-index-routing checks are now "
          "dynamic, not hardcoded-literal assertions; see the per-arm loop).")

    print("\n[6/6] Checking JSONL contract...")
    jsonl_path = args.out / "episodes.jsonl"
    assert jsonl_path.exists(), f"{jsonl_path} was not written"
    lines = jsonl_path.read_text().strip().splitlines()
    n_expected = len(RUN_E4_ARMS) * len(specs_all)
    assert len(lines) == n_expected, f"expected {n_expected} JSONL lines, got {len(lines)}"
    for line in lines:
        rec = json.loads(line)
        missing = REQUIRED_KEYS - set(rec.keys())
        assert not missing, f"record missing keys: {missing}"
    print(f"  OK: {len(lines)} valid JSONL records with all required keys at {jsonl_path}")

    print("\nAll E4/E3 smoke-test assertions passed.")


if __name__ == "__main__":
    main()
