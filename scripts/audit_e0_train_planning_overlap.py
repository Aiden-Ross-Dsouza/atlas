"""
scripts/audit_e0_train_planning_overlap.py -- E0_RECOVERY_PLAN.md P2c.

Checks whether any real Push-T demo episode used to TRAIN an E0 chart
(source='dataset'/'hybrid' in scripts/run_e0.py, --data-split=train) is also
drawn by run_e0_planning.py::sample_dataset_init_goal for planning seeds
0..(--num-planning-seeds - 1). If a chart trains on the same episode its
planning eval is drawn from, that is leakage.

Reproduces sample_dataset_init_goal's own episode_idx draw (it doesn't return
episode_idx directly) by replaying the identical RandomState call sequence,
and asserts the reproduced init_state matches the real one -- so a silent
drift between this script and the real function's logic fails loudly instead
of quietly reporting a wrong (but plausible-looking) answer.

Usage:
    python scripts/audit_e0_train_planning_overlap.py --manifest atlas_out/e0_v2/e0_seed_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_e0_planning import GOAL_TRAJ_LEN, load_dataset_states, sample_dataset_init_goal  # noqa: E402


def reproduce_planning_episodes(states, seq_lengths, num_seeds: int, min_block_pos_diff: float) -> set[int]:
    valid_eps = [i for i, l in enumerate(seq_lengths) if l >= GOAL_TRAJ_LEN]
    eps = set()
    for seed in range(num_seeds):
        rs = np.random.RandomState(seed)
        init_state, _goal_state = sample_dataset_init_goal(
            states, seq_lengths, rs, min_block_pos_diff=min_block_pos_diff)

        rs2 = np.random.RandomState(seed)
        ep_idx = None
        for _attempt in range(20):
            ep_idx = valid_eps[rs2.randint(len(valid_eps))]
            max_offset = seq_lengths[ep_idx] - GOAL_TRAJ_LEN
            offset = rs2.randint(max_offset + 1) if max_offset > 0 else 0
            init_s = states[ep_idx, offset]
            goal_s = states[ep_idx, offset + GOAL_TRAJ_LEN - 1]
            if np.linalg.norm(goal_s[2:4] - init_s[2:4]) >= min_block_pos_diff:
                break
        assert np.allclose(init_s, init_state[:5]), (
            f"seed {seed}: reproduced init_state does not match sample_dataset_init_goal's own "
            f"draw -- this script's reproduction has drifted from the real function, fix before trusting output")
        eps.add(ep_idx)
    return eps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                         help="e0_seed_manifest.json from an E0 run (scripts/run_e0.py --out ...).")
    parser.add_argument("--num-planning-seeds", type=int, default=20)
    parser.add_argument("--min-block-pos-diff", type=float, default=40.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    train_eps, eval_eps = set(), set()
    for _regime, d in manifest.items():
        train_eps.update(r["episode_idx"] for r in d.get("train", []) if r["episode_idx"] is not None)
        eval_eps.update(r["episode_idx"] for r in d.get("eval", []) if r["episode_idx"] is not None)

    print(f"E0 TRAIN episode_idx set (n={len(train_eps)}): {sorted(train_eps)}")
    print(f"E0 offline-EVAL episode_idx set (n={len(eval_eps)}): {sorted(eval_eps)}")

    states, seq_lengths = load_dataset_states()
    planning_eps = reproduce_planning_episodes(
        states, seq_lengths, args.num_planning_seeds, args.min_block_pos_diff)
    print(f"\nPlanning episode_idx set for seeds 0..{args.num_planning_seeds - 1} "
          f"(n={len(planning_eps)}): {sorted(planning_eps)}")

    overlap_train = train_eps & planning_eps
    overlap_eval = eval_eps & planning_eps
    print(f"\nOverlap (TRAIN vs planning): {sorted(overlap_train)}")
    print(f"Overlap (offline-EVAL vs planning): {sorted(overlap_eval)}")

    if overlap_train:
        print("\n*** LEAKAGE: a chart trained on an episode its planning eval also draws from ***")
        raise SystemExit(1)
    print("\nNo train/planning-eval overlap -- no leakage detected.")


if __name__ == "__main__":
    main()
