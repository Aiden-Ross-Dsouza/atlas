"""Diagnostic script: test VideoWM.encode_act + forward_pred shapes for G2 gate."""
import sys, traceback, torch

sys.path.insert(0, '.')

import torch

# Load model (same way smoke_gates.py does)
model, prep = torch.hub.load(
    "facebookresearch/jepa-wms", "dino_wm_pusht",
    force_reload=False, trust_repo=True,
)
wm = model.model if hasattr(model, "model") else model
predictor = wm.predictor

grid   = wm.grid_size
D      = 384
N      = grid * grid
T      = 5
device = next(predictor.parameters()).device

print(f"grid_size={grid} N={N} action_dim={wm.action_dim}")
print(f"action_encoder_inpred={wm.action_encoder_inpred}")
print(f"action_conditioning={wm.action_conditioning}")
print()

# ── Test 1: _one_step_loss path (a_t_raw is [2]) ─────────────────────────────
print("=== Test 1: _one_step_loss with a_t_raw=[2] ===")
a_t_raw    = torch.randn(2, device=device)
z_cur_flat = torch.randn(N, D, device=device)
try:
    a_shaped  = a_t_raw.reshape(1, 1, -1)
    print(f"  a_shaped.shape = {a_shaped.shape}")
    act_feats = wm.encode_act(a_shaped)
    print(f"  act_feats.shape = {act_feats.shape}")
    z_cur     = z_cur_flat.reshape(1, 1, 1, grid, grid, D)
    pred_vis, _, _ = wm.forward_pred(z_cur, act_feats, None)
    print(f"  pred_vis.shape = {pred_vis.shape}   SUCCESS")
except Exception:
    traceback.print_exc()

print()

# ── Test 2: _open_loop_rollout path (actions is [T, 2]) ──────────────────────
print("=== Test 2: _open_loop_rollout with actions=[T,2] ===")
actions = torch.randn(T, 2, device=device)
try:
    a_seq = actions.unsqueeze(0)
    print(f"  a_seq.shape = {a_seq.shape}")
    act_feats_all = wm.encode_act(a_seq)
    print(f"  act_feats_all.shape = {act_feats_all.shape}")
    new_act = act_feats_all[:, 0:1]
    print(f"  new_act.shape = {new_act.shape}")
    z_cur = torch.randn(N, D, device=device).reshape(1, 1, 1, grid, grid, D)
    pred_vis, _, _ = wm.forward_pred(z_cur, new_act, None)
    print(f"  pred_vis.shape = {pred_vis.shape}   SUCCESS")
except Exception:
    traceback.print_exc()
