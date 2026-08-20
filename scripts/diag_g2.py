import torch
from atlas.chart import Chart

model, prep = torch.hub.load(
    "facebookresearch/jepa-wms", "dino_wm_pusht",
    force_reload=False, trust_repo=True,
)
wm = model.model if hasattr(model, "model") else model
predictor = wm.predictor

chart = Chart(predictor, "ln_act")
chart.apply_(predictor)

for n, p in predictor.named_parameters():
    if n in chart._param_names:
        p.requires_grad_(True)
    else:
        p.requires_grad_(False)

grid = wm.grid_size
D = 384
z_vis = torch.randn(1, 1, 1, grid, grid, D)
actions = torch.randn(1, 1, 10)  # 1 step, 10-D action

act_emb = wm.encode_act(actions)  # [1, 1, 256, 10]
print(f"act_emb shape: {act_emb.shape}")

prop_emb = torch.zeros(1, 1, grid * grid, 20)  # [1, 1, 256, 20]

pred_video_features, _, _ = wm.forward_pred(
    z_vis,
    act_emb,
    prop_emb,
)

print(f"pred_video_features shape: {pred_video_features.shape}")
print(f"pred_video_features requires_grad: {pred_video_features.requires_grad}")
print(f"pred_video_features grad_fn: {pred_video_features.grad_fn}")
