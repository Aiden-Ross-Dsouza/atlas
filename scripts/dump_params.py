"""
scripts/dump_params.py — Day-2 reconnaissance: dump predictor parameter names.

Implementation plan §5.1: before writing any chart code, inspect the predictor
to confirm LN vs AdaLN and count parameter surfaces.

Usage:
    python scripts/dump_params.py
"""

from __future__ import annotations

import torch
import atlas  # sets TORCH_HOME


def main() -> None:
    print("Loading dino_wm_pusht via torch.hub...")
    model, prep = torch.hub.load(
        "facebookresearch/jepa-wms", "dino_wm_pusht",
        force_reload=False, trust_repo=True,
    )
    wm = model.model if hasattr(model, "model") else model
    predictor = wm.predictor
    total = sum(p.numel() for p in predictor.parameters())
    print(f"\nPredictor total params: {total:,}")

    # ln_act chart is LN-affine ONLY (10,764 params), NOT "LN + action
    # encoder": the action encoder is a SIBLING of the predictor
    # (video_wm.py:82-83), so a Chart built on `predictor.named_parameters()`
    # alone can never reach it structurally -- the `is_act` filter below finds
    # nothing (0 matches) against the real predictor's own parameters, which
    # is exactly the evidence for that (E0_IMPLEMENTATION_PLAN.md T12 #10).
    # Kept as a check, not deleted, so a future checkpoint with a differently
    # structured predictor would visibly surface any action-named params here.
    print("\n── LN affine (ndim ≤ 1 with 'norm' or 'ln' in name) — 'action' filter is a structural no-op, see comment ──")
    ln_act_count = 0
    for n, p in predictor.named_parameters():
        is_ln = (p.ndim == 1) and ("norm" in n or "ln" in n)
        is_act = "action" in n
        if is_ln or is_act:
            tag = "[LN]" if is_ln else "[ACT]"
            print(f"  {tag} {n:60s} {tuple(p.shape)}")
            ln_act_count += p.numel()
    print(f"  Subtotal: {ln_act_count:,} params")

    print("\n── Attention qkv/proj (LoRA targets) ──")
    lora_count = 0
    for n, p in predictor.named_parameters():
        if any(k in n for k in ("qkv", "proj", "q_proj", "k_proj", "v_proj", "out_proj")):
            print(f"  {n:60s} {tuple(p.shape)}")
            lora_count += p.numel()
    print(f"  Subtotal (base weights, before LoRA injection): {lora_count:,} params")

    print("\n── AdaLN check: look for conditioning MLPs ──")
    adaln_found = False
    for n, p in predictor.named_parameters():
        if "adaln" in n.lower() or "adaLN" in n or "adanorm" in n.lower():
            print(f"  [AdaLN] {n:60s} {tuple(p.shape)}")
            adaln_found = True
    if not adaln_found:
        print("  No AdaLN parameters found — plain LN confirmed. ln_act surface is valid.")
    else:
        print("  WARNING: AdaLN detected. LN affine surface may be empty.")
        print("  Target the AdaLN conditioning MLP's output projection instead.")


if __name__ == "__main__":
    main()
