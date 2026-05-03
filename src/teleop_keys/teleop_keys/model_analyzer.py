#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze how strongly the learned IL policy depends on the GOAL input.

- Loads best checkpoint (runs/il_nav/best_<variant>.pt).
- Recreates the same train/val/test split using the manifest and seed.
- Uses the saved lidar/goal normalization stats from the checkpoint.
- Runs:
    1) Zero-goal ablation: ||a(real_goal) - a(zero_goal)||.
    2) Gradient-based sensitivity: ||∂a/∂goal|| and per-dimension importance summary.

Assumes your training script defines:
    Cfg, NavILDataset, collate, PolicyNet, build_image_transform
exactly as in train_il_multimodal_with_plot.py.
"""

from pathlib import Path
import json
import numpy as np
import torch

from train_il_multimodal_with_plot import (
    Cfg,
    NavILDataset,
    collate,
    PolicyNet,
    build_image_transform,
)

# Use a tiny batch to avoid CUDA OOM
ANALYSIS_BATCH_SIZE = 2   # very small because backward is heavy


# -----------------------------
# Rebuild splits & datasets
# -----------------------------
def build_splits_and_datasets(cfg: Cfg, lidar_stats, goal_stats):
    """
    Recreate the same train/val/test splits and datasets as in run(cfg),
    but using lidar_stats and goal_stats loaded from the checkpoint.
    We mainly care about the val split here.
    """
    # Load manifest
    with open(cfg.manifest, "r") as f:
        all_samples = json.load(f)

    N = len(all_samples)
    n_train = int(0.7 * N)
    n_val = int(0.2 * N)
    n_test = N - n_train - n_val

    rng = np.random.RandomState(cfg.seed)
    idx = rng.permutation(N)
    tr_idx = idx[:n_train]
    va_idx = idx[n_train:n_train + n_val]
    te_idx = idx[n_train + n_val:]

    train_samples = [all_samples[i] for i in tr_idx]
    val_samples = [all_samples[i] for i in va_idx]
    test_samples = [all_samples[i] for i in te_idx]

    tf_eval = build_image_transform(False, cfg)

    # goal_stats in checkpoint was saved as (mean_list, std_list) or None
    goal_mean_std = None
    if goal_stats is not None:
        g_mean, g_std = goal_stats
        goal_mean_std = (
            np.asarray(g_mean, dtype=np.float32),
            np.asarray(g_std, dtype=np.float32),
        )

    ds_train = NavILDataset(
        train_samples,
        image_transform=tf_eval,
        lidar_replace_nan=cfg.lidar_replace_nan,
        lidar_clip=cfg.lidar_clip,
        lidar_log=cfg.lidar_log,
        lidar_norm_stats=lidar_stats,
        goal_norm_stats=goal_mean_std,
    )
    ds_val = NavILDataset(
        val_samples,
        image_transform=tf_eval,
        lidar_replace_nan=cfg.lidar_replace_nan,
        lidar_clip=cfg.lidar_clip,
        lidar_log=cfg.lidar_log,
        lidar_norm_stats=lidar_stats,
        goal_norm_stats=goal_mean_std,
    )
    ds_test = NavILDataset(
        test_samples,
        image_transform=tf_eval,
        lidar_replace_nan=cfg.lidar_replace_nan,
        lidar_clip=cfg.lidar_clip,
        lidar_log=cfg.lidar_log,
        lidar_norm_stats=lidar_stats,
        goal_norm_stats=goal_mean_std,
    )

    return ds_train, ds_val, ds_test


# -----------------------------
# Zero-goal ablation
# -----------------------------
@torch.no_grad()
def zero_goal_ablation(model, batch, device: str):
    """
    Compare actions with real goal vs goal set to zero.
    """
    goal = batch["goal"]
    if goal is None:
        print("\n[Zero-goal ablation] goal is None – nothing to test.")
        return

    goal = goal.to(device, non_blocking=True)
    image = batch["image"].to(device, non_blocking=True) if batch["image"] is not None else None
    lidar = batch["lidar"].to(device, non_blocking=True) if batch["lidar"] is not None else None

    a_real = model(image, lidar, goal)  # [B, A]

    # zero goal (same shape)
    goal_zero = torch.zeros_like(goal)
    a_zero = model(image, lidar, goal_zero)

    diff = (a_real - a_zero).norm(dim=-1).cpu().numpy()

    print("\n[Zero-goal ablation]")
    print(f"Mean ||Δaction|| (real vs zero goal): {diff.mean():.6f}")
    print(f"Median ||Δaction||:                   {np.median(diff):.6f}")
    print(f"95th percentile ||Δaction||:          {np.percentile(diff, 95):.6f}")


# -----------------------------
# Gradient-based sensitivity wrt GOAL
# -----------------------------
def gradient_goal_sensitivity(model, batch, device: str):
    """
    Gradient of actions w.r.t. goal.
    Uses only a small subset of the batch to avoid CUDA OOM.
    Summarizes per-dimension gradients.
    """
    if batch["goal"] is None:
        print("\n[Grad-based] goal is None – nothing to test.")
        return

    # --- only keep first N samples for grad analysis ---
    max_n = min(ANALYSIS_BATCH_SIZE, batch["goal"].shape[0])
    for k in ["image", "lidar", "goal"]:
        if batch[k] is not None:
            batch[k] = batch[k][:max_n]
    # --------------------------------------------------

    goal = batch["goal"].to(device, non_blocking=True)      # [B, G]
    image = batch["image"].to(device, non_blocking=True) if batch["image"] is not None else None
    lidar = batch["lidar"].to(device, non_blocking=True) if batch["lidar"] is not None else None

    # Enable gradients wrt goal
    goal = goal.clone().detach().requires_grad_(True)

    # Forward pass
    action = model(image, lidar, goal)  # [B, A]

    # Scalar to backprop from: L2 norm of actions
    loss = action.norm(dim=-1).mean()

    model.zero_grad(set_to_none=True)
    loss.backward()

    goal_grad = goal.grad.detach()  # [B, G]
    grad_norm_per_sample = goal_grad.view(goal_grad.size(0), -1).norm(dim=-1).cpu().numpy()

    print("\n[Grad-based sensitivity wrt GOAL]")
    print(f"Mean ||∂a/∂goal||:   {grad_norm_per_sample.mean():.6f}")
    print(f"Median ||∂a/∂goal||: {np.median(grad_norm_per_sample):.6f}")
    print(f"95th percentile:     {np.percentile(grad_norm_per_sample, 95):.6f}")

    # Per-dimension importance (average over batch)
    per_dim_importance = goal_grad.abs().mean(dim=0).cpu().numpy()  # [G]
    print("\nPer-dimension importance of GOAL components:")
    for i, val in enumerate(per_dim_importance):
        print(f"  dim[{i}]: {val:.6f}")


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = Cfg()
    device = cfg.device
    print(f"Using device: {device}")

    # Best checkpoint path based on your training code
    ckpt_path = Path(cfg.out_dir) / f"best_{cfg.variant}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # lidar_stats is stored as (mean, std) or None
    lidar_stats = ckpt.get("lidar_stats", None)
    # goal_stats stored as (mean_list, std_list) or None
    goal_stats = ckpt.get("goal_stats", None)

    # Build datasets using saved stats
    ds_train, ds_val, ds_test = build_splits_and_datasets(cfg, lidar_stats, goal_stats)

    # Use the validation set for analysis
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=min(ANALYSIS_BATCH_SIZE, len(ds_val)),
        shuffle=True,
        num_workers=0,  # keep simple for analysis
        pin_memory=True,
        collate_fn=collate,
    )

    # Figure out dims from one sample – just for model construction
    ex = ds_train[0]
    goal_dim = ex["goal"].numel()
    lidar_dim = ex["lidar"].numel() if ex["lidar"] is not None else None

    has_image = cfg.variant in ("image_goal", "multi")
    has_lidar = cfg.variant in ("laser_goal", "multi")

    # Build model and load weights
    model = PolicyNet(
        cfg,
        lidar_in_dim=lidar_dim,
        goal_in_dim=goal_dim,
        use_image=has_image,
        use_lidar=has_lidar,
    ).to(device)

    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    # Take one batch from val set
    batch = next(iter(dl_val))

    # 1) Zero-goal ablation
    zero_goal_ablation(model, batch, device)

    # 2) Gradient-based sensitivity
    gradient_goal_sensitivity(model, batch, device)


if __name__ == "__main__":
    main()

