#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Imitation Learning (BC) for Indoor Navigation (Weighted MSE)
- Same as your script, but adds sample-weighted MSE to handle action imbalance.

Key idea:
- Compute action-class weights from the TRAIN split (based on unique command tuples)
- For each batch, weight each sample's squared error by its class weight
"""

import math, os, random, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2

import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
class Cfg:
    data_root = Path("/media/manas/Public/Manas' folder/Synced_folder_collected_old_without_plan_freeze/Helper_scripts")
    manifest = data_root / "filtered_training_data.json"

    batch_size = 32
    epochs = 200
    lr_start = 1e-3
    lr_decay_gamma = 0.96
    lr_min = 1e-4
    weight_decay = 0.0
    num_workers = 8
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    variant = "multi"

    img_feat_dim = 32
    lidar_feat_dim = 32
    goal_feat_dim = 16
    fused_dim = 128

    lidar_replace_nan = 100.0
    lidar_clip = (0.05, 100.0)
    lidar_norm = True
    lidar_log = False

    goal_norm = True
    image_size = (224, 224)

    out_dir = Path("./runs/il_nav")
    save_every = 10

    # -------- NEW (weighting behavior) --------
    # If True, weights are normalized so avg weight ~ 1.0 (keeps loss scale stable).
    normalize_class_weights = True
    # If True, weights are clipped to avoid extreme gradients.
    class_weight_clip = (0.25, 8.0)
    # If True, compute class = unique command tuple (cmd0, cmd1).
    # This matches your discrete teleop-like data.
    use_tuple_classes = True
    # -----------------------------------------


# -----------------------------
# Utils
# -----------------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def exp_decay_with_floor(epoch: int, base_lr: float, gamma: float, floor: float):
    return max(floor / base_lr, (gamma ** epoch))


# -----------------------------
# Dataset
# -----------------------------
class NavILDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_transform=None,
        lidar_replace_nan: float = 100.0,
        lidar_clip: Tuple[float, float] = (0.05, 100.0),
        lidar_log: bool = False,
        lidar_norm_stats: Optional[Tuple[float, float]] = None,
        goal_norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.samples = samples
        self.image_transform = image_transform
        self.lidar_replace_nan = lidar_replace_nan
        self.lidar_clip = lidar_clip
        self.lidar_log = lidar_log
        self.lidar_mean_std = lidar_norm_stats
        self.goal_mean_std = goal_norm_stats

    def __len__(self): return len(self.samples)

    def _load_img(self, p: str):
        img = Image.open(p).convert("RGB")
        return self.image_transform(img) if self.image_transform else transforms.ToTensor()(img)

    def _proc_lidar(self, arr):
        x = np.asarray(arr, dtype=np.float32)
        x = np.nan_to_num(x, nan=self.lidar_replace_nan, posinf=self.lidar_replace_nan, neginf=0.0)
        lo, hi = self.lidar_clip
        x = np.clip(x, lo, hi)
        if self.lidar_log:
            x = np.log(x + 1e-3)
        if self.lidar_mean_std is not None:
            m, s = self.lidar_mean_std
            x = (x - m) / (s + 1e-8)
        return torch.from_numpy(x)

    def _proc_goal(self, g):
        g = np.asarray(g, dtype=np.float32)
        if self.goal_mean_std is not None:
            m, s = self.goal_mean_std
            g = (g - m) / (s + 1e-8)
        return torch.from_numpy(g)

    def __getitem__(self, i):
        s = self.samples[i]
        image = self._load_img(s["image_path"]) if "image_path" in s else None
        lidar = self._proc_lidar(s["lidar"]) if "lidar" in s else None
        goal  = self._proc_goal(s["goal"]) if "goal" in s else None
        cmd   = torch.tensor(s["command"], dtype=torch.float32)  # shape [2] for you
        return {"image": image, "lidar": lidar, "goal": goal, "command": cmd}


# -----------------------------
# Models
# -----------------------------
class LidarMLP(nn.Module):
    def __init__(self, in_dim: int = 900, out_dim: int = 32):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.branch_left = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.branch_right = nn.Linear(128, 64)
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.branch_left(x) + self.branch_right(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        return x

class GoalMLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 16):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.branch_left = nn.Sequential(
            nn.Linear(8, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.branch_right = nn.Linear(8, out_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.branch_left(x) + self.branch_right(x)
        x = self.fc_out(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = mobilenet_v2(weights="DEFAULT")
        features_list = list(mobilenet.features.children())
        self.mobilenet_block = nn.Sequential(*features_list[:14])

        self.right_conv = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),  # keep as in your current script
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.left_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=4, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        left = self.left_conv(x)
        left_gap = self.gap(left).view(x.size(0), 128)

        mob_feat = self.mobilenet_block(x)
        right = self.right_conv(mob_feat)
        right_gap = self.gap(right).view(x.size(0), 64)

        fused = torch.cat([left_gap, right_gap], dim=1)
        out = self.fc1(fused)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class PolicyNet(nn.Module):
    def __init__(self, cfg: Cfg, lidar_in_dim: Optional[int], goal_in_dim: int, use_image: bool, use_lidar: bool):
        super().__init__()
        self.use_image = use_image
        self.use_lidar = use_lidar

        if use_image:
            self.img_enc = ImageEncoder()
        if use_lidar:
            self.lidar_enc = LidarMLP(in_dim=lidar_in_dim, out_dim=cfg.lidar_feat_dim)
        self.goal_enc = GoalMLP(in_dim=goal_in_dim, out_dim=cfg.goal_feat_dim)

        fused_in = cfg.goal_feat_dim + (cfg.img_feat_dim if use_image else 0) + (cfg.lidar_feat_dim if use_lidar else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, cfg.fused_dim),
            nn.BatchNorm1d(cfg.fused_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Linear(cfg.fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.head = nn.Linear(16, 2)  # your policy outputs 2 dims

    def forward(self, image, lidar, goal):
        feats = []
        if self.use_image:
            feats.append(self.img_enc(image))
        if self.use_lidar:
            feats.append(self.lidar_enc(lidar))
        feats.append(self.goal_enc(goal))
        z = torch.cat(feats, dim=1)
        z = self.fusion(z)
        return self.head(z)


# -----------------------------
# Collate
# -----------------------------
def collate(batch):
    out = {}
    has_img = batch[0]["image"] is not None
    out["image"] = torch.stack([b["image"] for b in batch], 0) if has_img else None
    out["lidar"] = torch.stack([b["lidar"] for b in batch], 0) if batch[0]["lidar"] is not None else None
    out["goal"] = torch.stack([b["goal"] for b in batch], 0)
    out["command"] = torch.stack([b["command"] for b in batch], 0)
    return out


# -----------------------------
# Stats fitting
# -----------------------------
def fit_lidar_stats(samples: List[Dict[str, Any]], replace_nan: float, clip: Tuple[float, float], log: bool):
    vals = []
    for s in samples:
        x = np.asarray(s["lidar"], dtype=np.float32)
        x = np.nan_to_num(x, nan=replace_nan, posinf=replace_nan, neginf=0.0)
        x = np.clip(x, clip[0], clip[1])
        if log:
            x = np.log(x + 1e-3)
        vals.append(x)
    arr = np.concatenate(vals)
    return float(arr.mean()), float(arr.std())

def fit_goal_stats(samples: List[Dict[str, Any]]):
    G = np.stack([np.asarray(s["goal"], dtype=np.float32) for s in samples], 0)
    return G.mean(0), G.std(0)


# -----------------------------
# Image transforms
# -----------------------------
def build_image_transform(train: bool, cfg: Cfg):
    if train:
        return transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


# -----------------------------
# NEW: compute class weights from train split
# -----------------------------
def command_to_key(cmd: List[float]) -> Tuple[float, float]:
    # Your commands are effectively discrete; still keep them stable with rounding.
    return (round(float(cmd[0]), 4), round(float(cmd[1]), 4))

def compute_class_weights(train_samples: List[Dict[str, Any]], cfg: Cfg):
    """
    Returns:
      class_weight_map: dict {cmd_tuple_key -> weight}
    """
    from collections import Counter

    keys = [command_to_key(s["command"]) for s in train_samples]
    counts = Counter(keys)
    total = sum(counts.values())

    # Inverse frequency weighting
    weight_map = {}
    for k, c in counts.items():
        w = total / (len(counts) * c)  # average weight ~ 1 if perfectly balanced
        weight_map[k] = float(w)

    # Optional: normalize mean weight to 1.0 exactly
    if cfg.normalize_class_weights:
        mean_w = sum(weight_map.values()) / len(weight_map)
        for k in list(weight_map.keys()):
            weight_map[k] /= mean_w

    # Optional: clip weights for stability
    if cfg.class_weight_clip is not None:
        lo, hi = cfg.class_weight_clip
        for k in list(weight_map.keys()):
            weight_map[k] = float(np.clip(weight_map[k], lo, hi))

    print("\n[Class weights from TRAIN split]")
    for k, w in sorted(weight_map.items(), key=lambda kv: -kv[1]):
        print(f"  cmd={k} -> weight={w:.3f} | count={counts[k]}")
    return weight_map


# -----------------------------
# NEW: weighted MSE loss (per-sample)
# -----------------------------
def weighted_mse(pred: torch.Tensor, target: torch.Tensor, sample_w: torch.Tensor) -> torch.Tensor:
    """
    pred:   [B, D]
    target: [B, D]
    sample_w: [B]
    returns scalar loss
    """
    # per-sample MSE across action dims
    per_sample = ((pred - target) ** 2).mean(dim=1)  # [B]
    return (per_sample * sample_w).mean()


# -----------------------------
# Train / Eval
# -----------------------------
def run(cfg: Cfg):
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.manifest, "r") as f:
        all_samples = json.load(f)

    # Split 70/20/10
    N = len(all_samples)
    n_train = int(0.7 * N)
    n_val = int(0.2 * N)
    n_test = N - n_train - n_val

    rng = np.random.RandomState(cfg.seed)
    idx = rng.permutation(N)
    tr_idx, va_idx, te_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    train_samples = [all_samples[i] for i in tr_idx]
    val_samples = [all_samples[i] for i in va_idx]
    test_samples = [all_samples[i] for i in te_idx]

    # Fit stats on train
    lidar_stats = None
    if cfg.lidar_norm and len(train_samples) > 0 and "lidar" in train_samples[0]:
        lidar_stats = fit_lidar_stats(train_samples, cfg.lidar_replace_nan, cfg.lidar_clip, cfg.lidar_log)
    goal_stats = None
    if cfg.goal_norm:
        goal_stats = fit_goal_stats(train_samples)

    # NEW: compute class weights from TRAIN only (avoid leakage)
    class_weight_map = compute_class_weights(train_samples, cfg)

    tf_train = build_image_transform(True, cfg)
    tf_eval = build_image_transform(False, cfg)

    ds_train = NavILDataset(train_samples, image_transform=tf_train,
                            lidar_replace_nan=cfg.lidar_replace_nan,
                            lidar_clip=cfg.lidar_clip, lidar_log=cfg.lidar_log,
                            lidar_norm_stats=lidar_stats, goal_norm_stats=goal_stats)
    ds_val = NavILDataset(val_samples, image_transform=tf_eval,
                          lidar_replace_nan=cfg.lidar_replace_nan,
                          lidar_clip=cfg.lidar_clip, lidar_log=cfg.lidar_log,
                          lidar_norm_stats=lidar_stats, goal_norm_stats=goal_stats)
    ds_test = NavILDataset(test_samples, image_transform=tf_eval,
                           lidar_replace_nan=cfg.lidar_replace_nan,
                           lidar_clip=cfg.lidar_clip, lidar_log=cfg.lidar_log,
                           lidar_norm_stats=lidar_stats, goal_norm_stats=goal_stats)

    has_image = cfg.variant in ("image_goal", "multi")
    has_lidar = cfg.variant in ("laser_goal", "multi")

    ex = ds_train[0]
    goal_dim = ex["goal"].numel()
    lidar_dim = ex["lidar"].numel() if ex["lidar"] is not None else None

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate)

    model = PolicyNet(cfg, lidar_in_dim=lidar_dim, goal_in_dim=goal_dim,
                      use_image=has_image, use_lidar=has_lidar).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    lr_lambda = lambda e: exp_decay_with_floor(e, cfg.lr_start, cfg.lr_decay_gamma, cfg.lr_min)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        # -------- Train --------
        model.train()
        tr_loss = 0.0

        for batch in dl_train:
            image = batch["image"]
            if image is not None:
                image = image.to(cfg.device, non_blocking=True)
            lidar = batch["lidar"].to(cfg.device, non_blocking=True) if batch["lidar"] is not None else None
            goal = batch["goal"].to(cfg.device, non_blocking=True)
            cmd = batch["command"].to(cfg.device, non_blocking=True)

            pred = model(image, lidar, goal)  # [B,2]

            # NEW: build sample weights from the command tuples
            cmd_cpu = cmd.detach().cpu().numpy()
            keys = [command_to_key(c) for c in cmd_cpu]
            w = torch.tensor([class_weight_map.get(k, 1.0) for k in keys],
                             dtype=torch.float32, device=cfg.device)

            loss = weighted_mse(pred, cmd, w)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * cmd.size(0)

        tr_loss /= len(ds_train)

        # -------- Val (unweighted MSE for honest reporting) --------
        model.eval()
        va_loss = 0.0
        with torch.inference_mode():
            for batch in dl_val:
                image = batch["image"]
                if image is not None:
                    image = image.to(cfg.device, non_blocking=True)
                lidar = batch["lidar"].to(cfg.device, non_blocking=True) if batch["lidar"] is not None else None
                goal = batch["goal"].to(cfg.device, non_blocking=True)
                cmd = batch["command"].to(cfg.device, non_blocking=True)

                pred = model(image, lidar, goal)
                loss = ((pred - cmd) ** 2).mean()  # standard MSE (scalar)
                va_loss += loss.item() * cmd.size(0)

        va_loss /= len(ds_val)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        sched.step()
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | train(w) {tr_loss:.5f} | val {va_loss:.5f} | lr {sched.get_last_lr()[0]:.6f}")

        # Save best
        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "cfg": vars(cfg),
                "lidar_stats": lidar_stats,
                "goal_stats": (goal_stats[0].tolist(), goal_stats[1].tolist()) if goal_stats is not None else None,
                "epoch": epoch,
                "val_loss": va_loss,
                "class_weight_map": class_weight_map,  # NEW: save weights for reproducibility
            }, cfg.out_dir / f"best_{cfg.variant}_weighted.pt")

        if (epoch + 1) % cfg.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "cfg": vars(cfg),
                "epoch": epoch,
            }, cfg.out_dir / f"checkpoint_{cfg.variant}_weighted_e{epoch+1}.pt")

    # Plot losses
    epochs = range(1, cfg.epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss (weighted)")
    plt.plot(epochs, val_losses, label="Val Loss (unweighted MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (Weighted MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cfg.out_dir / f"loss_curve_{cfg.variant}_weighted.png")
    plt.show()

    # Final test (unweighted MSE)
    model.eval()
    te_loss = 0.0
    with torch.inference_mode():
        for batch in dl_test:
            image = batch["image"]
            if image is not None:
                image = image.to(cfg.device, non_blocking=True)
            lidar = batch["lidar"].to(cfg.device, non_blocking=True) if batch["lidar"] is not None else None
            goal = batch["goal"].to(cfg.device, non_blocking=True)
            cmd = batch["command"].to(cfg.device, non_blocking=True)

            pred = model(image, lidar, goal)
            loss = ((pred - cmd) ** 2).mean()
            te_loss += loss.item() * cmd.size(0)

    te_loss /= len(ds_test)
    print(f"TEST MSE (unweighted): {te_loss:.5f}")


if __name__ == "__main__":
    cfg = Cfg()
    run(cfg)

