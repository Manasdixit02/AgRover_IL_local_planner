#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Imitation Learning (BC) for Indoor Navigation
- Modalities: RGB image, 2D LiDAR, goal vector (relative pose)
- Three variants: (laser+goal), (image+goal), (laser+image+goal)
- Encoders:
    * Image: MobileNetV2 backbone (pretrained) + small head
    * LiDAR: MLP + residual blocks
    * Goal:  MLP + residual block (smaller dim)
- Fusion: concat -> MLP
- Target: expert velocities [v_x, v_y, ω]
- Loss: MSE (Behavioral Cloning)
- Optim: Adam + exponential LR decay with floor

This mirrors the paper's methodology/training details.
"""

import math, os, random, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import mobilenet_v2

import matplotlib.pyplot as plt   # <-- NEW

# -----------------------------
# Config
# -----------------------------
class Cfg:
    # Data: you should populate a manifest.json with a list of samples
    # each sample: {"image_path": "...", "lidar": [...], "goal": [...], "command": [vx, vy, wz]}
    data_root = Path("/home/manas/isaacsim_teleop/src/teleop_keys/teleop_keys")
    manifest = data_root / "filtered_training_data.json"

    # Training settings (paper: bs=128, epochs=200, Adam, exp decay from 0.005 with floor)
    batch_size = 32 #128   #inc the number reduces the learning rate and reducing this increases unstability
    epochs = 200
    lr_start = 1e-3 #5e-3
    lr_decay_gamma = 0.96  # per-epoch factor; tune as needed
    lr_min = 1e-4          # floor
    weight_decay = 0.0
    num_workers = 8
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Modalities to use: "laser_goal", "image_goal", or "multi"
    variant = "multi"

    # Feature dims (paper: image/laser feature 32, goal feature 16 before fusion)
    img_feat_dim = 32
    lidar_feat_dim = 32
    goal_feat_dim = 16
    fused_dim = 128 #256  #might need to change

    # LiDAR preprocessing
    lidar_replace_nan = 100.0  # "infinite" distance
    lidar_clip = (0.05, 100.0) 
    lidar_norm = True         # zero-mean/unit-var (fit on train)
    lidar_log = False         # optional

    # Goal normalization (zero-mean/unit-var fit on train)
    goal_norm = True

    # Image size and normalization (use ImageNet stats for MobileNetV2)
    image_size = (224, 224) #needs to be changed

    # Checkpoint/output
    out_dir = Path("./runs/il_nav")
    save_every = 10 #change based on the memory


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
    """
    Expects a manifest.json list of dicts:
      {
        "image_path": str,         # path to RGB image
        "lidar": [float,...],      # 1D scan
        "goal":  [float,...],      # e.g., [d, theta] or [dx, dy, dtheta]
        "command":[vx, vy, wz]     # expert joystick command
      }
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_transform=None,
        lidar_replace_nan: float = 100.0,
        lidar_clip: Tuple[float, float] = (0.05, 100.0),
        lidar_log: bool = False,
        lidar_norm_stats: Optional[Tuple[float, float]] = None,  # (mean, std)
        goal_norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (mean_vec, std_vec)
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
        cmd   = torch.tensor(s["command"], dtype=torch.float32)  # [vx, vy, wz]
        return {"image": image, "lidar": lidar, "goal": goal, "command": cmd}


# -----------------------------
# Models
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return F.leaky_relu(x + self.net(x), negative_slope=0.01, inplace=True)

class LidarMLP(nn.Module):
    def __init__(self, in_dim: int = 900, out_dim: int = 32):
        super().__init__()

        # "Normalization" block at the top
        # You can also use BatchNorm1d(in_dim) if you prefer
        self.input_norm = nn.LayerNorm(in_dim)

        # 900 -> 128 : Dense + BN + LeakyReLU
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Left branch: 128 -> 64 : Dense + BN + LeakyReLU
        self.branch_left = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Right branch: 128 -> 64 : plain Dense (for Add)
        self.branch_right = nn.Linear(128, 64)

        # After Add: 64 -> 64 : Dense + BN + LeakyReLU
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Final: 64 -> 32 : Dense + BN + LeakyReLU
        self.fc_out = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        """
        x: [batch_size, 513]
        returns: [batch_size, 32]  (Laser Feature Vector)
        """
        x = self.input_norm(x)          # Normalization (513)
        x = self.fc1(x)                 # 513 -> 128

        left = self.branch_left(x)      # 128 -> 64
        right = self.branch_right(x)    # 128 -> 64

        x = left + right                # "Add" layer: element-wise sum (64)

        x = self.fc2(x)                 # 64 -> 64
        x = self.fc_out(x)              # 64 -> 32

        return x

class GoalMLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 16):
        super().__init__()

        # "Normalization" block at the top
        # You can also use BatchNorm1d(in_dim) if you prefer
        self.input_norm = nn.LayerNorm(in_dim)

        # 900 -> 128 : Dense + BN + LeakyReLU
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Left branch: 128 -> 64 : Dense + BN + LeakyReLU
        self.branch_left = nn.Sequential(
            nn.Linear(8, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Right branch: 128 -> 64 : plain Dense (for Add)
        self.branch_right = nn.Linear(8, out_dim)

        # After Add: 64 -> 64 : Dense + BN + LeakyReLU
        self.fc_out = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        """
        x: [batch_size, 513]
        returns: [batch_size, 32]  (Laser Feature Vector)
        """
        x = self.input_norm(x)          # Normalization (513)
        x = self.fc1(x)                 # 513 -> 128

        left = self.branch_left(x)      # 128 -> 64
        right = self.branch_right(x)    # 128 -> 64

        x = left + right                # "Add" layer: element-wise sum (64)

        x = self.fc_out(x)                 # 64 -> 64

        return x


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # --------------------------
        # 1. Rescaling layer
        # --------------------------
        self.rescale = nn.Identity()   # divide by 255 inside forward()

        # --------------------------
        # 2. MobileNetV2 feature extractor
        # --------------------------
        mobilenet = mobilenet_v2(weights='DEFAULT')

        # Extract features up to block_13_expand_relu
        features_list = list(mobilenet.features.children())
        # block13 = layer index 13
        self.mobilenet_block = nn.Sequential(*features_list[:14])
        # Output of this block: (batch, 576, 14, 14)

        # --------------------------
        # 3. Right Conv2D Branch (MobileNet output)
        # --------------------------
        self.right_conv = nn.Sequential(
            #nn.Conv2d(576, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # --------------------------
        # 4. Left Conv2D Branch (Raw image)
        # --------------------------
        self.left_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 224→112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 112→56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=4, padding=0), # 56→14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # Output: (batch, 128, 14, 14)

        # --------------------------
        # 5. Global Average Pooling (left branch)
        # --------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # → (batch, 128, 1, 1)

        # --------------------------
        # 6. Dense Fusion Layers
        # --------------------------
        self.fc1 = nn.Sequential(
            nn.Linear(128 + 64, 128),  # concat left GAP + right pooled
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
        """
        Input: x = RGB image (B, 3, 224, 224)
        Output: image feature vector (B, 32)
        """

        # 1. Rescaling
        #x = x / 255.0

        # 2. Raw image left conv branch
        left = self.left_conv(x)               # (B, 128, 14, 14)
        left_gap = self.gap(left).view(x.size(0), 128)

        # 3. MobileNetV2 right branch
        mob_feat = self.mobilenet_block(x)     # (B, 576, 14, 14)
        right = self.right_conv(mob_feat)      # (B, 64, 14, 14)
        right_gap = self.gap(right).view(x.size(0), 64)

        # 4. Concatenate left + right features
        fused = torch.cat([left_gap, right_gap], dim=1)   # (B, 128 + 64 = 192)

        # 5. Dense fusion layers
        out = self.fc1(fused)                  # 192→128
        out = self.fc2(out)                    # 128→64
        out = self.fc3(out)                    # 64→32

        return out

class PolicyNet(nn.Module):
    def __init__(self, cfg: Cfg, lidar_in_dim: Optional[int], goal_in_dim: int, use_image: bool, use_lidar: bool):
        super().__init__()
        self.use_image = use_image
        self.use_lidar = use_lidar

        if use_image:
            self.img_enc = ImageEncoder()  # MobileNetV2 per paper
        if use_lidar:
            self.lidar_enc = LidarMLP(in_dim=lidar_in_dim, out_dim=cfg.lidar_feat_dim)
        self.goal_enc = GoalMLP(in_dim=goal_in_dim, out_dim=cfg.goal_feat_dim)

        fused_in = cfg.goal_feat_dim + (cfg.img_feat_dim if use_image else 0) + (cfg.lidar_feat_dim if use_lidar else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, cfg.fused_dim), nn.BatchNorm1d(cfg.fused_dim), nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Linear(cfg.fused_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Linear(64, 16), nn.BatchNorm1d(16), nn.LeakyReLU(negative_slope=0.01,inplace=True),
        )
        # Output head: predict [vx, vy, wz]
        self.head = nn.Linear(16, 2)  #changed from 3 to 2

    def forward(self, image, lidar, goal):
        feats = []
        if self.use_image:
            feats.append(self.img_enc(image))            # [B, img_feat]
        if self.use_lidar:
            feats.append(self.lidar_enc(lidar))           # [B, lidar_feat]
        feats.append(self.goal_enc(goal))                 # [B, goal_feat]
        z = torch.cat(feats, dim=1)
        z = self.fusion(z)
        return self.head(z)  # [B,3]
# -----------------------------
# Collate (all are fixed-size here)
# -----------------------------
def collate(batch):
    out = {}
    # Some samples may not have an image if you're training a laser-only variant
    has_img = batch[0]["image"] is not None
    if has_img:
        out["image"] = torch.stack([b["image"] for b in batch], 0)
    else:
        out["image"] = None
    out["lidar"]   = torch.stack([b["lidar"] for b in batch], 0) if batch[0]["lidar"] is not None else None
    out["goal"]    = torch.stack([b["goal"] for b in batch], 0)
    out["command"] = torch.stack([b["command"] for b in batch], 0)
    return out

# -----------------------------
# Stats fitting
# -----------------------------
def fit_lidar_stats(samples: List[Dict[str, Any]], replace_nan: float, clip: Tuple[float,float], log: bool):
    vals = []
    for s in samples:
        x = np.asarray(s["lidar"], dtype=np.float32)
        x = np.nan_to_num(x, nan=replace_nan, posinf=replace_nan, neginf=0.0)
        x = np.clip(x, clip[0], clip[1])
        if log: x = np.log(x + 1e-3)
        vals.append(x)
    arr = np.concatenate(vals)
    return float(arr.mean()), float(arr.std())

def fit_goal_stats(samples: List[Dict[str, Any]]):
    G = np.stack([np.asarray(s["goal"], dtype=np.float32) for s in samples], 0)
    return G.mean(0), G.std(0)

# -----------------------------
# Image transforms (aug as in paper: brightness/contrast/saturation jitter)
# -----------------------------
def build_image_transform(train: bool, cfg: Cfg):
    if train:
        return transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

# -----------------------------
# Training / Eval
# -----------------------------
def run(cfg: Cfg):
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(cfg.manifest, "r") as f:
        all_samples = json.load(f)

    # Split 70/20/10 (paper)
    N = len(all_samples)
    n_train = int(0.7 * N)
    n_val   = int(0.2 * N)
    n_test  = N - n_train - n_val
    # stable split with seed
    rng = np.random.RandomState(cfg.seed)
    idx = rng.permutation(N)
    tr_idx, va_idx, te_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    train_samples = [all_samples[i] for i in tr_idx]
    val_samples   = [all_samples[i] for i in va_idx]
    test_samples  = [all_samples[i] for i in te_idx]

    # Fit stats on train (LiDAR and goal)
    lidar_stats = None
    if cfg.lidar_norm and len(train_samples) > 0 and "lidar" in train_samples[0]:
        lidar_stats = fit_lidar_stats(train_samples, cfg.lidar_replace_nan, cfg.lidar_clip, cfg.lidar_log)
    goal_stats = None
    if cfg.goal_norm:
        goal_stats = fit_goal_stats(train_samples)

    # Build datasets
    tf_train = build_image_transform(True, cfg)
    tf_eval  = build_image_transform(False, cfg)

    ds_train = NavILDataset(train_samples, image_transform=tf_train,
                            lidar_replace_nan=cfg.lidar_replace_nan,
                            lidar_clip=cfg.lidar_clip, lidar_log=cfg.lidar_log,
                            lidar_norm_stats=lidar_stats, goal_norm_stats=goal_stats)
    ds_val   = NavILDataset(val_samples, image_transform=tf_eval,
                            lidar_replace_nan=cfg.lidar_replace_nan,
                            lidar_clip=cfg.lidar_clip, lidar_log=cfg.lidar_log,
                            lidar_norm_stats=lidar_stats, goal_norm_stats=goal_stats)
    ds_test  = NavILDataset(test_samples, image_transform=tf_eval,
                            lidar_replace_nan=cfg.lidar_replace_nan,
                            lidar_clip=cfg.lidar_clip, lidar_log=cfg.lidar_log,
                            lidar_norm_stats=lidar_stats, goal_norm_stats=goal_stats)

    # Infer dimensions
    has_image = cfg.variant in ("image_goal", "multi")
    has_lidar = cfg.variant in ("laser_goal", "multi")

    # Sample shapes
    ex = ds_train[0]
    goal_dim = ex["goal"].numel()
    lidar_dim = ex["lidar"].numel() if ex["lidar"] is not None else None

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate)
    dl_test  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate)

    # Model
    model = PolicyNet(cfg, lidar_in_dim=lidar_dim, goal_in_dim=goal_dim,
                      use_image=has_image, use_lidar=has_lidar).to(cfg.device)

    # Optim & LR scheduler (exp decay + floor)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    lr_lambda = lambda e: exp_decay_with_floor(e, cfg.lr_start, cfg.lr_decay_gamma, cfg.lr_min)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    # Loss (MSE BC)
    mse = nn.MSELoss()

    # -----------------
    # Track losses
    # -----------------
    train_losses = []   # <-- NEW
    val_losses = []     # <-- NEW

    # Train
    best_val = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for batch in dl_train:
            image = batch["image"]
            if image is not None: image = image.to(cfg.device, non_blocking=True)
            lidar = batch["lidar"].to(cfg.device, non_blocking=True) if batch["lidar"] is not None else None
            goal  = batch["goal"].to(cfg.device, non_blocking=True)
            cmd   = batch["command"].to(cfg.device, non_blocking=True)

            pred = model(image, lidar, goal)  # [B,3]
            loss = mse(pred, cmd)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * cmd.size(0)

        tr_loss /= len(ds_train)

        # Val
        model.eval()
        va_loss = 0.0
        with torch.inference_mode():
            for batch in dl_val:
                image = batch["image"]
                if image is not None: image = image.to(cfg.device, non_blocking=True)
                lidar = batch["lidar"].to(cfg.device, non_blocking=True) if batch["lidar"] is not None else None
                goal  = batch["goal"].to(cfg.device, non_blocking=True)
                cmd   = batch["command"].to(cfg.device, non_blocking=True)
                pred = model(image, lidar, goal)
                loss = mse(pred, cmd)
                va_loss += loss.item() * cmd.size(0)
        va_loss /= len(ds_val)

        # record losses
        train_losses.append(tr_loss)   # <-- NEW
        val_losses.append(va_loss)     # <-- NEW

        sched.step()

        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | train {tr_loss:.5f} | val {va_loss:.5f} | lr {sched.get_last_lr()[0]:.6f}")

        # Save best
        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "cfg": vars(cfg),
                "lidar_stats": lidar_stats,
                "goal_stats": (goal_stats[0].tolist(), goal_stats[1].tolist()) if goal_stats is not None else None,
                "epoch": epoch,
                "val_loss": va_loss
            }, cfg.out_dir / f"best_{cfg.variant}.pt")

        if (epoch + 1) % cfg.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "cfg": vars(cfg),
                "epoch": epoch,
            }, cfg.out_dir / f"checkpoint_{cfg.variant}_e{epoch+1}.pt")

    # -------------
    # Plot losses
    # -------------
    epochs = range(1, cfg.epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cfg.out_dir / f"loss_curve_{cfg.variant}.png")
    plt.show()  # optionally enable if you want an interactive window

    # Final test
    model.eval()
    te_loss = 0.0
    with torch.inference_mode():
        for batch in dl_test:
            image = batch["image"]
            if image is not None: image = image.to(cfg.device, non_blocking=True)
            lidar = batch["lidar"].to(cfg.device, non_blocking=True) if batch["lidar"] is not None else None
            goal  = batch["goal"].to(cfg.device, non_blocking=True)
            cmd   = batch["command"].to(cfg.device, non_blocking=True)
            pred = model(image, lidar, goal)
            loss = mse(pred, cmd)
            te_loss += loss.item() * cmd.size(0)
    te_loss /= len(ds_test)
    print(f"TEST MSE: {te_loss:.5f}")

if __name__ == "__main__":
    cfg = Cfg()
    run(cfg)

