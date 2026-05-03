#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D, Twist

from cv_bridge import CvBridge

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2


# =========================
# Minimal Config (runtime)
# =========================
class Cfg:
    # Only the fields that matter for the model + preprocessing
    variant = "multi"          # "laser_goal", "image_goal", "multi"

    img_feat_dim = 32
    lidar_feat_dim = 32
    goal_feat_dim = 16
    fused_dim = 128

    # LiDAR preprocessing
    lidar_replace_nan = 100.0
    lidar_clip = (0.05, 100.0)
    lidar_norm = True
    lidar_log = False

    # Goal normalization
    goal_norm = True

    # Image size + normalization (ImageNet stats)
    image_size = (224, 224)


# =========================
# Model definitions
# =========================
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

        left = self.branch_left(x)
        right = self.branch_right(x)

        x = left + right
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

        left = self.branch_left(x)
        right = self.branch_right(x)

        x = left + right
        x = self.fc_out(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Just need the architecture; weights will come from checkpoint
        mobilenet = mobilenet_v2(weights=None)

        features_list = list(mobilenet.features.children())
        # Up to block 13
        self.mobilenet_block = nn.Sequential(*features_list[:14])

        self.right_conv = nn.Sequential(
            # For MobileNetV2 block_13 output: 96 channels
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )

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
        # Left raw-image branch
        left = self.left_conv(x)
        left_gap = self.gap(left).view(x.size(0), 128)

        # Right MobileNet branch
        mob_feat = self.mobilenet_block(x)
        right = self.right_conv(mob_feat)
        right_gap = self.gap(right).view(x.size(0), 64)

        fused = torch.cat([left_gap, right_gap], dim=1)

        out = self.fc1(fused)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class PolicyNet(nn.Module):
    def __init__(self, cfg: Cfg, lidar_in_dim: int, goal_in_dim: int,
                 use_image: bool, use_lidar: bool):
        super().__init__()
        self.use_image = use_image
        self.use_lidar = use_lidar

        if use_image:
            self.img_enc = ImageEncoder()
        if use_lidar:
            self.lidar_enc = LidarMLP(in_dim=lidar_in_dim, out_dim=cfg.lidar_feat_dim)

        self.goal_enc = GoalMLP(in_dim=goal_in_dim, out_dim=cfg.goal_feat_dim)

        fused_in = cfg.goal_feat_dim
        if use_image:
            fused_in += cfg.img_feat_dim
        if use_lidar:
            fused_in += cfg.lidar_feat_dim

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

        # Head predicts [v, w] (you changed from 3 → 2)
        self.head = nn.Linear(16, 2)

    def forward(self, image, lidar, goal):
        feats = []
        if self.use_image:
            feats.append(self.img_enc(image))
        if self.use_lidar:
            feats.append(self.lidar_enc(lidar))
        feats.append(self.goal_enc(goal))
        z = torch.cat(feats, dim=1)
        z = self.fusion(z)
        return self.head(z)  # [B, 2]


# =========================
# ROS2 Node
# =========================
class ILLocalPolicyNode(Node):
    """
    - Subscribes: /rgb (optional), /scan_lidar, /local_goal
    - Runs IL policy (PyTorch) to output [v, w]
    - Publishes: /cmd_vel (Twist)
    """
    def __init__(self):
        super().__init__("il_local_policy_node")

        # ---------- Device ----------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # ---------- Load model ----------
        # Adjust path to your best checkpoint
        ckpt_path = Path("/home/manas/isaacsim_teleop/src/teleop_keys/teleop_keys/runs/il_nav/best_multi.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt["model"]
        cfg_dict = ckpt["cfg"]

        # Rebuild cfg and override defaults
        self.cfg = Cfg()
        for k, v in cfg_dict.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        # Normalization stats from training
        self.lidar_stats = ckpt.get("lidar_stats", None)
        goal_stats = ckpt.get("goal_stats", None)
        if goal_stats is not None:
            self.goal_mean = np.asarray(goal_stats[0], dtype=np.float32)
            self.goal_std = np.asarray(goal_stats[1], dtype=np.float32)
        else:
            self.goal_mean = None
            self.goal_std = None

        # Use modalities according to training variant
        self.variant = self.cfg.variant
        self.use_image = self.variant in ("image_goal", "multi")
        self.use_lidar = self.variant in ("laser_goal", "multi")

        # Infer input dims from checkpoint (so you don't hardcode 513/900 etc)
        lidar_in_dim = None
        if self.use_lidar:
            for k in state_dict.keys():
                if "lidar_enc.input_norm.weight" in k:
                    lidar_in_dim = state_dict[k].shape[0]
                    break

        goal_in_dim = None
        for k in state_dict.keys():
            if "goal_enc.input_norm.weight" in k:
                goal_in_dim = state_dict[k].shape[0]
                break

        if goal_in_dim is None:
            raise RuntimeError("Could not infer goal_in_dim from checkpoint.")

        self.get_logger().info(f"Inferred lidar_in_dim={lidar_in_dim}, goal_in_dim={goal_in_dim}")

        # Create model and load weights
        self.model = PolicyNet(self.cfg,
                               lidar_in_dim=lidar_in_dim if lidar_in_dim is not None else 0,
                               goal_in_dim=goal_in_dim,
                               use_image=self.use_image,
                               use_lidar=self.use_lidar).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.get_logger().info("IL policy model loaded successfully.")

        # ---------- Image transform (eval-time) ----------
        if self.use_image:
            self.image_transform = transforms.Compose([
                transforms.Resize(self.cfg.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
            ])
            self.bridge = CvBridge()
        else:
            self.image_transform = None
            self.bridge = None

        # ---------- Subscriptions & Publisher ----------
        qos = 10
        if self.use_image:
            self.sub_rgb = self.create_subscription(
                RosImage, "/rgb", self.rgb_callback, qos
            )

        if self.use_lidar:
            self.sub_lidar = self.create_subscription(
                LaserScan, "/scan_lidar", self.lidar_callback, qos
            )

        self.sub_goal = self.create_subscription(
            Pose2D, "/local_goal", self.goal_callback, qos
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_nav2", qos)

        # ---------- State buffers ----------
        self.latest_image = None   # torch.Tensor [1,3,H,W] or None
        self.latest_lidar = None   # torch.Tensor [1,N] or None
        self.latest_goal = None    # torch.Tensor [1,G]

        # Control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

    # -------------------------
    # Callbacks
    # -------------------------
    def rgb_callback(self, msg: RosImage):
        if not self.use_image:
            return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil_img = Image.fromarray(cv_img)
            img_t = self.image_transform(pil_img).unsqueeze(0).to(self.device)
            self.latest_image = img_t
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def lidar_callback(self, msg: LaserScan):
        if not self.use_lidar:
            return
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Same preprocessing as training
        ranges = np.nan_to_num(
            ranges,
            nan=self.cfg.lidar_replace_nan,
            posinf=self.cfg.lidar_replace_nan,
            neginf=0.0,
        )
        lo, hi = self.cfg.lidar_clip
        ranges = np.clip(ranges, lo, hi)
        if self.cfg.lidar_log:
            ranges = np.log(ranges + 1e-3)

        if self.lidar_stats is not None and self.cfg.lidar_norm:
            m, s = self.lidar_stats
            ranges = (ranges - m) / (s + 1e-8)

        lidar_t = torch.from_numpy(ranges.astype(np.float32)).unsqueeze(0).to(self.device)
        self.latest_lidar = lidar_t

    def goal_callback(self, msg: Pose2D):
        # Assume dataset goal was [distance, heading] in robot frame
        dx = msg.x
        dy = msg.y
        #d = math.hypot(dx, dy)
        #theta = math.atan2(dy, dx)

        #g = np.array([d, theta], dtype=np.float32)
        g = np.array([dx, dy], dtype=np.float32)

        if self.goal_mean is not None and self.goal_std is not None and self.cfg.goal_norm:
            g = (g - self.goal_mean) / (self.goal_std + 1e-8)

        goal_t = torch.from_numpy(g).unsqueeze(0).to(self.device)
        self.latest_goal = goal_t

    # -------------------------
    # Control loop
    # -------------------------
    def control_loop(self):
        # Need goal + whatever modalities we trained with
        if self.latest_goal is None:
            return
        if self.use_lidar and self.latest_lidar is None:
            return
        if self.use_image and self.latest_image is None:
            return

        with torch.inference_mode():
            image = self.latest_image if self.use_image else None
            lidar = self.latest_lidar if self.use_lidar else None
            goal = self.latest_goal

            out = self.model(image, lidar, goal)  # [1, 2]
            v = float(out[0, 0].cpu().item())
            w = float(out[0, 1].cpu().item())

        # Build Twist
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w

        self.cmd_pub.publish(cmd)


# =========================
# main
# =========================
def main(args=None):
    rclpy.init(args=args)
    node = ILLocalPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

