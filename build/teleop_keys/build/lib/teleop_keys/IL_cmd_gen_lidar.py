#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
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
from torchvision import transforms
from torchvision.models import mobilenet_v2

# pygame for keyboard teleop toggle
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame


# =========================
# Config (must match training)
# =========================
class Cfg:
    variant = "multi"          # "laser_goal", "image_goal", "multi"

    img_feat_dim = 32
    lidar_feat_dim = 64        # CHANGED: must match training
    goal_feat_dim = 32
    fused_dim = 128

    lidar_skip_dim = 32        # NEW: must match training
    lidar_gate_scale = 0.30    # NEW: must match training

    # Present in training config, but only active during training
    goal_dropout_p = 0.05
    goal_dropout_scale = 0.85
    lidar_dropout_p = 0.08
    lidar_dropout_scale = 0.90

    # LiDAR preprocessing
    lidar_replace_nan = 100.0
    lidar_clip = (0.05, 100.0)
    lidar_norm = True
    lidar_log = False

    # Goal normalization
    goal_norm = True

    # Image size + normalization
    image_size = (224, 224)


# =========================
# Model definitions
# =========================
class LidarMLP(nn.Module):
    def __init__(self, in_dim: int = 900, out_dim: int = 64):
        super().__init__()

        self.input_norm = nn.LayerNorm(in_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.branch_left = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.branch_right = nn.Linear(128, 64)

        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.branch_left(x) + self.branch_right(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        return x


class GoalMLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 32):
        super().__init__()

        self.input_norm = nn.LayerNorm(in_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.branch_left = nn.Sequential(
            nn.Linear(8, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.branch_right = nn.Linear(8, out_dim)

        self.fc_out = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.01, inplace=True),
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

        # Use weights=None at runtime; checkpoint provides trained weights
        mobilenet = mobilenet_v2(weights=None)
        features_list = list(mobilenet.features.children())
        self.mobilenet_block = nn.Sequential(*features_list[:14])

        self.right_conv = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
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
    def __init__(self, cfg: Cfg, lidar_in_dim: int, goal_in_dim: int,
                 use_image: bool, use_lidar: bool):
        super().__init__()
        self.use_image = use_image
        self.use_lidar = use_lidar

        self.goal_dropout_p = cfg.goal_dropout_p
        self.goal_dropout_scale = cfg.goal_dropout_scale
        self.lidar_dropout_p = cfg.lidar_dropout_p
        self.lidar_dropout_scale = cfg.lidar_dropout_scale
        self.lidar_gate_scale = cfg.lidar_gate_scale

        if use_image:
            self.img_enc = ImageEncoder()

        if use_lidar:
            self.lidar_enc = LidarMLP(in_dim=lidar_in_dim, out_dim=cfg.lidar_feat_dim)
            self.lidar_gate = nn.Sequential(
                nn.Linear(cfg.lidar_feat_dim, 16),
                nn.LayerNorm(16),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(16, 16),
                nn.Sigmoid(),
            )
            self.lidar_skip = nn.Sequential(
                nn.Linear(cfg.lidar_feat_dim, cfg.lidar_skip_dim),
                nn.LayerNorm(cfg.lidar_skip_dim),
                nn.LeakyReLU(0.01, inplace=True),
            )

        self.goal_enc = GoalMLP(in_dim=goal_in_dim, out_dim=cfg.goal_feat_dim)

        fused_in = cfg.goal_feat_dim \
                   + (cfg.img_feat_dim if use_image else 0) \
                   + (cfg.lidar_feat_dim if use_lidar else 0)

        self.fusion = nn.Sequential(
            nn.Linear(fused_in, cfg.fused_dim),
            nn.LayerNorm(cfg.fused_dim),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(cfg.fused_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Linear(64, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(0.01, inplace=True),
        )

        head_in = 16 + cfg.goal_feat_dim + (cfg.lidar_skip_dim if use_lidar else 0)
        self.head = nn.Linear(head_in, 2)

    def _apply_weak_feature_scaling(self, feat: torch.Tensor, p: float, scale: float) -> torch.Tensor:
        # In eval mode this is a no-op, but keeping it preserves exact architecture behavior.
        if (not self.training) or p <= 0.0:
            return feat
        mask = (torch.rand(feat.size(0), 1, device=feat.device) < p).float()
        return feat * (1.0 - mask + mask * scale)

    def forward(self, image, lidar, goal):
        feats = []

        lidar_feat_raw = None
        lidar_feat = None
        lidar_skip = None

        if self.use_image:
            feats.append(self.img_enc(image))

        if self.use_lidar:
            lidar_feat_raw = self.lidar_enc(lidar)
            lidar_feat = self._apply_weak_feature_scaling(
                lidar_feat_raw, self.lidar_dropout_p, self.lidar_dropout_scale
            )
            feats.append(lidar_feat)

        goal_feat = self.goal_enc(goal)
        goal_feat = self._apply_weak_feature_scaling(
            goal_feat, self.goal_dropout_p, self.goal_dropout_scale
        )
        feats.append(goal_feat)

        z = torch.cat(feats, dim=1)
        z = self.fusion(z)

        if self.use_lidar:
            gate = self.lidar_gate(lidar_feat_raw)
            z = z * (1.0 + self.lidar_gate_scale * gate)
            lidar_skip = self.lidar_skip(lidar_feat)
            z = torch.cat([z, goal_feat, lidar_skip], dim=1)
        else:
            z = torch.cat([z, goal_feat], dim=1)

        return self.head(z)


# =========================
# ROS2 Node
# =========================
class ILLocalPolicyNode(Node):
    """
    - Subscribes: /rgb (optional), /scan_lidar, /local_goal
    - Runs IL policy (PyTorch) to output [v, w]
    - Publishes: /cmd_vel_nav2 (Twist)

    - Press 'o' to toggle override mode (teleop arrows) <-> neural net cmds
    """
    def __init__(self):
        super().__init__("il_local_policy_node")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # CHANGED: checkpoint path/name should match training output
        ckpt_path = Path("/home/manas/isaacsim_teleop/src/teleop_keys/teleop_keys/runs/il_nav/best_multi.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt["model"]
        cfg_dict = ckpt.get("cfg", {})

        self.cfg = Cfg()
        if isinstance(cfg_dict, dict):
            for k, v in cfg_dict.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)

        self.lidar_stats = ckpt.get("lidar_stats", None)
        goal_stats = ckpt.get("goal_stats", None)
        if goal_stats is not None:
            self.goal_mean = np.asarray(goal_stats[0], dtype=np.float32)
            self.goal_std = np.asarray(goal_stats[1], dtype=np.float32)
        else:
            self.goal_mean = None
            self.goal_std = None

        self.variant = self.cfg.variant
        self.use_image = self.variant in ("image_goal", "multi")
        self.use_lidar = self.variant in ("laser_goal", "multi")

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
        if self.use_lidar and lidar_in_dim is None:
            raise RuntimeError("Could not infer lidar_in_dim from checkpoint.")

        self.get_logger().info(f"Inferred lidar_in_dim={lidar_in_dim}, goal_in_dim={goal_in_dim}")

        if "fusion.0.weight" in state_dict:
            self.get_logger().info(f"Checkpoint fusion_in={state_dict['fusion.0.weight'].shape[1]}")
        if "head.weight" in state_dict:
            self.get_logger().info(f"Checkpoint head_in={state_dict['head.weight'].shape[1]}")
        if "goal_enc.branch_left.0.weight" in state_dict:
            self.get_logger().info(f"Checkpoint goal_feat_dim={state_dict['goal_enc.branch_left.0.weight'].shape[0]}")
        if "lidar_enc.fc_out.0.weight" in state_dict:
            self.get_logger().info(f"Checkpoint lidar_feat_dim={state_dict['lidar_enc.fc_out.0.weight'].shape[0]}")

        self.model = PolicyNet(
            self.cfg,
            lidar_in_dim=int(lidar_in_dim) if lidar_in_dim is not None else 0,
            goal_in_dim=int(goal_in_dim),
            use_image=self.use_image,
            use_lidar=self.use_lidar,
        ).to(self.device)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.get_logger().info("IL policy model loaded successfully.")

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

        qos = 10
        if self.use_image:
            self.sub_rgb = self.create_subscription(RosImage, "/rgb", self.rgb_callback, qos)
        if self.use_lidar:
            self.sub_lidar = self.create_subscription(LaserScan, "/scan_lidar", self.lidar_callback, qos)
        self.sub_goal = self.create_subscription(Pose2D, "/local_goal", self.goal_callback, qos)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_nav2", qos)

        self.latest_image = None
        self.latest_lidar = None
        self.latest_goal = None
        self.latest_goal_raw = None

        self.override_mode = False
        self.last_print = 0.0

        self.linear_step = 0.5
        self.angular_step = 1.0

        pygame.init()
        self.screen = pygame.display.set_mode((320, 120))
        pygame.display.set_caption("IL Local Policy (press 'o' to toggle override)")
        self.pg_clock = pygame.time.Clock()
        self.get_logger().info("Press 'o' to toggle OVERRIDE (teleop) <-> NN mode. Arrow keys drive in override. SPACE stops.")

        self.timer = self.create_timer(0.1, self.control_loop)

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
        dx = msg.x
        dy = msg.y

        self.latest_goal_raw = (float(dx), float(dy))

        g = np.array([dx, dy], dtype=np.float32)

        if self.goal_mean is not None and self.goal_std is not None and self.cfg.goal_norm:
            g = (g - self.goal_mean) / (self.goal_std + 1e-8)

        goal_t = torch.from_numpy(g).unsqueeze(0).to(self.device)
        self.latest_goal = goal_t

    def control_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    self.override_mode = not self.override_mode
                    self.get_logger().info(f"Override mode -> {self.override_mode}")

        if self.latest_goal is None:
            return
        if self.use_lidar and self.latest_lidar is None:
            return
        if self.use_image and self.latest_image is None:
            return

        if self.override_mode:
            keys = pygame.key.get_pressed()

            lin = (1 if keys[pygame.K_UP] else 0) - (1 if keys[pygame.K_DOWN] else 0)
            ang = (1 if keys[pygame.K_LEFT] else 0) - (1 if keys[pygame.K_RIGHT] else 0)

            v = float(lin * self.linear_step)
            w = float(ang * self.angular_step)

            if keys[pygame.K_SPACE]:
                v = 0.0
                w = 0.0

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.cmd_pub.publish(cmd)

            self.screen.fill((20, 20, 20))
            pygame.display.flip()
            self.pg_clock.tick(60)

            now = time.time()
            if now - self.last_print > 0.2:
                if self.latest_goal_raw is not None:
                    dx, dy = self.latest_goal_raw
                    self.get_logger().info(
                        f"[OVERRIDE] goal(dx,dy)=({dx:.3f},{dy:.3f}) | cmd(v,w)=({v:.3f},{w:.3f})"
                    )
                else:
                    self.get_logger().info(f"[OVERRIDE] goal(dx,dy)=(None) | cmd(v,w)=({v:.3f},{w:.3f})")
                self.last_print = now
            return

        with torch.inference_mode():
            image = self.latest_image if self.use_image else None
            lidar = self.latest_lidar if self.use_lidar else None
            goal = self.latest_goal

            out = self.model(image, lidar, goal)
            v = float(out[0, 0].cpu().item())
            w = float(out[0, 1].cpu().item())

        now = time.time()
        if now - self.last_print > 0.2:
            if self.latest_goal_raw is not None:
                dx, dy = self.latest_goal_raw
                self.get_logger().info(f"[NN] goal(dx,dy)=({dx:.3f},{dy:.3f}) | cmd(v,w)=({v:.3f},{w:.3f})")
            else:
                self.get_logger().info(f"[NN] goal(dx,dy)=(None) | cmd(v,w)=({v:.3f},{w:.3f})")
            self.last_print = now

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        self.screen.fill((20, 20, 20))
        pygame.display.flip()
        self.pg_clock.tick(60)


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
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
