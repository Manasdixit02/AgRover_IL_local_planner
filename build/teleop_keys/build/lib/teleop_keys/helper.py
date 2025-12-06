import torch

ckpt_path = "runs/il_nav/best_multi.pt"  # adjust if needed
ckpt = torch.load(ckpt_path, map_location="cpu")

lidar_mean, lidar_std = ckpt["lidar_stats"]
print("LiDAR mean:", lidar_mean)
print("LiDAR std:", lidar_std)

goal_mean, goal_std = ckpt["goal_stats"]
print("Goal mean:", goal_mean)
print("Goal std:", goal_std)
