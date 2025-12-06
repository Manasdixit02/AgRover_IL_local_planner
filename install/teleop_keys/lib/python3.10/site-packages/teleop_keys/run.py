#!/usr/bin/env python3
import os, time, math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Ensure pygame doesn't print a welcome message
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

HELP = """
Pygame Teleop → /cmd_vel  (simultaneous keys supported)
-------------------------------------------------------
↑ / ↓ : +linear x / -linear x (holdable, combines)
← / → : +angular z (CCW) / -angular z (CW) (holdable, combines)
SPACE : stop (zero both)
q / a : linear step +/-
r / f : angular step +/-
ESC   : quit
-------------------------------------------------------
"""

class PGTeleop(Node):
    def __init__(self):
        super().__init__("teleop_keys_pygame")
        self.pub = self.create_publisher(Twist, "/cmd_vel_nav2", 10)

        # Params
        self.declare_parameter("linear_step", 0.2)    # m/s per unit
        self.declare_parameter("angular_step", 0.6)   # rad/s per unit
        self.declare_parameter("publish_rate_hz", 40) # smoother updates

        self.linear_step  = float(self.get_parameter("linear_step").value)
        self.angular_step = float(self.get_parameter("angular_step").value)
        self.rate_hz      = float(self.get_parameter("publish_rate_hz").value)

        # Pygame init (tiny window just to capture key states)
        pygame.init()
        self.screen = pygame.display.set_mode((320, 120))
        pygame.display.set_caption("ROS2 Teleop (arrows)")
        self.clock = pygame.time.Clock()

        self.last_print = 0.0
        self.get_logger().info(HELP)
        self._log_steps()

    def _log_steps(self):
        self.get_logger().info(
            f"linear_step={self.linear_step:.2f} m/s, angular_step={self.angular_step:.2f} rad/s"
        )

    def tick(self):
        # Handle events (close window, key presses for step tweaks)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_q:
                    self.linear_step += 0.05
                    self.get_logger().info(f"linear_step → {self.linear_step:.2f} m/s")
                elif event.key == pygame.K_a:
                    self.linear_step = max(0.0, self.linear_step - 0.05)
                    self.get_logger().info(f"linear_step → {self.linear_step:.2f} m/s")
                elif event.key == pygame.K_r:
                    self.angular_step += 0.05
                    self.get_logger().info(f"angular_step → {self.angular_step:.2f} rad/s")
                elif event.key == pygame.K_f:
                    self.angular_step = max(0.0, self.angular_step - 0.05)
                    self.get_logger().info(f"angular_step → {self.angular_step:.2f} rad/s")

        keys = pygame.key.get_pressed()

        # Compute commands from simultaneous key states
        lin = (1 if keys[pygame.K_UP] else 0) - (1 if keys[pygame.K_DOWN] else 0)
        ang = (1 if keys[pygame.K_LEFT] else 0) - (1 if keys[pygame.K_RIGHT] else 0)

        lin_cmd = lin * self.linear_step
        ang_cmd = ang * self.angular_step

        # Space = stop both immediately
        if keys[pygame.K_SPACE]:
            lin_cmd = 0.0
            ang_cmd = 0.0

        # Publish
        msg = Twist()
        msg.linear.x = float(lin_cmd)
        msg.angular.z = float(ang_cmd)
        self.pub.publish(msg)

        # Light UI
        self.screen.fill((20, 20, 20))
        pygame.display.flip()

        # Throttle console prints
        now = time.time()
        if now - self.last_print > 0.5:
            self.get_logger().info(f"cmd: lin={lin_cmd:.2f} m/s, ang={ang_cmd:.2f} rad/s")
            self.last_print = now

        # Timing
        self.clock.tick(self.rate_hz)
        return True


def main():
    rclpy.init()
    node = PGTeleop()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if not node.tick():
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        try:
            pygame.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()

