#!/usr/bin/env python3

import csv
import math
import os
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from simulation_interfaces.srv import SetSimulationState
from simulation_interfaces.msg import SimulationState

import tf2_ros
from tf2_ros import TransformException

import time


class EvaluatorNode(Node):
    def __init__(self):
        super().__init__("evaluator_node")

        self.declare_parameter("scan_topic", "/scan_lidar")
        self.declare_parameter("goal_topic", "/current_eval_goal")
        self.declare_parameter("run_command_topic", "/eval/run_command")
        self.declare_parameter("run_result_topic", "/eval/run_result")

        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "base_link")

        self.declare_parameter("csv_path", "/home/manas/isaacsim_teleop/src/teleop_keys/teleop_keys/thesis_eval_results.csv")
        self.declare_parameter("goal_tolerance", 0.5)
        self.declare_parameter("collision_threshold", 0.25)
        self.declare_parameter("collision_cooldown_sec", 1.0)
        self.declare_parameter("timeout_sec", 300.0)

        self.declare_parameter("method_name", "IL_multimodal")
        self.declare_parameter("environment_name", "unseen_env_01")

        self.scan_topic = self.get_parameter("scan_topic").value
        self.goal_topic = self.get_parameter("goal_topic").value
        self.run_command_topic = self.get_parameter("run_command_topic").value
        self.run_result_topic = self.get_parameter("run_result_topic").value

        self.global_frame = self.get_parameter("global_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value

        self.csv_path = self.get_parameter("csv_path").value
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.collision_threshold = float(self.get_parameter("collision_threshold").value)
        self.collision_cooldown_sec = float(self.get_parameter("collision_cooldown_sec").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)

        self.method_name = self.get_parameter("method_name").value
        self.environment_name = self.get_parameter("environment_name").value

        self.current_goal = None
        self.latest_min_lidar = None

        self.current_robot_x = None
        self.current_robot_y = None

        self.active = False
        self.run_id = -1
        self.start_time = None

        self.collision_flag = 0
        self.collision_count = 0
        self.last_collision_time = None
        self.min_lidar_observed = float("inf")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self.goal_callback, 10)
        self.create_subscription(String, self.run_command_topic, self.command_callback, 10)

        self.result_pub = self.create_publisher(String, self.run_result_topic, 10)

        self.set_sim_client = self.create_client(
            SetSimulationState,
            "/isaacsim/SetSimulationState"
        )

        self.get_logger().info("Waiting for Isaac Sim SetSimulationState service...")
        self.set_sim_client.wait_for_service()
        self.get_logger().info("Isaac Sim SetSimulationState service available.")

        self.timer = self.create_timer(0.5, self.evaluate)

        self.initialize_csv()

        self.get_logger().info("Evaluator Node ready.")
        self.get_logger().info(f"Using scan topic: {self.scan_topic}")
        self.get_logger().info(f"Using TF: {self.global_frame} -> {self.robot_frame}")
        self.get_logger().info(f"CSV path: {self.csv_path}")

    def initialize_csv(self):
        directory = os.path.dirname(self.csv_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_id",
                    "method",
                    "environment",
                    "goal_x",
                    "goal_y",
                    "success",
                    "timeout",
                    "collision_flag",
                    "collision_count",
                    "min_lidar_observed",
                    "time_to_goal_sec"
                ])

    def scan_callback(self, msg: LaserScan):
        valid_ranges = [r for r in msg.ranges if math.isfinite(r) and r > 0.0]

        if not valid_ranges:
            return

        self.latest_min_lidar = min(valid_ranges)

        if self.active:
            self.min_lidar_observed = min(
                self.min_lidar_observed,
                self.latest_min_lidar
            )

    def goal_callback(self, msg: PoseStamped):
        self.current_goal = msg.pose

        self.get_logger().info(
            f"Received goal: x={self.current_goal.position.x:.2f}, "
            f"y={self.current_goal.position.y:.2f}"
        )

    def command_callback(self, msg: String):
        tokens = msg.data.strip().split(",")

        if tokens[0] == "START":
            self.run_id = int(tokens[1])
            self.start_run()

        elif tokens[0] == "STOP":
            self.active = False
            self.get_logger().info("Received STOP command. Evaluation paused.")

    def start_run(self):
        if self.current_goal is None:
            self.get_logger().warn("Cannot start run: no current goal received.")
            return

        self.active = True
        self.start_time = self.get_clock().now()

        self.collision_flag = 0
        self.collision_count = 0
        self.last_collision_time = None
        self.min_lidar_observed = float("inf")

        self.get_logger().info(f"Started evaluation for run_id={self.run_id}")

    def get_elapsed_time(self):
        if self.start_time is None:
            return 0.0

        now = self.get_clock().now()
        return (now - self.start_time).nanoseconds / 1e9

    def get_robot_pose_from_tf(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time()
            )

            self.current_robot_x = tf.transform.translation.x
            self.current_robot_y = tf.transform.translation.y

            return self.current_robot_x, self.current_robot_y

        except TransformException as ex:
            self.get_logger().warn(
                f"TF lookup failed: {self.global_frame} -> {self.robot_frame}: {ex}"
            )
            return None, None

    def get_distance_to_goal(self):
        if self.current_goal is None:
            return float("inf")

        robot_x, robot_y = self.get_robot_pose_from_tf()

        if robot_x is None or robot_y is None:
            return float("inf")

        dx = robot_x - self.current_goal.position.x
        dy = robot_y - self.current_goal.position.y

        return math.sqrt(dx * dx + dy * dy)

    def update_collision_count(self):
        if self.latest_min_lidar is None:
            return

        if self.latest_min_lidar >= self.collision_threshold:
            return

        now = self.get_clock().now()

        if self.last_collision_time is None:
            self.collision_count += 1
            self.collision_flag = 1
            self.last_collision_time = now
            self.get_logger().warn(
                f"Collision detected | min_lidar={self.latest_min_lidar:.3f}"
            )
            return

        dt = (now - self.last_collision_time).nanoseconds / 1e9

        if dt >= self.collision_cooldown_sec:
            self.collision_count += 1
            self.collision_flag = 1
            self.last_collision_time = now
            self.get_logger().warn(
                f"Collision detected | min_lidar={self.latest_min_lidar:.3f}"
            )

    def evaluate(self):
        if not self.active:
            return

        self.update_collision_count()

        elapsed = self.get_elapsed_time()
        distance = self.get_distance_to_goal()

        if self.current_goal is None:
            self.get_logger().warn("No current goal received yet.")
            return

        if self.current_robot_x is None or self.current_robot_y is None:
            self.get_logger().warn("No robot pose from TF yet.")
            return

        lidar_text = "None"
        if self.latest_min_lidar is not None:
            lidar_text = f"{self.latest_min_lidar:.3f}"

        self.get_logger().info(
            f"[RUN {self.run_id}] elapsed={elapsed:.2f}s | "
            f"dist={distance:.2f} | "
            f"robot=({self.current_robot_x:.2f}, {self.current_robot_y:.2f}) | "
            f"goal=({self.current_goal.position.x:.2f}, {self.current_goal.position.y:.2f}) | "
            f"min_lidar={lidar_text} | "
            f"collisions={self.collision_count}"
        )

        if distance <= self.goal_tolerance:
            self.get_logger().info("SUCCESS condition met.")
            self.finish_run(success=1, timeout=0, time_to_goal=elapsed)
            return

        if elapsed >= self.timeout_sec:
            self.get_logger().warn("TIMEOUT condition met.")
            self.finish_run(success=0, timeout=1, time_to_goal=elapsed)
            return

    def stop_simulation(self):
        req = SetSimulationState.Request()
        req.state.state = SimulationState.STATE_STOPPED

        future = self.set_sim_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            self.get_logger().error("Stop simulation service timed out.")
            return False

        if future.result() is None:
            self.get_logger().error("Failed to stop simulation.")
            return False

        result = future.result().result
        self.get_logger().info(
            f"Stopped simulation | result={result.result}, msg={result.error_message}"
        )

        return True

    def finish_run(self, success: int, timeout: int, time_to_goal: float):
        self.active = False

        goal_x = self.current_goal.position.x
        goal_y = self.current_goal.position.y

        min_lidar = self.min_lidar_observed
        if min_lidar == float("inf"):
            min_lidar = -1.0

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.run_id,
                self.method_name,
                self.environment_name,
                goal_x,
                goal_y,
                success,
                timeout,
                self.collision_flag,
                self.collision_count,
                min_lidar,
                time_to_goal
            ])

        result_msg = String()
        result_msg.data = (
            f"DONE,{self.run_id},{success},{timeout},"
            f"{time_to_goal:.3f},{self.collision_count}"
        )
        
        for _ in range(5):
            self.result_pub.publish(result_msg)
            time.sleep(0.1)

        self.get_logger().info(
            f"Published DONE result: {result_msg.data}"
        )

        self.stop_simulation()

        self.get_logger().info(
            f"Run {self.run_id} done | success={success}, timeout={timeout}, "
            f"time={time_to_goal:.2f}, collisions={self.collision_count}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = EvaluatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
