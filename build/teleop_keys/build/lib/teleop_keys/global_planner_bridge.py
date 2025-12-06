#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time

from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped, Pose2D
from nav_msgs.msg import Path

import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs


class GlobalPlannerBridge(Node):
    """
    - Subscribes to high-level goal in map frame: /nav_goal (PoseStamped)
    - Calls Nav2 global planner (ComputePathToPose) to get nav_msgs/Path
    - Keeps the full path
    - On each cycle:
        * Finds the closest path pose ahead of the robot
        * Picks a lookahead pose some steps further along the path
        * Transforms that pose into robot frame (base_link)
        * Publishes a relative 2D goal on /local_goal (Pose2D), updated continuously
    """
    def __init__(self):
        super().__init__('global_planner_bridge')

        # Action client for Nav2 global planner
        self._planner_client = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose'
        )

        # Incoming high-level goals in map frame
        self._goal_sub = self.create_subscription(
            PoseStamped,
            '/nav_goal',  # you can remap RViz NavGoal to this
            self.goal_callback,
            10
        )

        # For debugging: publish full global path
        self._path_pub = self.create_publisher(Path, '/global_path', 10)

        # For local planner / IL policy: publish local goal (x,y,theta) in base_link
        self._local_goal_pub = self.create_publisher(Pose2D, '/local_goal', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Parameters
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('lookahead_index', 10)

        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        self.lookahead_idx = self.get_parameter('lookahead_index').get_parameter_value().integer_value

        # Stored full global path (for dynamic lookahead)
        self.global_path: Path = None

        # Timer to continuously update /local_goal from latest TF
        self._timer = self.create_timer(0.1, self._timer_cb)  # 10 Hz

        self.get_logger().info(
            "GlobalPlannerBridge (dynamic lookahead) ready. Publish PoseStamped to /nav_goal."
        )

    # ----------------------------------------------------------------------
    #  Goal + planner callbacks
    # ----------------------------------------------------------------------
    def goal_callback(self, goal_msg: PoseStamped):
        self.get_logger().info("Received /nav_goal, calling Nav2 ComputePathToPose...")

        if goal_msg.header.frame_id != self.global_frame:
            self.get_logger().warn(
                f"Goal frame_id '{goal_msg.header.frame_id}' != global_frame '{self.global_frame}'."
            )

        goal = ComputePathToPose.Goal()
        goal.goal = goal_msg
        # If start is not set, Nav2 will use current robot pose.

        if not self._planner_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Planner action server not available.")
            return

        send_future = self._planner_client.send_goal_async(
            goal,
            feedback_callback=self._planner_feedback_cb
        )
        send_future.add_done_callback(self._planner_goal_response_cb)

    def _planner_feedback_cb(self, feedback_msg):
        # Optional: inspect partial paths here if you want
        pass

    def _planner_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Planner goal rejected.")
            return

        self.get_logger().info("Planner goal accepted. Waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._planner_result_cb)

    def _planner_result_cb(self, future):
        result = future.result().result
        path: Path = result.path

        if not path.poses:
            self.get_logger().error("Planner returned empty path.")
            self.global_path = None
            return

        # Store full path for dynamic lookahead
        self.global_path = path

        # Publish for visualization/debug
        self._path_pub.publish(path)
        self.get_logger().info(f"Received path with {len(path.poses)} poses.")
        self.get_logger().info(
            f"Dynamic lookahead enabled with lookahead_index={self.lookahead_idx}."
        )

    # ----------------------------------------------------------------------
    #  Timer: dynamic lookahead + TF + local_goal publish
    # ----------------------------------------------------------------------
    def _timer_cb(self):
        """
        Periodically:
        - Transform global path poses into robot frame
        - Find the closest pose ahead of the robot
        - Pick a lookahead pose (index + lookahead_idx, clamped to end)
        - Publish /local_goal (Pose2D in robot_frame)
        """
        if self.global_path is None or not self.global_path.poses:
            return

        # Find closest pose AHEAD of the robot in robot frame
        best_idx = None
        best_dist = float('inf')

        for i, pose_stamped in enumerate(self.global_path.poses):
            # Build a PoseStamped in global frame with "latest" time
            pose_in_global = PoseStamped()
            pose_in_global.header.frame_id = (
                pose_stamped.header.frame_id or self.global_frame
            )
            # time=0 in TF semantics => "latest available transform"
            pose_in_global.header.stamp = Time().to_msg()
            pose_in_global.pose = pose_stamped.pose

            # Transform pose into robot frame
            try:
                pose_in_robot = self.tf_buffer.transform(
                    pose_in_global,
                    self.robot_frame
                )
            except TransformException as ex:
                self.get_logger().debug(f"Transform of pose {i} failed: {ex}")
                continue

            x_r = pose_in_robot.pose.position.x
            y_r = pose_in_robot.pose.position.y
            dist = math.sqrt(x_r * x_r + y_r * y_r)

            # Prefer points in front of the robot
            if x_r > 0.0 and dist < best_dist:
                best_dist = dist
                best_idx = i

        # Fallback: if no pose is strictly ahead, use the closest overall
        if best_idx is None:
            self.get_logger().debug("No pose strictly ahead; using closest over entire path.")
            best_idx = 0
            best_dist = float('inf')

            for i, pose_stamped in enumerate(self.global_path.poses):
                pose_in_global = PoseStamped()
                pose_in_global.header.frame_id = (
                    pose_stamped.header.frame_id or self.global_frame
                )
                pose_in_global.header.stamp = Time().to_msg()
                pose_in_global.pose = pose_stamped.pose

                try:
                    pose_in_robot = self.tf_buffer.transform(
                        pose_in_global,
                        self.robot_frame
                    )
                except TransformException:
                    continue

                x_r = pose_in_robot.pose.position.x
                y_r = pose_in_robot.pose.position.y
                dist = math.sqrt(x_r * x_r + y_r * y_r)

                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

        # Apply lookahead in index space, with clamping near the goal
        target_idx = best_idx + self.lookahead_idx
        if target_idx >= len(self.global_path.poses):
            target_idx = len(self.global_path.poses) - 1  # clamp to final goal

        target_pose_stamped = self.global_path.poses[target_idx]

        # Transform that target pose into robot frame
        target_in_global = PoseStamped()
        target_in_global.header.frame_id = (
            target_pose_stamped.header.frame_id or self.global_frame
        )
        target_in_global.header.stamp = Time().to_msg()
        target_in_global.pose = target_pose_stamped.pose

        try:
            target_pose_robot = self.tf_buffer.transform(
                target_in_global,
                self.robot_frame
            )
        except TransformException as ex:
            self.get_logger().warn(f"Transform for target_idx={target_idx} failed: {ex}")
            return

        x = target_pose_robot.pose.position.x
        y = target_pose_robot.pose.position.y

        # NEW: compute yaw (theta) from the orientation of the target pose in robot frame
        ori = target_pose_robot.pose.orientation
        theta = self.quat_to_yaw(ori.x, ori.y, ori.z, ori.w)

        local_goal = Pose2D()
        local_goal.x = x
        local_goal.y = y
        local_goal.theta = theta   # <<--- changed from 0.0 to actual yaw

        self._local_goal_pub.publish(local_goal)

        self.get_logger().debug(
            f"Dynamic /local_goal (idx={target_idx}, ahead of idx={best_idx}) in {self.robot_frame}: "
            f"x={x:.2f}, y={y:.2f}, theta={theta:.2f}"
        )

    # ----------------------------------------------------------------------
    @staticmethod
    def quat_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

