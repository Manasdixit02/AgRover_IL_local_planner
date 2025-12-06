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
    - Picks a lookahead pose from the path
    - Transforms that pose into robot frame (base_link)
    - Publishes a relative 2D goal on /local_goal (Pose2D), updated continuously
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
            '/nav_goal',     # you can remap RViz NavGoal to this
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
        self.robot_frame  = self.get_parameter('robot_frame').get_parameter_value().string_value
        self.lookahead_idx = self.get_parameter('lookahead_index').get_parameter_value().integer_value

        # Stored target in global frame (for continuous updates)
        self.global_target_pose = None

        # Timer to continuously update /local_goal from latest TF
        self._timer = self.create_timer(0.1, self._timer_cb)  # 10 Hz

        self.get_logger().info("GlobalPlannerBridge ready. Publish PoseStamped to /nav_goal.")

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
        # You can inspect partial paths here if you want
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
            return

        self._path_pub.publish(path)
        self.get_logger().info(f"Received path with {len(path.poses)} poses.")

        # Choose a lookahead point and store it in global frame
        idx = min(self.lookahead_idx, len(path.poses) - 1)
        self.global_target_pose = path.poses[idx]

        # Ensure frame is set
        if not self.global_target_pose.header.frame_id:
            self.global_target_pose.header.frame_id = self.global_frame

        self.get_logger().info(
            f"Stored global_target_pose at index {idx} in frame '{self.global_target_pose.header.frame_id}'."
        )

    def _timer_cb(self):
        """
        Periodically transform stored global_target_pose into robot frame and publish /local_goal.
        This makes /local_goal update as the robot moves.
        """
        if self.global_target_pose is None:
            return

        # Pose in global frame; use Time()=0 to request latest transform
        pose_in_global = PoseStamped()
        pose_in_global.header.frame_id = self.global_target_pose.header.frame_id
        pose_in_global.header.stamp = Time().to_msg()  # time=0 => latest TF
        pose_in_global.pose = self.global_target_pose.pose

        try:
            local_pose: PoseStamped = self.tf_buffer.transform(
                pose_in_global,
                self.robot_frame  # target frame, e.g. "base_link"
            )
        except TransformException as ex:
            # Can happen transiently during startup / TF gaps
            self.get_logger().warn(f"TF transform failed in timer: {ex}")
            return

        # Convert to Pose2D (x,y) in base_link frame
        x = local_pose.pose.position.x
        y = local_pose.pose.position.y

        local_goal = Pose2D()
        local_goal.x = x
        local_goal.y = y
        local_goal.theta = 0.0  # IL planner ignores orientation

        self._local_goal_pub.publish(local_goal)
        self.get_logger().debug(
            f"Published /local_goal in {self.robot_frame}: x={x:.2f}, y={y:.2f}, theta=0.0"
        )

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

