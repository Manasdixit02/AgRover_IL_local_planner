#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time as RclTime

from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped, Pose2D
from nav_msgs.msg import Path
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock   # NEW: to subscribe to /clock

import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs


class GlobalPlannerBridge(Node):
    def __init__(self):
        super().__init__('global_planner_bridge')

        self._planner_client = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose'
        )

        self._goal_sub = self.create_subscription(
            PoseStamped, '/nav_goal', self.goal_callback, 10
        )

        self._path_pub = self.create_publisher(Path, '/global_path', 10)

        # Pose2D local goal
        self._local_goal_pub = self.create_publisher(Pose2D, '/local_goal', 10)

        # Timestamp topic for IL alignment
        self._stamp_pub = self.create_publisher(Time, '/local_goal_stamp', 10)

        # NEW: subscribe to /clock explicitly
        self._clock_sub = self.create_subscription(
            Clock, '/clock', self._clock_callback, 10
        )
        self._last_clock: Time | None = None  # will store builtin_interfaces/Time

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('lookahead_index', 10)

        self.global_frame = self.get_parameter('global_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.lookahead_idx = self.get_parameter('lookahead_index').value

        self.global_path = None
        self._last_best_idx = 0

        self._timer = self.create_timer(0.1, self._timer_cb)

        self.get_logger().info("GlobalPlannerBridge (dynamic lookahead) ready.")

    # ---------------- /clock callback ----------------
    def _clock_callback(self, msg: Clock):
        # msg.clock is builtin_interfaces/Time
        self._last_clock = msg.clock

    # --- planner callbacks unchanged ---
    def goal_callback(self, goal_msg: PoseStamped):
        self.get_logger().info("Received /nav_goal, calling Nav2 ComputePathToPose...")
        goal = ComputePathToPose.Goal()
        goal.goal = goal_msg

        if not self._planner_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Planner action server not available.")
            return

        send_future = self._planner_client.send_goal_async(
            goal, feedback_callback=self._planner_feedback_cb
        )
        send_future.add_done_callback(self._planner_goal_response_cb)

    def _planner_feedback_cb(self, feedback_msg):
        pass

    def _planner_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Planner goal rejected.")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._planner_result_cb)

    def _planner_result_cb(self, future):
        result = future.result().result
        path: Path = result.path

        if not path.poses:
            self.global_path = None
            return

        self.global_path = path
        self._path_pub.publish(path)
        self.get_logger().info(f"Received path with {len(path.poses)} poses.")

    # ---------------- Timer Callback ----------------
    def _timer_cb(self):
        if self.global_path is None or not self.global_path.poses:
            return

        # If we haven't received /clock yet, don't publish stamp/goal
        if self._last_clock is None:
            self.get_logger().debug("No /clock received yet; skipping /local_goal publish.")
            return

        best_idx = None
        best_dist = float('inf')

        # find closest-AHEAD waypoint in robot frame
        for i, pose_stamped in enumerate(self.global_path.poses):
            pose_in_global = PoseStamped()
            pose_in_global.header.frame_id = pose_stamped.header.frame_id or self.global_frame
            pose_in_global.header.stamp = RclTime().to_msg()
            pose_in_global.pose = pose_stamped.pose

            try:
                pose_in_robot = self.tf_buffer.transform(pose_in_global, self.robot_frame)
            except TransformException:
                continue

            x_r = pose_in_robot.pose.position.x
            y_r = pose_in_robot.pose.position.y

            dist = math.sqrt(x_r*x_r + y_r*y_r)

            if x_r > 0 and dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None:
            best_idx = self._last_best_idx
        else:
            self._last_best_idx = best_idx

        target_idx = min(best_idx + self.lookahead_idx, len(self.global_path.poses) - 1)
        target_pose = self.global_path.poses[target_idx]

        target_global = PoseStamped()
        target_global.header.frame_id = target_pose.header.frame_id or self.global_frame
        target_global.header.stamp = RclTime().to_msg()
        target_global.pose = target_pose.pose

        try:
            target_robot = self.tf_buffer.transform(target_global, self.robot_frame)
        except TransformException:
            return

        x = target_robot.pose.position.x
        y = target_robot.pose.position.y

        ori = target_robot.pose.orientation
        theta = self.quat_to_yaw(ori.x, ori.y, ori.z, ori.w)

        # ---- Pose2D message ----
        local_goal = Pose2D()
        local_goal.x = x
        local_goal.y = y
        local_goal.theta = theta

        # ---- Timestamp message using explicit /clock ----
        stamp = Time()
        stamp.sec = self._last_clock.sec
        stamp.nanosec = self._last_clock.nanosec

        # Publish both
        self._local_goal_pub.publish(local_goal)
        self._stamp_pub.publish(stamp)

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w*z + x*y)
        cosy_cosp = 1.0 - 2.0*(y*y + z*z)
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

