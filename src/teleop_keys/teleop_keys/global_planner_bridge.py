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
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from visualization_msgs.msg import Marker

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

        self._eval_result_sub = self.create_subscription(
            String, '/eval/run_result', self._eval_result_cb, 10
        )

        self._path_pub = self.create_publisher(Path, '/global_path', 10)
        self._local_goal_pub = self.create_publisher(Pose2D, '/local_goal', 10)
        self._stamp_pub = self.create_publisher(Time, '/local_goal_stamp', 10)
        self._lookahead_marker_pub = self.create_publisher(Marker, '/lookahead_marker', 10)

        self._clock_sub = self.create_subscription(
            Clock, '/clock', self._clock_callback, 10
        )
        self._last_clock: Time | None = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('lookahead_index', 70)
        self.declare_parameter('freeze_first_path', True)

        self.global_frame = self.get_parameter('global_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.lookahead_idx = self.get_parameter('lookahead_index').value
        self.freeze_first_path = self.get_parameter('freeze_first_path').value

        self.global_path: Path | None = None
        self._path_frozen = False
        self._last_best_idx = 0

        self._timer = self.create_timer(0.1, self._timer_cb)

        self.get_logger().info(
            f"GlobalPlannerBridge ready. freeze_first_path={self.freeze_first_path}"
        )

    def reset_path_state(self):
        self.global_path = None
        self._path_frozen = False
        self._last_best_idx = 0
        self.get_logger().info("Reset path state after eval DONE.")

    def _eval_result_cb(self, msg: String):
        tokens = msg.data.split(",")

        if not tokens:
            return

        if tokens[0] == "DONE":
            self.reset_path_state()

    def _clock_callback(self, msg: Clock):
        self._last_clock = msg.clock

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
            self.get_logger().warn("Planner returned empty path; keeping existing path if any.")
            return

        if self.freeze_first_path and self._path_frozen:
            self.get_logger().info("Path already frozen; ignoring replanned path.")
            return

        self.global_path = path
        self._path_pub.publish(path)
        self._last_best_idx = 0

        if self.freeze_first_path:
            self._path_frozen = True

        self.get_logger().info(
            f"Stored global path with {len(path.poses)} poses. Frozen={self._path_frozen}"
        )

    def _timer_cb(self):
        if self.global_path is None or not self.global_path.poses:
            return

        if self._last_clock is None:
            self.get_logger().debug("No /clock received yet; skipping /local_goal publish.")
            return

        best_idx = None
        best_dist = float('inf')

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
            dist = math.sqrt(x_r * x_r + y_r * y_r)

            if x_r > 0.0 and dist < best_dist:
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

        local_goal = Pose2D()
        local_goal.x = x
        local_goal.y = y
        local_goal.theta = theta

        stamp = Time()
        stamp.sec = self._last_clock.sec
        stamp.nanosec = self._last_clock.nanosec

        self._local_goal_pub.publish(local_goal)
        self._stamp_pub.publish(stamp)

        marker = Marker()
        marker.header.frame_id = self.robot_frame
        marker.header.stamp = RclTime().to_msg()
        marker.ns = "lookahead"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation = target_robot.pose.orientation

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self._lookahead_marker_pub.publish(marker)

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
