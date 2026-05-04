#!/usr/bin/env python3

import math
import time
import yaml
import threading
import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from std_msgs.msg import String

from simulation_interfaces.srv import SetSimulationState, SetEntityState
from simulation_interfaces.msg import SimulationState, EntityState

from nav2_msgs.action import NavigateToPose
from lifecycle_msgs.srv import GetState


def yaw_to_quaternion(yaw: float):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return 0.0, 0.0, qz, qw


class ExperimentManagerNode(Node):
    def __init__(self):
        super().__init__("experiment_manager_node")

        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("goals_yaml", "")
        self.declare_parameter("robot_entity", "/rover")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("world_frame", "World")
        self.declare_parameter("settle_time_sec", 5.0)
        self.declare_parameter("bt_wait_timeout_sec", 15.0)

        self.declare_parameter("start_x", -2.0008)
        self.declare_parameter("start_y", -0.5197)
        self.declare_parameter("start_z", 0.2205)
        self.declare_parameter("start_yaw", 0.0)

        self.declare_parameter("initialpose_topic", "/initialpose")
        self.declare_parameter("current_eval_goal_topic", "/current_eval_goal")
        self.declare_parameter("run_command_topic", "/eval/run_command")
        self.declare_parameter("run_result_topic", "/eval/run_result")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_nav2")
        self.declare_parameter("navigate_to_pose_action", "/navigate_to_pose")
        self.declare_parameter("bt_navigator_state_service", "/bt_navigator/get_state")

        self.goals_yaml = self.get_parameter("goals_yaml").value
        self.robot_entity = self.get_parameter("robot_entity").value
        self.map_frame = self.get_parameter("map_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.settle_time_sec = float(self.get_parameter("settle_time_sec").value)
        self.bt_wait_timeout_sec = float(self.get_parameter("bt_wait_timeout_sec").value)

        self.start_pose = {
            "x": float(self.get_parameter("start_x").value),
            "y": float(self.get_parameter("start_y").value),
            "z": float(self.get_parameter("start_z").value),
            "yaw": float(self.get_parameter("start_yaw").value),
        }

        self.initialpose_topic = self.get_parameter("initialpose_topic").value
        self.current_eval_goal_topic = self.get_parameter("current_eval_goal_topic").value
        self.run_command_topic = self.get_parameter("run_command_topic").value
        self.run_result_topic = self.get_parameter("run_result_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.navigate_to_pose_action = self.get_parameter("navigate_to_pose_action").value
        self.bt_navigator_state_service = self.get_parameter("bt_navigator_state_service").value

        if not self.goals_yaml:
            raise RuntimeError("Parameter goals_yaml is required.")

        self.goals = self.load_goals(self.goals_yaml)

        self.current_eval_goal_pub = self.create_publisher(
            PoseStamped, self.current_eval_goal_topic, 10
        )
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, self.initialpose_topic, 10
        )
        self.run_command_pub = self.create_publisher(
            String, self.run_command_topic, 10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist, self.cmd_vel_topic, 10
        )

        self.result_sub = self.create_subscription(
            String,
            self.run_result_topic,
            self.result_callback,
            10,
            callback_group=self.cb_group
        )

        self.set_sim_client = self.create_client(
            SetSimulationState,
            "/isaacsim/SetSimulationState",
            callback_group=self.cb_group
        )

        self.set_entity_client = self.create_client(
            SetEntityState,
            "/isaacsim/SetEntityState",
            callback_group=self.cb_group
        )

        self.bt_state_client = self.create_client(
            GetState,
            self.bt_navigator_state_service,
            callback_group=self.cb_group
        )

        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            self.navigate_to_pose_action,
            callback_group=self.cb_group
        )

        self.get_logger().info("Waiting for Isaac Sim services...")
        self.set_sim_client.wait_for_service()
        self.set_entity_client.wait_for_service()
        self.get_logger().info("Isaac Sim services available.")

        self.get_logger().info("Waiting for bt_navigator state service...")
        self.bt_state_client.wait_for_service()
        self.get_logger().info("bt_navigator state service available.")

        self.get_logger().info("Waiting for NavigateToPose action server...")
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info("NavigateToPose action server available.")

        self.current_run_idx = -1
        self.waiting_for_result = False
        self.started = False
        self.next_run_pending = False

        self.timer = self.create_timer(
            2.0,
            self.start_next_run_once,
            callback_group=self.cb_group
        )

        self.next_run_timer = self.create_timer(
            1.0,
            self.check_start_next_run,
            callback_group=self.cb_group
        )

        self.get_logger().info(f"Loaded {len(self.goals)} goals.")
        self.get_logger().info(f"Using robot entity: {self.robot_entity}")
        self.get_logger().info(f"Using cmd_vel topic: {self.cmd_vel_topic}")
        self.get_logger().info(f"Using fixed start pose: {self.start_pose}")
        self.get_logger().info(f"Using Nav2 action: {self.navigate_to_pose_action}")

    def wait_for_future(self, future, timeout_sec: float):
        event = threading.Event()

        def done_callback(_):
            event.set()

        future.add_done_callback(done_callback)

        finished = event.wait(timeout=timeout_sec)

        if not finished:
            return False

        return future.done()

    def load_goals(self, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        exp = data["experiment"]

        if "start_pose" in exp:
            sp = exp["start_pose"]
            self.start_pose["x"] = float(sp.get("x", self.start_pose["x"]))
            self.start_pose["y"] = float(sp.get("y", self.start_pose["y"]))
            self.start_pose["z"] = float(sp.get("z", self.start_pose["z"]))
            self.start_pose["yaw"] = float(sp.get("yaw", self.start_pose["yaw"]))

        return exp["goals"]

    def make_pose_stamped(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        return msg

    def make_initialpose(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.0685

        return msg

    def call_set_sim_state(self, state_value):
        req = SetSimulationState.Request()
        req.state.state = state_value

        future = self.set_sim_client.call_async(req)

        if not self.wait_for_future(future, 5.0):
            self.get_logger().error("SetSimulationState timed out.")
            return False

        if future.result() is None:
            self.get_logger().error("SetSimulationState failed.")
            return False

        result = future.result().result
        self.get_logger().info(
            f"SetSimulationState({state_value}) result={result.result}, msg={result.error_message}"
        )

        return result.result in [result.RESULT_OK, 101]

    def play_simulation(self):
        return self.call_set_sim_state(SimulationState.STATE_PLAYING)

    def set_robot_pose(self, x, y, z, yaw):
        req = SetEntityState.Request()
        req.entity = self.robot_entity

        state = EntityState()
        state.header.frame_id = self.world_frame

        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)

        qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw

        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        req.state = state

        future = self.set_entity_client.call_async(req)

        if not self.wait_for_future(future, 5.0):
            self.get_logger().error("SetEntityState timed out.")
            return False

        if future.result() is None:
            self.get_logger().error("SetEntityState failed.")
            return False

        result = future.result().result
        self.get_logger().info(
            f"SetEntityState result={result.result}, msg={result.error_message}"
        )

        return result.result == result.RESULT_OK

    def is_bt_navigator_active(self):
        req = GetState.Request()
        future = self.bt_state_client.call_async(req)

        if not self.wait_for_future(future, 2.0):
            return False

        if future.result() is None:
            return False

        state = future.result().current_state
        self.get_logger().info(f"bt_navigator state: {state.label} [{state.id}]")
        return state.label == "active" or state.id == 3

    def wait_for_bt_navigator_active(self):
        self.get_logger().info("Waiting for bt_navigator to become active...")

        start_time = time.time()

        while time.time() - start_time < self.bt_wait_timeout_sec:
            if self.is_bt_navigator_active():
                self.get_logger().info("bt_navigator is active.")
                return True

            time.sleep(0.5)

        self.get_logger().error("Timed out waiting for bt_navigator to become active.")
        return False

    def send_nav2_goal(self, pose_msg: PoseStamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_msg

        self.get_logger().info(
            f"Sending Nav2 goal: x={pose_msg.pose.position.x:.3f}, y={pose_msg.pose.position.y:.3f}"
        )

        future = self.nav_to_pose_client.send_goal_async(goal_msg)

        if not self.wait_for_future(future, 5.0):
            self.get_logger().error("NavigateToPose goal request timed out.")
            return False

        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error("NavigateToPose returned no goal handle.")
            return False

        if not goal_handle.accepted:
            self.get_logger().error("NavigateToPose goal was rejected.")
            return False

        self.get_logger().info("NavigateToPose goal accepted.")
        return True

    def publish_zero_cmd(self):
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def start_next_run_once(self):
        if self.started:
            return

        self.started = True
        self.start_next_run()

    def check_start_next_run(self):
        if not self.next_run_pending:
            return

        if self.waiting_for_result:
            return

        self.next_run_pending = False
        self.get_logger().info("Starting next run from timer callback...")
        self.start_next_run()

    def start_next_run(self):
        self.current_run_idx += 1

        if self.current_run_idx >= len(self.goals):
            self.get_logger().info("All experiment runs completed.")
            return

        run_id = self.current_run_idx
        goal = self.goals[run_id]

        sx = self.start_pose["x"]
        sy = self.start_pose["y"]
        sz = self.start_pose["z"]
        syaw = self.start_pose["yaw"]

        gx = goal["x"]
        gy = goal["y"]
        gyaw = goal.get("yaw", 0.0)

        self.get_logger().info(f"Preparing run {run_id}: goal=({gx}, {gy})")

        self.publish_zero_cmd()
        time.sleep(0.2)

        self.get_logger().info("Setting robot pose...")
        pose_set = self.set_robot_pose(sx, sy, sz, syaw)

        if not pose_set:
            self.get_logger().error("Failed to set robot pose. Aborting this run.")
            return

        self.get_logger().info("Robot pose set.")

        initialpose_msg = self.make_initialpose(sx, sy, syaw)
        goal_msg = self.make_pose_stamped(gx, gy, gyaw)

        self.get_logger().info("Publishing initial pose and evaluator goal...")
        for _ in range(5):
            self.initialpose_pub.publish(initialpose_msg)
            self.current_eval_goal_pub.publish(goal_msg)
            time.sleep(0.1)

        self.get_logger().info("Waiting for Nav2/AMCL to process initial pose...")
        time.sleep(self.settle_time_sec)

        self.get_logger().info("Starting simulation...")
        sim_started = self.play_simulation()

        if not sim_started:
            self.get_logger().error("Failed to start simulation. Aborting this run.")
            return

        time.sleep(0.5)

        if not self.wait_for_bt_navigator_active():
            self.get_logger().error("bt_navigator is not active. Aborting this run.")
            return

        self.current_eval_goal_pub.publish(goal_msg)

        nav_goal_sent = self.send_nav2_goal(goal_msg)

        if not nav_goal_sent:
            self.get_logger().error("Failed to send Nav2 goal. Aborting this run.")
            return

        self.waiting_for_result = True
        self.get_logger().info(f"waiting_for_result set TRUE for run {run_id}")

        cmd = String()
        cmd.data = f"START,{run_id}"
        self.run_command_pub.publish(cmd)

        self.get_logger().info(f"Run {run_id} started.")

    def result_callback(self, msg: String):
        self.get_logger().info(f"Received /eval/run_result message: {msg.data}")

        if not self.waiting_for_result:
            self.get_logger().warn(
                "Received result but waiting_for_result is False. Ignoring."
            )
            return

        tokens = msg.data.split(",")

        if tokens[0] != "DONE":
            self.get_logger().warn(f"Ignoring non-DONE result message: {msg.data}")
            return

        if len(tokens) < 2:
            self.get_logger().warn(
                f"Ignoring malformed DONE message. Expected DONE,<run_id>, got: {msg.data}"
            )
            return

        try:
            done_run_id = int(tokens[1])
        except ValueError:
            self.get_logger().warn(
                f"Ignoring DONE message with invalid run id: {msg.data}"
            )
            return

        if done_run_id != self.current_run_idx:
            self.get_logger().warn(
                f"Ignoring result for run {done_run_id}; current run is {self.current_run_idx}"
            )
            return

        self.get_logger().info(f"Received result for run {done_run_id}: {msg.data}")

        self.waiting_for_result = False
        self.publish_zero_cmd()

        self.next_run_pending = True
        self.get_logger().info("Next run marked as pending.")


def main(args=None):
    rclpy.init(args=args)

    node = ExperimentManagerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
