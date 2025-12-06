#!/usr/bin/env python3
import json
import os
import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Image
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

from cv_bridge import CvBridge
import cv2


class JsonReplayNode(Node):
    def __init__(self):
        super().__init__('json_replay_node')

        # -------- Parameters --------
        # Path to JSON file
        self.declare_parameter('json_path', '/home/manas/isaacsim_teleop/src/teleop_keys/teleop_keys/manifest.json')
        # Directory that image paths are relative to (optional)
        self.declare_parameter('image_root', '/home/manas/isaacsim_teleop/src/teleop_keys/teleop_keys/')
        # Playback rate (Hz): how many entries per second
        self.declare_parameter('rate', 20.0)
        # Frames
        self.declare_parameter('lidar_frame', 'lidar')
        self.declare_parameter('camera_frame', 'camera')
        self.declare_parameter('world_frame', 'map')

        json_path = self.get_parameter('json_path').get_parameter_value().string_value
        self.image_root = self.get_parameter('image_root').get_parameter_value().string_value
        self.rate = self.get_parameter('rate').get_parameter_value().double_value
        self.lidar_frame = self.get_parameter('lidar_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value

        # -------- Load JSON --------
        self.get_logger().info(f'Loading JSON from {json_path}')
        with open(json_path, 'r') as f:
            self.entries = json.load(f)

        if not isinstance(self.entries, list):
            raise RuntimeError("Top-level JSON must be a list of entries")

        self.index = 0
        self.num_entries = len(self.entries)
        self.get_logger().info(f'Loaded {self.num_entries} entries.')

        # -------- Publishers --------
        self.scan_pub = self.create_publisher(LaserScan, 'scan', 10)
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.marker_pub = self.create_publisher(Marker, 'goal_marker', 10)

        # CV bridge for images
        self.bridge = CvBridge()

        # Timer for playback
        period = 1.0 / self.rate
        self.timer = self.create_timer(period, self.publish_next)

    def publish_next(self):
        if self.index >= self.num_entries:
            self.get_logger().info('Finished replaying all entries.')
            # Optional: stop the node
            rclpy.shutdown()
            return

        entry = self.entries[self.index]
        self.index += 1

        # Single timestamp for all three messages
        now = self.get_clock().now().to_msg()

        # ===================== LIDAR -> LaserScan =====================
        try:
            ranges = entry['lidar']  # list of floats
        except KeyError:
            self.get_logger().warn(f'Entry {self.index-1} missing "lidar" key, skipping.')
            return

        scan_msg = LaserScan()
        scan_msg.header.stamp = now
        scan_msg.header.frame_id = self.lidar_frame

        # You may want to adjust these based on your real sensor
        scan_msg.range_min = 0.05
        scan_msg.range_max = 100.0

        n = len(ranges)
        if n == 0:
            self.get_logger().warn(f'Entry {self.index-1} has empty lidar array, skipping scan publish.')
        else:
            # Example: full 360° scan
            scan_msg.angle_min = -math.pi
            scan_msg.angle_max = math.pi
            scan_msg.angle_increment = (scan_msg.angle_max - scan_msg.angle_min) / n
            scan_msg.time_increment = 0.0
            scan_msg.scan_time = 0.0
            scan_msg.ranges = ranges
            self.scan_pub.publish(scan_msg)

        # ===================== RGB -> Image =====================
        try:
            rgb_rel_path = entry['image_path']  # string
        except KeyError:
            self.get_logger().warn(f'Entry {self.index-1} missing "rgb_path" key, skipping image.')
            rgb_rel_path = None

        if rgb_rel_path is not None:
            img_path = os.path.join(self.image_root, rgb_rel_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn(f'Could not read image at {img_path}')
            else:
                img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                img_msg.header.stamp = now
                img_msg.header.frame_id = self.camera_frame
                self.image_pub.publish(img_msg)

        # ===================== Goal -> Marker =====================
        goal = entry.get('goal', None)
        if goal is not None:
            try:
                gx = float(goal[0])
                gy = float(goal[1])
            except (KeyError, ValueError, TypeError):
                self.get_logger().warn(f'Entry {self.index-1} has invalid "goal", skipping marker.')
            else:
                marker = Marker()
                marker.header.stamp = now
                marker.header.frame_id = self.world_frame

                marker.ns = "goal"
                marker.id = 0
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                marker.pose.position.x = gx
                marker.pose.position.y = gy
                marker.pose.position.z = 0.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3

                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

                # Keep marker alive until changed/removed
                marker.lifetime = Duration(sec=0, nanosec=0)
                self.marker_pub.publish(marker)

        self.get_logger().info(f'Published entry {self.index}/{self.num_entries}')


def main(args=None):
    rclpy.init(args=args)
    node = JsonReplayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

