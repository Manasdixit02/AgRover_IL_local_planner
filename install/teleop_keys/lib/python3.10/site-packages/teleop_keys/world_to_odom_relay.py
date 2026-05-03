#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class WorldToOdomRelay(Node):
    def __init__(self):
        super().__init__('world_to_odom_relay')

        # CHANGE THESE IF YOUR FRAME NAMES DIFFER
        self.source_parent = 'World'     # Isaac parent frame
        self.source_child  = 'base_link' # Isaac child frame
        self.target_parent = 'odom'      # what Nav2 expects
        self.target_child  = 'base_link' # same robot base frame

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.02, self.tick)  # 50 Hz

    def tick(self):
        try:
            tf = self.buffer.lookup_transform(
                self.source_parent,
                self.source_child,
                rclpy.time.Time()
            )
        except Exception:
            return

        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.target_parent
        out.child_frame_id = self.target_child
        out.transform = tf.transform
        self.broadcaster.sendTransform(out)

def main():
    rclpy.init()
    node = WorldToOdomRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

