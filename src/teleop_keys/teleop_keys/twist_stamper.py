#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time

class CmdVelStamper(Node):
    def __init__(self):
        super().__init__('cmdvel_stamper')

        # store last clock message
        self._last_clock = None

        # subscribe to sim clock and cmd_vel
        self.create_subscription(Clock, '/clock', self._clock_cb, 10)
        self.create_subscription(Int32MultiArray, '/spyder/cmd_vel', self._cmd_cb, 10)

        # publish a stamped version (array + stamp as one message)
        # or simplest: just publish timestamp alongside the same array
        self.pub_data = self.create_publisher(Int32MultiArray, '/spyder/cmd_vel_synced', 10)
        self.pub_time = self.create_publisher(Time, '/spyder/cmd_vel_stamp', 10)

        self.get_logger().info("Syncing /spyder/cmd_vel with /clock")

    def _clock_cb(self, msg: Clock):
        self._last_clock = msg.clock

    def _cmd_cb(self, msg: Int32MultiArray):
        if self._last_clock is None:
            return  # no sim time yet
        # Re-publish the same data
        self.pub_data.publish(msg)
        # Publish its current sim-time stamp on a parallel topic
        self.pub_time.publish(self._last_clock)

def main():
    rclpy.init()
    node = CmdVelStamper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

