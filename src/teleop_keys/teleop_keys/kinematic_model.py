# !/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64 , Int32MultiArray
from math import atan, degrees, sqrt
# from rover_controller.pyserial_arduino import RawValues # Arduino sender function
# from rover_controller.PID_Controller import PIDController  # PID controller function
# import time


# Constants
WHEELBASE = 0.64  # Wheelbase in meters
TRACK_WIDTH = 0.495  # Track width in meters
d1 = 0.065  # Wheel radius (m)
d2 = 0.32 #0.25    # Half chassis length (m)
d3 = 0.24    # Half chassis width (m)
R = 0.55     # Instantaneous center of rotation (m)
MAX_VEL = 0.314  # Max motor velocity at 255 PWM (m/s)
LINEAR_LIMIT = 1.0
ANGULAR_LIMIT = 1.0
MIN_LIN_THRESH = 0.05
DEADBAND_PWM = 10

class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__('cmd_vel_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.steering_angle = self.create_publisher(Float64, 'steerAngle', 10)
        self.spyder_pub = self.create_publisher(Int32MultiArray, '/spyder/cmd_vel', 10)

        self.get_logger().info('Subscribed to /cmd_vel and publishing /spyder/cmd_vel -- /steerAngle')        

        # Parameters for d1, d2, d3 (wheel radius, half chassis length, half chassis width)
        self.declare_parameter("hardware_distances", [0.065, 0.25, 0.24])  # d1, d2, d3  0.0625, 0.27, 0.17
        self.declare_parameter("linear_limit", 1.0)
        self.declare_parameter("angular_limit", 1.0)
        self.declare_parameter("min_lin_threshold", 0.05)
        self.declare_parameter("deadband_pwm", 10)
        self.declare_parameter("max_motor_velocity", 0.314)  # m/s at 255 PWM

        # Get parameter values for d1, d2, d3
        d = self.get_parameter("hardware_distances").get_parameter_value().double_array_value
        self.d1, self.d2, self.d3 = d
        self.linear_limit = self.get_parameter("linear_limit").get_parameter_value().double_value
        self.angular_limit = self.get_parameter("angular_limit").get_parameter_value().double_value
        self.min_lin_threshold = self.get_parameter("min_lin_threshold").get_parameter_value().double_value
        self.deadband_pwm = self.get_parameter("deadband_pwm").get_parameter_value().integer_value
        self.max_motor_velocity = self.get_parameter("max_motor_velocity").get_parameter_value().double_value

        # Rover geometry (fixed values)
        self.WHEELBASE = 0.64 #0.5 #0.3285
        self.TRACK_WIDTH = 0.495 #0.48 #0.34

        # Time parameters
        # prevT = 0
        # self.currT = time.time()  # time() returns time in seconds as float
        # self.deltaT = self.currT - prevT
        # prevT = self.currT

        # Encoder data 
        # self.setpoint_pwm = []
        # self.encoder = EncoderSubscribe(self)
        # self.right_pid_setpoint = 0
        # self.right_PWM = 0
        
        # self.right_pid = PIDController(1, 0, 0)

        # self.arduino = RawValues()
        


    def cmd_vel_callback(self, msg):
    

        lin_vel = msg.linear.x
        ang_vel = msg.angular.z


        if lin_vel == 0 and ang_vel == 0:
            print("Stop")
        elif lin_vel == 0:
            print("Spin Left" if ang_vel > 0 else "Spin Right")
        elif ang_vel == 0:
            print("Forward" if lin_vel > 0 else "Backward")
        else:
            print(f"{'Forward' if lin_vel > 0 else 'Backward'} {'Left' if ang_vel > 0 else 'Right'} Turn")

        angles = self.compute_steering_angles(lin_vel, ang_vel)
        pwms = self.compute_velocities(lin_vel, ang_vel)



        # self.setpoint_right_pwm = pwms[4]
        # self.right_pid_setpoint = pwms[4]
        

        # ///////////////// ToDO: PID   ////////////////
        
        # self.right_PWM = self.right_pid.compute(self.right_pid_setpoint, self.encoder.actual_pwm_right)
        # print(f"setpoint={self.right_pid_setpoint}, output={self.right_PWM}")
        # pwms[4] = int(self.right_PWM)

        

        arduino_data = angles + tuple(pwms)
        # self.arduino.sender(arduino_data)  # Send to Arduino
        self.spyder_pub.publish(Int32MultiArray(data=list(arduino_data)))
        # print("Sent to Arduino:", arduino_data)

        

        
    def compute_steering_angles(self, lin_vel, ang_vel):
        if abs(ang_vel) < 1e-4:
            self.steering_angle.publish(Float64(data=0.0))
            return (0, 0, 0, 0)

        R_inst = lin_vel / ang_vel
        FL = atan(self.WHEELBASE / (R_inst - self.TRACK_WIDTH / 2))
        FR = atan(self.WHEELBASE / (R_inst + self.TRACK_WIDTH / 2))
        RL = -FL
        RR = -FR
        FL = max(-0.9, min(FL, 0.9))
        FR = max(-0.9, min(FR, 0.9))
        RL = max(-0.9, min(RL, 0.9))
        RR = max(-0.9, min(RR, 0.9))

        Steering_center = (FL + FR) / 2.0
        print(Steering_center)
        
        self.steering_angle.publish(Float64(data=Steering_center))

        if lin_vel < 0:
            return tuple(int(-degrees(a)) for a in (RL, RR, FL, FR))   #FR, FL, RR, RL
        else:
            return tuple(int(degrees(a)) for a in (FL, FR, RL, RR))

        

    def vel_to_pwm(self, v):
        v = max(-self.max_motor_velocity, min(self.max_motor_velocity, v))
        # print(f"velocity={v}")
        pwm = int((v / self.max_motor_velocity) * 255)
        # print(pwm)
        return pwm if abs(pwm) >= self.deadband_pwm else 0

    def compute_velocities(self, lin_vel, ang_vel):
        if abs(lin_vel) < 1e-4 and abs(ang_vel) < 1e-4:
            return [0] * 6

        # Special case: straight motion
        if abs(ang_vel) < 1e-4:
            pwm = self.vel_to_pwm(lin_vel)
            return [pwm, pwm, pwm, pwm, pwm, pwm]

        # Special case: Z-spin
        if abs(lin_vel) < 1e-4:
            pwm = self.vel_to_pwm(ang_vel * 0.25)
            return [-pwm, -pwm, -pwm, pwm, pwm, pwm]

        

        # Turning velocities
        v = max(-1.0, min(1.0, lin_vel / self.linear_limit))
        # r = max(-1.0, min(1.0, ang_vel / self.angular_limit))

        if 0 < abs(v) < self.min_lin_threshold:
            v = self.min_lin_threshold * (1 if v > 0 else -1)

        b = self.d2 ** 2
        c = (R + self.d3) ** 2
        d = (R - self.d3) ** 2
        e = R - self.d1
        f = R + self.d1

        abs_v1 = v * sqrt(b + d)
        abs_v2 = v * e
        abs_v3 = v * sqrt(b + d)
        abs_v4 = v * sqrt(b + c)
        abs_v5 = v * f
        abs_v6 = v * sqrt(b + c)

        max_abs = max(abs(abs_v1), abs(abs_v2), abs(abs_v3), abs(abs_v4), abs(abs_v5), abs(abs_v6))
        if max_abs > 0:
            abs_v1 = (abs_v1 / max_abs) * MAX_VEL
            abs_v2 = (abs_v2 / max_abs) * MAX_VEL
            abs_v3 = (abs_v3 / max_abs) * MAX_VEL
            abs_v4 = (abs_v4 / max_abs) * MAX_VEL
            abs_v5 = (abs_v5 / max_abs) * MAX_VEL
            abs_v6 = (abs_v6 / max_abs) * MAX_VEL

        if lin_vel < 0:
            vel = [abs_v4, abs_v5, abs_v6, abs_v1, abs_v2, abs_v3] if ang_vel > 0 else \
                  [abs_v1, abs_v2, abs_v3, abs_v4, abs_v5, abs_v6]
        else:
            vel = [abs_v4, abs_v5, abs_v6, abs_v1, abs_v2, abs_v3] if ang_vel < 0 else \
                  [abs_v1, abs_v2, abs_v3, abs_v4, abs_v5, abs_v6]



        return [self.vel_to_pwm(v) for v in vel]

# class EncoderSubscribe:

#     def __init__(self, node: Node):
#         self.node = node
#         self.prev_left_tick = None
#         self.prev_right_tick = None
#         self.curr_left_tick = None
#         self.curr_right_tick = None
#         self.speed_right = 0.0
#         self.speed_left = 0.0

        

#         self.node.create_subscription(Int32, 'right_ticks', self.right_callback, 10)
#         self.node.create_subscription(Int32, 'left_ticks', self.left_callback, 10)
#         self.node.get_logger().info('Subscribed to /wheel_tick_node')


#     def right_callback(self, msg):
#         self.curr_right_tick = msg.data
#         curr_time = time.time()

#         if self.prev_right_tick is not None:
#             delta_right = self.curr_right_tick - self.prev_right_tick
#             delta_time = curr_time - self.prev_time_right  # seconds
#             if delta_time > 0:
#                 self.speed_right = (delta_right / 286) / delta_time  # m/s
#                 self.actual_pwm_right = self.node.vel_to_pwm(self.speed_right)
#                 print(f"[Speed] Δticks={delta_right}, Δt={delta_time:.4f}s, Speed={self.speed_right:.4f} m/s, PWM={self.actual_pwm_right}")
        
#         self.prev_right_tick = self.curr_right_tick
#         self.prev_time_right = curr_time


#     def left_callback(self, msg):
#         self.curr_left_tick = msg.data
#         if self.prev_left_tick is not None:
#             delta_left = self.curr_left_tick - self.prev_left_tick
#             self.speed_left = delta_left / 2.86      # 286 ticks per meter - 2.86 for 10ms  (change to 1.43 if velocity is 0.5m/s) //// Actual Speed
#             # self.node.get_logger().info(f"[Left] Speed: {self.speed_left:.3f} m/s")
#         self.prev_left_tick = self.curr_left_tick
#         if self.prev_left_tick is not None:
#             delta_left = self.curr_left_tick - self.prev_left_tick
#             self.speed_left = delta_left / 2.86      # 286 ticks per meter - 2.86 for 10ms  (change to 1.43 if velocity is 0.5m/s) //// Actual Speed
#             # self.node.get_logger().info(f"[Left] Speed: {self.speed_left:.3f} m/s")
#         self.prev_left_tick = self.curr_left_tick

    



def main(args=None):
    rclpy.init(args=args)
    cmd_node = CmdVelSubscriber()
    # print(f"[PID] Right: Setpoint={cmd_node.right_pid_setpoint:.2f}, Actual={cmd_node.right_PWM:.2f}")
    try:
        rclpy.spin(cmd_node)
    except KeyboardInterrupt:
        cmd_node.get_logger().info('Node stopped by user.')
    finally:
        if rclpy.ok():
            cmd_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()


