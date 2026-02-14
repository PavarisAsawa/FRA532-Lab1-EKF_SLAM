import signal
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import TransformStamped, Quaternion , PoseStamped
from nav_msgs.msg import Odometry ,Path
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from .state_estimator.dead_reckoning import *
from .state_estimator.ekf import *
from .state_estimator.MotionIMU import *

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import math

class OdometryNode(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # Joint state sub
        self.joint_sub = self.create_subscription( JointState, '/joint_states', self.joint_callback, 10)
        # IMU sub
        self.imu_sub = self.create_subscription( Imu, '/imu', self.imu_callback, 10)
        
        # --- Publishers ---
        self.path_pub = self.create_publisher(Path, '/ekf_path', 10)
        self.wheel_path_pub = self.create_publisher(Path, '/wheel_path', 10)
        self.wheel_imu_path_pub = self.create_publisher(Path, '/wheel_imu_path', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.wheel_odom_pub  = self.create_publisher(Odometry, '/wheel_odom', 10)
        self.wheel_imu_odom_pub  = self.create_publisher(Odometry, '/wheel_imu_odom', 10)

        # --- Path Data Container ---
        self.path = Path()
        self.path.header.frame_id = 'odom'
        self.wheel_path = Path()
        self.wheel_path.header.frame_id = 'odom'
        self.wheel_imu_path = Path()
        self.wheel_imu_path.header.frame_id = 'odom'

        # --- Wheel odom
        self.wheel_odom = DeadReckoning()
        self.wheel_imu = DiffDriveIMU()
        self.ekf = EKF()

        # --- Initial Condition
        self.joint_pos = [0,0]
        self.joint_vel = [0,0]
        self.x , self.y ,self.theta = 0,0,0
        self.euler = [0,0,0]
        self.calibrated_yaw = 0.0
        self.imu_init = False
        self.joint_init = False

        # --- IMU Calibration ---
        self.imu_yaw_offset = 0.0
        self.imu_calibrated = False
        self.calibration_samples = []
        self.calibration_sample_count = 50  # Collect 50 samples (~1 second at 50Hz)
        
        self.get_logger().info('='*60)
        self.get_logger().info('Starting IMU calibration...')
        self.get_logger().info('Please keep the robot STATIONARY!')
        self.get_logger().info('='*60)

        # --- Timer
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.timer_callback)

        # --- Matplotlib setup
        self.show_animation = False
        if self.show_animation:
            plt.ion()
            self.xa , self.ya = [] , []
            self.xb , self.yb = [] , []
            self.xs , self.ys = [] , []

            self.ellipse = None
            self.fig, self.ax = plt.subplots()
            self.wheel_line, = self.ax.plot([], [], 'b-', linewidth=2 ,label="Wheel Odom")
            self.wheel_imu_line, = self.ax.plot([], [], 'k-', linewidth=2 ,label="Wheel IMU")
            self.ekf_line, = self.ax.plot([], [], 'r-', linewidth=2 ,label="EKF")
            
            self.ax.legend(fontsize=8)
            self.set_plot_axis()
            self.animation_frames = []
            self.anim = None

    def set_plot_axis(self):
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Live Odometry")
        self.ax.legend(fontsize=8,loc='lower left')
        self.ax.set_xlim(-5, 22.5)
        self.ax.set_ylim(-25.0, 12.5)
        self.ax.set_aspect('equal')
        self.ax.grid(which='both', linestyle='--', color='gray', alpha=0.5)

    def joint_callback(self, msg):
        '''
        name=['wheel_left_joint', 'wheel_right_joint']
        '''
        self.joint_pos, self.joint_vel = msg.position , msg.velocity
        self.joint_init = True

    def imu_callback(self, msg):
        quaternion = msg.orientation
        euler = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        raw_yaw = euler[2]
        
        # --- IMU Calibration Phase ---
        if not self.imu_calibrated:
            self.calibration_samples.append(raw_yaw)
            
            if len(self.calibration_samples) >= self.calibration_sample_count:
                # Calculate average yaw as offset
                self.imu_yaw_offset = sum(self.calibration_samples) / len(self.calibration_samples)
                self.imu_calibrated = True
                
                self.get_logger().info('='*60)
                self.get_logger().info('IMU Calibration COMPLETE!')
                self.get_logger().info(f'Yaw offset: {math.degrees(self.imu_yaw_offset):.2f}°')
                self.get_logger().info('Robot ready to move!')
                self.get_logger().info('='*60)
                
                self.imu_init = True
            else:
                # Still calibrating - show progress
                progress = len(self.calibration_samples)
                if progress % 10 == 0:
                    self.get_logger().info(f'Calibrating IMU... {progress}/{self.calibration_sample_count}')
            return
        
        # --- Apply calibration offset ---
        self.calibrated_yaw = raw_yaw - self.imu_yaw_offset
        
        # Normalize to [-pi, pi]
        self.calibrated_yaw = math.atan2(math.sin(self.calibrated_yaw), math.cos(self.calibrated_yaw))
        
        # Update euler with calibrated yaw
        self.euler = [euler[0], euler[1], self.calibrated_yaw]

    def timer_callback(self):
        if not self.joint_init or not self.imu_init or not self.imu_calibrated: 
            return 
        
        # ---------------------------- Update estimator ---------------------------- #
        self.wheel_odom.predict(self.joint_pos)
        self.wheel_imu.predict(self.joint_pos, z_yaw=self.calibrated_yaw)
        self.x , self.y ,self.theta = self.ekf.predict(self.joint_pos, [0, 0, self.calibrated_yaw])

        # ---------------------------- Set pub ------------------------------------- #        
        now = self.get_clock().now().to_msg()
        
        # EKF
        self.publish_path(self.x, self.y, self.theta, now, self.path, self.path_pub, frame_id="odom")
        self.publish_odom(self.x, self.y, self.theta, now, self.odom_pub, parent_frame="odom", child_frame="base_link", publish_tf=True)

        # Wheel odom
        self.publish_path(self.wheel_odom.x, self.wheel_odom.y, self.wheel_odom.theta, now,
                                self.wheel_path, self.wheel_path_pub, frame_id="odom")
        self.publish_odom(self.wheel_odom.x, self.wheel_odom.y, self.wheel_odom.theta, now,
                                self.wheel_odom_pub, parent_frame="odom", child_frame="base_link", publish_tf=False)
        
        # Wheel_imu odom
        self.publish_path(self.wheel_imu.x, self.wheel_imu.y, self.wheel_imu.theta, now,
                                self.wheel_imu_path, self.wheel_imu_path_pub, frame_id="odom")
        self.publish_odom(self.wheel_imu.x, self.wheel_imu.y, self.wheel_imu.theta, now,
                                self.wheel_imu_odom_pub, parent_frame="odom", child_frame="base_link", publish_tf=False)
        
        # ---------------------------- Plot ---------------------------------------- #        
        if self.show_animation:
            self.xs.append(self.x)
            self.ys.append(self.y)
            self.xa.append(self.wheel_odom.x)
            self.ya.append(self.wheel_odom.y)
            self.xb.append(self.wheel_imu.x)
            self.yb.append(self.wheel_imu.y)
            
            self.wheel_line.set_data(self.xa, self.ya)
            self.ekf_line.set_data(self.xs, self.ys)
            self.wheel_imu_line.set_data(self.xb, self.yb)
            plt.pause(0.005)
    
    def publish_path(
        self,
        x: float,
        y: float,
        theta: float,
        timestamp,
        path_msg: Path,
        path_pub,
        frame_id: str = "odom",
        max_len: int = 10_000_000,
    ):
        """Publish a Path to any topic via the given publisher + Path container."""
        pose = PoseStamped()
        pose.header.stamp = timestamp
        pose.header.frame_id = frame_id

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, float(theta))
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q

        path_msg.header.stamp = timestamp
        path_msg.header.frame_id = frame_id
        path_msg.poses.append(pose)

        if len(path_msg.poses) > max_len:
            path_msg.poses.pop(0)

        path_pub.publish(path_msg)

    def publish_odom(
        self,
        x: float,
        y: float,
        theta: float,
        timestamp,
        odom_pub,
        parent_frame: str = "odom",
        child_frame: str = "base_link",
        publish_tf: bool = True,
    ):
        """Publish Odometry to any topic (and optionally TF) with configurable frame names."""
        q = quaternion_from_euler(0.0, 0.0, float(theta))

        # 1) TF (optional)
        if publish_tf:
            t = TransformStamped()
            t.header.stamp = timestamp
            t.header.frame_id = parent_frame
            t.child_frame_id = child_frame

            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = 0.0

            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.tf_broadcaster.sendTransform(t)

        # 2) Odometry msg
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = parent_frame
        odom.child_frame_id = child_frame

        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom_pub.publish(odom)
        
def main():
    rclpy.init(args=None)
    subscriber = OdometryNode()
    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()