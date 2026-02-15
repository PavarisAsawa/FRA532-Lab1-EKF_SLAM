import signal
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import TransformStamped, Quaternion, PoseStamped
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from .state_estimator.dead_reckoning import *
from .state_estimator.ekf import *
from .state_estimator.MotionIMU import *
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import math

class EvalNode(Node):
    def __init__(self):
        super().__init__('odometry_analyzer')
        
        # Storage for path data
        self.wheel_poses = []
        self.ekf_poses = []
        self.icp_poses = []
        
        # Flags to track if we've received data
        self.wheel_received = False
        self.ekf_received = False
        self.icp_received = False
        #
        self.plot = False
        # Wheeled only sub
        self.wheel_sub = self.create_subscription(
            Path, '/wheel_path', self.wheel_cb, 10)
        
        # EKF sub
        self.ekf_sub = self.create_subscription(
            Path, '/ekf_path', self.ekf_cb, 10)
        
        # ICP sub
        self.icp_sub = self.create_subscription(
            Path, '/robot_path', self.icp_cb, 10)
        
        self.get_logger().info('Odometry Analyzer Node Started - collecting data...')
        self.get_logger().info('Press Ctrl+C to stop and see results')
    
    def wheel_cb(self, msg):
        if len(msg.poses) > 0:
            self.wheel_poses = msg.poses
            self.wheel_received = True
    
    def ekf_cb(self, msg):
        if len(msg.poses) > 0:
            self.ekf_poses = msg.poses
            self.ekf_received = True
            self.display_final_results()

    def icp_cb(self, msg):
        if len(msg.poses) > 0:
            self.icp_poses = msg.poses
            self.icp_received = True
        
    
    def calculate_metrics(self, poses, name):
        """Calculate distance and heading change from initial to final pose"""
        if len(poses) < 2:
            self.get_logger().warn(f'{name}: Not enough data (need at least 2 poses)')
            return None
        
        # Get initial and final poses
        initial_pose = poses[0].pose
        final_pose = poses[-1].pose
        
        # Calculate position distance
        dx = final_pose.position.x - initial_pose.position.x
        dy = final_pose.position.y - initial_pose.position.y
        dz = final_pose.position.z - initial_pose.position.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calculate heading change
        initial_quat = [
            initial_pose.orientation.x,
            initial_pose.orientation.y,
            initial_pose.orientation.z,
            initial_pose.orientation.w
        ]
        final_quat = [
            final_pose.orientation.x,
            final_pose.orientation.y,
            final_pose.orientation.z,
            final_pose.orientation.w
        ]
        
        initial_euler = euler_from_quaternion(initial_quat)
        final_euler = euler_from_quaternion(final_quat)
        
        # Heading change (yaw difference)
        heading_change = final_euler[2] - initial_euler[2]
        
        # Normalize to [-pi, pi]
        heading_change = math.atan2(math.sin(heading_change), math.cos(heading_change))
        
        return {
            'name': name,
            'initial_pos': (initial_pose.position.x, initial_pose.position.y, initial_pose.position.z),
            'final_pos': (final_pose.position.x, final_pose.position.y, final_pose.position.z),
            'distance': distance,
            'initial_heading': math.degrees(initial_euler[2]),
            'final_heading': math.degrees(final_euler[2]),
            'heading_change': math.degrees(heading_change),
            'num_poses': len(poses)
        }
    
    def display_final_results(self):
        """Display final results at the end"""
        self.get_logger().info('\n' + '='*80)
        self.get_logger().info('FINAL PATH ANALYSIS RESULTS')
        self.get_logger().info('='*80)
        
        results = []
        
        if self.wheel_received:
            wheel_metrics = self.calculate_metrics(self.wheel_poses, 'Wheel Odometry')
            if wheel_metrics:
                results.append(wheel_metrics)
                self.print_metrics(wheel_metrics)
        
        if self.ekf_received:
            ekf_metrics = self.calculate_metrics(self.ekf_poses, 'EKF')
            if ekf_metrics:
                results.append(ekf_metrics)
                self.print_metrics(ekf_metrics)
        
        if self.icp_received:
            icp_metrics = self.calculate_metrics(self.icp_poses, 'ICP/SLAM')
            if icp_metrics:
                results.append(icp_metrics)
                self.print_metrics(icp_metrics)
    
    def print_metrics(self, metrics):
        """Pretty print metrics for a single path"""
        self.get_logger().info(f'\n--- {metrics["name"]} ---')
        self.get_logger().info(f'Number of poses: {metrics["num_poses"]}')
        self.get_logger().info(f'Initial Position: ({metrics["initial_pos"][0]:.3f}, {metrics["initial_pos"][1]:.3f}, {metrics["initial_pos"][2]:.3f})')
        self.get_logger().info(f'Final Position:   ({metrics["final_pos"][0]:.3f}, {metrics["final_pos"][1]:.3f}, {metrics["final_pos"][2]:.3f})')
        self.get_logger().info(f'Total Distance: {metrics["distance"]:.3f} m')
        self.get_logger().info(f'Initial Heading: {metrics["initial_heading"]:.2f}°')
        self.get_logger().info(f'Final Heading:   {metrics["final_heading"]:.2f}°')
        self.get_logger().info(f'Heading Change:  {metrics["heading_change"]:.2f}°')
    

def main():
    rclpy.init(args=None)
    subscriber = EvalNode()
    
    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        subscriber.get_logger().info('\n\nStopping data collection...')
        subscriber.display_final_results()
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()