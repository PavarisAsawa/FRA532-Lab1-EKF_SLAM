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
from state_estimator.dead_reckoning import *
from state_estimator.ekf import *
from state_estimator.MotionIMU import *
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
        
        if self.plot:
            if len(results) > 0:
                self.plot_results(results)
            else:
                self.get_logger().warn('No data received from any topic!')
    
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
    
    def plot_results(self, results):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Path Analysis Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        names = [r['name'] for r in results]
        distances = [r['distance'] for r in results]
        heading_changes = [r['heading_change'] for r in results]
        
        # Plot 1: Distance comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(names, distances, color=['blue', 'green', 'red'][:len(names)])
        ax1.set_ylabel('Distance (m)', fontsize=12)
        ax1.set_title('Total Distance Traveled', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for bar, dist in zip(bars1, distances):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{dist:.2f}m', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Heading change comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(names, heading_changes, color=['blue', 'green', 'red'][:len(names)])
        ax2.set_ylabel('Heading Change (degrees)', fontsize=12)
        ax2.set_title('Total Heading Change', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        for bar, heading in zip(bars2, heading_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{heading:.1f}°', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
        
        # Plot 3: 2D Path trajectories
        ax3 = axes[1, 0]
        colors = ['blue', 'green', 'red']
        for i, result in enumerate(results):
            if result['name'] == 'Wheel Odometry' and self.wheel_poses:
                x = [pose.pose.position.x for pose in self.wheel_poses]
                y = [pose.pose.position.y for pose in self.wheel_poses]
                ax3.plot(x, y, color=colors[i], label=result['name'], linewidth=2, alpha=0.7)
                ax3.scatter(x[0], y[0], color=colors[i], s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
                ax3.scatter(x[-1], y[-1], color=colors[i], s=100, marker='s', edgecolors='black', linewidths=2, zorder=5)
            elif result['name'] == 'EKF' and self.ekf_poses:
                x = [pose.pose.position.x for pose in self.ekf_poses]
                y = [pose.pose.position.y for pose in self.ekf_poses]
                ax3.plot(x, y, color=colors[i], label=result['name'], linewidth=2, alpha=0.7)
                ax3.scatter(x[0], y[0], color=colors[i], s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
                ax3.scatter(x[-1], y[-1], color=colors[i], s=100, marker='s', edgecolors='black', linewidths=2, zorder=5)
            elif result['name'] == 'ICP/SLAM' and self.icp_poses:
                x = [pose.pose.position.x for pose in self.icp_poses]
                y = [pose.pose.position.y for pose in self.icp_poses]
                ax3.plot(x, y, color=colors[i], label=result['name'], linewidth=2, alpha=0.7)
                ax3.scatter(x[0], y[0], color=colors[i], s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
                ax3.scatter(x[-1], y[-1], color=colors[i], s=100, marker='s', edgecolors='black', linewidths=2, zorder=5)
        
        ax3.set_xlabel('X Position (m)', fontsize=12)
        ax3.set_ylabel('Y Position (m)', fontsize=12)
        ax3.set_title('2D Path Trajectories', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = [['Method', 'Distance (m)', 'Heading (°)', '# Poses']]
        for r in results:
            table_data.append([
                r['name'],
                f"{r['distance']:.3f}",
                f"{r['heading_change']:.1f}",
                f"{r['num_poses']}"
            ])
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.35, 0.25, 0.25, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.tight_layout()
        plt.savefig('path_analysis_results.png', dpi=300, bbox_inches='tight')
        self.get_logger().info(f'\nPlot saved as: path_analysis_results.png')
        plt.show()

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