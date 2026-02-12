import signal
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import TransformStamped, Quaternion , PoseStamped
from nav_msgs.msg import Odometry ,Path, OccupancyGrid
from std_msgs.msg import Header

from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from laser_geometry import LaserProjection

from state_estimator.dead_reckoning import *
from state_estimator.ekf import *

from state_estimator.MotionIMU import *
from state_estimator.icp3 import *
from state_estimator.scan import *

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

class KeyFrame:
    """Store keyframe information"""
    def __init__(self, pose, scan, timestamp):
        self.pose = pose.copy()  # [x, y, theta]
        self.scan = scan.copy()  # Point cloud in base_link frame
        self.timestamp = timestamp
        self.scan_in_map = None  # Will be computed when added to map

class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_slam_node')
        # --- QoS profile for /scan ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=300
        )

        # --- Subscribers ---
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, qos_profile)
        
        # --- Publishers ---
        self.icp_path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.ekf_path_pub = self.create_publisher(Path, '/ekf_path', 10)  # Add EKF path for comparison
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)
        self.current_scan_pub = self.create_publisher(PointCloud2, '/current_scan', 10)  # Visualize current scan
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # --- Timer ---
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        # --- Buffer ---
        self.joint_pos, self.joint_vel = [0, 0], [0, 0]
        self.euler = [0, 0, 0]

        # --- Estimator & SLAM --- 
        self.ekf = EKF()
        self.ekf_pose = np.array([0.0, 0.0, 0.0])
        self.prev_ekf_pose = np.array([0.0, 0.0, 0.0])
        
        # --- ICP SLAM Position ---
        self.icp_pose = np.array([0.0, 0.0, 0.0])
        
        # --- Keyframe Management ---
        self.keyframes = []
        self.last_keyframe_pose = np.array([0.0, 0.0, 0.0])
        
        # RELAXED keyframe thresholds for debugging
        self.keyframe_distance_threshold = 0.30  # Increased from 0.15
        self.keyframe_angle_threshold = np.deg2rad(25)  # Increased from 15
        self.min_keyframe_interval = 0.5
        self.last_keyframe_time = 0.0
        
        # --- Store point cloud ---
        self.current_scan = None
        self.local_map_point_cloud = np.array([]).reshape(0, 2)
        
        # --- Map management - MORE CONSERVATIVE ---
        self.max_map_size = 15000  # Reduced from 10000
        self.icp_map_size = 2000  # Reduced from 3000
        self.local_map_radius = 8.0  # Reduced from 10.0 - use closer points only
        
        # --- Path ---
        self.icp_path_msg = Path()
        self.icp_path_msg.header.frame_id = 'odom'
        self.ekf_path_msg = Path()
        self.ekf_path_msg.header.frame_id = 'odom'
        
        # --- Statistics & Diagnostics ---
        self.icp_success_count = 0
        self.icp_fail_count = 0
        self.total_scans = 0
        self.keyframe_count = 0
        self.large_correction_count = 0
        self.consecutive_failures = 0
        
        # --- DIAGNOSTIC MODE ---
        self.enable_icp = True  # Can disable to see pure EKF drift
        self.max_consecutive_failures = 20  # Reset if too many failures
        
        self.get_logger().info('='*60)
        self.get_logger().info('ICP SLAM NODE - DIAGNOSTIC MODE')
        self.get_logger().info('Publishing /ekf_path for comparison with /robot_path')
        self.get_logger().info('='*60)

    def is_keyframe(self, current_pose, current_time):
        """Determine if current pose should be a keyframe"""
        if (current_time - self.last_keyframe_time) < self.min_keyframe_interval:
            return False
        
        if len(self.keyframes) == 0:
            return True
        
        delta = current_pose - self.last_keyframe_pose
        distance = np.sqrt(delta[0]**2 + delta[1]**2)
        angle = abs(wrap(delta[2]))
        
        if distance > self.keyframe_distance_threshold or angle > self.keyframe_angle_threshold:
            return True
        
        return False

    def get_local_map_for_icp(self, current_pose, radius=None):
        """Extract local map around current position for ICP"""
        if radius is None:
            radius = self.local_map_radius
        
        if self.local_map_point_cloud.shape[0] == 0:
            return np.array([]).reshape(0, 2)
        
        distances = np.sqrt(
            (self.local_map_point_cloud[:, 0] - current_pose[0])**2 + 
            (self.local_map_point_cloud[:, 1] - current_pose[1])**2
        )
        
        mask = distances < radius
        local_points = self.local_map_point_cloud[mask]
        
        # Log local map stats
        if self.total_scans % 50 == 0:
            self.get_logger().info(
                f'Local map: {local_points.shape[0]}/{self.local_map_point_cloud.shape[0]} points '
                f'within {radius:.1f}m of ({current_pose[0]:.2f}, {current_pose[1]:.2f})'
            )
        
        if local_points.shape[0] > self.icp_map_size:
            local_points = local_points[1000:-1,:]
            # indices = np.random.choice(local_points.shape[0], self.icp_map_size, replace=False)
            # local_points = local_points[indices]
        
        return local_points

    def add_keyframe(self, pose, scan, timestamp):
        """Add a new keyframe and update the map"""
        kf = KeyFrame(pose, scan, timestamp)
        kf.scan_in_map = pointcloud_to_odom(scan, pose[0], pose[1], pose[2])
        
        self.keyframes.append(kf)
        self.keyframe_count += 1
        
        if self.local_map_point_cloud.shape[0] == 0:
            self.local_map_point_cloud = kf.scan_in_map
        else:
            self.local_map_point_cloud = np.vstack((self.local_map_point_cloud, kf.scan_in_map))
        
        # More aggressive filtering
        if self.local_map_point_cloud.shape[0] > self.max_map_size:
            self.get_logger().warn(f'Map too large ({self.local_map_point_cloud.shape[0]} points), filtering...')
            self.local_map_point_cloud = remove_outliers(self.local_map_point_cloud, 0.15, 5)
            self.local_map_point_cloud = voxel_grid_filter(self.local_map_point_cloud, leaf_size=0.08)
            self.get_logger().info(f'Map filtered to {self.local_map_point_cloud.shape[0]} points')
        
        self.last_keyframe_pose = pose.copy()
        self.last_keyframe_time = timestamp
        
        self.get_logger().info(
            f'Keyframe #{self.keyframe_count} at ({pose[0]:.2f}, {pose[1]:.2f}, {np.rad2deg(pose[2]):.1f}°) '
            f'- Map: {self.local_map_point_cloud.shape[0]} points'
        )

    def laser_scan_callback(self, msg):
        self.total_scans += 1
        point_cloud = scan_to_pointcloud(msg)
        if point_cloud.shape[0] < 25: 
            print("RETUNR NO ENOUGH POINT CLOUD")
            return
        
        # Preprocessing
        point_cloud = remove_outliers(point_cloud, 0.20, 4)
        point_cloud = voxel_grid_filter(point_cloud, leaf_size=0.02)
        self.current_scan = point_cloud
        
        current_time = self.get_clock().now().seconds_nanoseconds()[0] + \
                      self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        

        # -------------------------------- 
        # Get EKF odometry estimate
        # --------------------------------   
        if self.prev_ekf_pose is None:
            self.prev_ekf_pose = self.ekf_pose.copy()
        
        delta_ekf = self.ekf_pose - self.prev_ekf_pose
        delta_ekf[2] = wrap(delta_ekf[2])
        
        self.prev_ekf_pose = self.ekf_pose.copy()

        # Check for unreasonable EKF deltas
        delta_dist = np.sqrt(delta_ekf[0]**2 + delta_ekf[1]**2)

        if delta_dist > 0.5:  # 50cm in one scan is way too much
            self.get_logger().error(f'HUGE EKF DELTA: {delta_dist:.3f}m - Check odometry!')
            return
        
        predicted_icp_pose = self.icp_pose + delta_ekf
        predicted_icp_pose[2] = wrap(predicted_icp_pose[2])

        if self.current_scan is None or self.current_scan.shape[0] < 80:
            self.icp_pose = predicted_icp_pose
            return
        # -------------------------------- 
        # ICP Scan Matching
        # --------------------------------
        
        icp_correction_applied = False
        if self.enable_icp and len(self.keyframes) > 0:
            local_map = self.get_local_map_for_icp(predicted_icp_pose)
            
            if local_map.shape[0] > 100:  # Increased minimum map for 100 point
   
                # x, y, theta, count, error, success = icp_matching(
                #     previous_points=local_map,
                #     current_points=current_scan_in_map,
                #     init_x=None,
                #     init_y=None,
                #     init_theta=None,
                #     MAX_ITERATION=30,
                #     ERROR_BOUND=0.001
                # )
                
                x, y, theta, count, error, success = icp_matching(
                    previous_points=local_map,          # (N,2) in odom
                    current_points=self.current_scan,   # (M,2) in base_link
                    init_x=predicted_icp_pose[0],
                    init_y=predicted_icp_pose[1],
                    init_theta=predicted_icp_pose[2],
                    MAX_ITERATION=30,
                    ERROR_BOUND=0.001,
                    max_corr_dist=0.2   # strongly recommended
                )
                
                # DIAGNOSTIC: Calculate correction magnitude
                correction_dist = np.sqrt((x - predicted_icp_pose[0])**2 + 
                                        (y - predicted_icp_pose[1])**2)
                correction_angle = abs(wrap(theta - predicted_icp_pose[2]))
                
                # Log ICP details periodically
                if self.total_scans % 10 == 0:
                    self.get_logger().info(
                        f'ICP: err={error:.4f}, iter={count}, '
                        f'correction={correction_dist:.3f}m/{np.rad2deg(correction_angle):.1f}°, '
                        f'predicted=({predicted_icp_pose[0]:.2f},{predicted_icp_pose[1]:.2f}), '
                        f'result=({x:.2f},{y:.2f})'
                    )
                
                # MUCH MORE RELAXED validation for now
                if success and error < 0.20:  # Increased from 0.15
                    # Very permissive thresholds to see what ICP wants to do
                    if correction_dist < 0.1 and correction_angle < np.deg2rad(15):  # threshold
                        self.icp_pose = np.array([x, y, theta])
                        self.icp_success_count += 1
                        self.consecutive_failures = 0
                        icp_correction_applied = True
                        
                        # if correction_dist > 0.1:  # Still log large ones
                        #     self.get_logger().warn(
                        #         f'LARGE correction accepted: {correction_dist:.3f}m, {np.rad2deg(correction_angle):.1f}°'
                        #     )
                    else:
                        self.icp_pose = predicted_icp_pose
                        self.icp_fail_count += 1
                        self.large_correction_count += 1
                        self.consecutive_failures += 1
                        self.get_logger().error(
                            f'ICP correction REJECTED: {correction_dist:.3f}m, {np.rad2deg(correction_angle):.1f}° '
                            f'(consecutive failures: {self.consecutive_failures})'
                        )
                else:
                    self.get_logger().error(f'ICP NOT SUCCESS')
                    self.icp_pose = predicted_icp_pose
                    self.icp_fail_count += 1
                    self.consecutive_failures += 1
            else:
                self.icp_pose = predicted_icp_pose
                if self.total_scans % 20 == 0:
                    self.get_logger().warn(f'Insufficient local map: {local_map.shape[0]} points')
        else:
            self.icp_pose = predicted_icp_pose
        
        # Reset if too many consecutive failures
        if self.consecutive_failures > self.max_consecutive_failures:
            self.get_logger().error(
                f'TOO MANY FAILURES ({self.consecutive_failures}) - '
                f'Map may be corrupted. Consider resetting.'
            )
        
        
        # -------------------------------- 
        # Keyframe Decision
        # --------------------------------
        if self.is_keyframe(self.icp_pose, current_time):
            self.add_keyframe(self.icp_pose, self.current_scan, current_time)
        
        # -------------------------------- 
        # Publish
        # --------------------------------
        now = self.get_clock().now().to_msg()
        
        # Publish both ICP and EKF paths for comparison
        self.broadcast_tf(self.icp_pose[0], self.icp_pose[1], self.icp_pose[2], "odom", "base_link", now)
        self.publish_path(self.icp_path_pub, self.icp_path_msg, 
                         self.icp_pose[0], self.icp_pose[1], self.icp_pose[2], now)
        self.publish_path(self.ekf_path_pub, self.ekf_path_msg,
                         self.ekf_pose[0], self.ekf_pose[1], self.ekf_pose[2], now)
        
        # Publish current scan in map frame for visualization
        if self.current_scan is not None:
            scan_in_map = pointcloud_to_odom(
                self.current_scan,
                self.icp_pose[0],
                self.icp_pose[1],
                self.icp_pose[2]
            )
            self.publish_processed_pc(scan_in_map, now, "odom", self.current_scan_pub)
        
        # Publish map
        if self.total_scans % 10 == 0:
            self.publish_processed_pc(self.local_map_point_cloud, now, "odom", self.point_cloud_pub)

    def joint_callback(self, msg):
        self.joint_pos, self.joint_vel = msg.position, msg.velocity

    def imu_callback(self, msg):
        self.euler = euler_from_quaternion([msg.orientation.x, msg.orientation.y, 
                                           msg.orientation.z, msg.orientation.w])

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        self.ekf_pose = self.ekf.predict(self.joint_pos, [0, 0, self.euler[2]])
        self.ekf_pose[2] = wrap(self.ekf_pose[2])

    def publish_path(self, path_pub, path_msg, x, y, theta, timestamp):
        pose = PoseStamped()
        pose.header.stamp = timestamp
        pose.header.frame_id = path_msg.header.frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(x), float(y), 0.0

        q = quaternion_from_euler(0, 0, float(theta))
        pose.pose.orientation.x, pose.pose.orientation.y = q[0], q[1]
        pose.pose.orientation.z, pose.pose.orientation.w = q[2], q[3]

        path_msg.header.stamp = timestamp
        path_msg.poses.append(pose)

        if len(path_msg.poses) > 10000:
            path_msg.poses.pop(0)
        
        path_pub.publish(path_msg)

    def publish_processed_pc(self, points, timestamp, frame_id, publisher):
        if points is None or points.shape[0] == 0:
            return
        
        if points.shape[1] == 2:
            z_col = np.zeros((points.shape[0], 1))
            points_3d = np.hstack((points, z_col))
        else:
            points_3d = points

        header = Header()
        header.stamp = timestamp
        header.frame_id = frame_id
        pc_msg = pc2.create_cloud_xyz32(header, points_3d)
        publisher.publish(pc_msg)

    def broadcast_tf(self, x, y, theta, head, child, timestamp):
        t = TransformStamped()
        
        t.header.stamp = timestamp
        t.header.frame_id = head
        t.child_frame_id = child

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0

        q = quaternion_from_euler(0, 0, float(theta))
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init(args=None)
    node = ICPNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('='*60)
        node.get_logger().info(f'DIAGNOSTIC STATISTICS:')
        node.get_logger().info(f'  Total scans: {node.total_scans}')
        node.get_logger().info(f'  Keyframes: {node.keyframe_count}')
        node.get_logger().info(f'  Map points: {node.local_map_point_cloud.shape[0]}')
        node.get_logger().info(f'  ICP success: {node.icp_success_count}/{node.total_scans} '
                             f'({100*node.icp_success_count/max(node.total_scans,1):.1f}%)')
        node.get_logger().info(f'  ICP failures: {node.icp_fail_count}')
        node.get_logger().info(f'  Large corrections rejected: {node.large_correction_count}')
        node.get_logger().info(f'  Final ICP pose: ({node.icp_pose[0]:.2f}, {node.icp_pose[1]:.2f}, '
                             f'{np.rad2deg(node.icp_pose[2]):.1f}°)')
        node.get_logger().info(f'  Final EKF pose: ({node.ekf_pose[0]:.2f}, {node.ekf_pose[1]:.2f}, '
                             f'{np.rad2deg(node.ekf_pose[2]):.1f}°)')
        node.get_logger().info('='*60)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()