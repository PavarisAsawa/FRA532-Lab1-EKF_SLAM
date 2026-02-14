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

from .state_estimator.dead_reckoning import *
from .state_estimator.ekf import *

from .state_estimator.MotionIMU import *
from .state_estimator.icp3 import *
from .state_estimator.scan import *

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import math

class KeyFrame:
    """Store keyframe information"""
    def __init__(self, pose, scan, timestamp):
        self.pose = pose.copy()  # [x, y, theta]
        self.scan = scan.copy()  # Point cloud in base_link frame
        self.timestamp = timestamp
        self.scan_in_map = None  # Will be computed when added to map
        self.robot_position = pose[:2].copy()  # Store robot x, y when scan was taken

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
        self.ekf_path_pub = self.create_publisher(Path, '/ekf_path', 10)
        self.icp_map_pub = self.create_publisher(OccupancyGrid, '/icp_map', 10)  # Occupancy grid publisher
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)
        self.current_scan_pub = self.create_publisher(PointCloud2, '/current_scan', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # --- Occupancy Grid Parameters ---
        self.grid_resolution = 0.05  # 5cm per cell
        self.grid_width = 800  # 40m at 0.05m resolution
        self.grid_height = 800  # 40m at 0.05m resolution
        self.grid_origin_x = -20.0  # meters
        self.grid_origin_y = -20.0  # meters
        
        # --- Timer ---
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        # --- Buffer ---
        self.joint_pos, self.joint_vel = [0, 0], [0, 0]
        self.euler = [0, 0, 0]
        self.calibrated_yaw = 0.0

        # --- IMU Calibration ---
        self.imu_yaw_offset = 0.0
        self.imu_calibrated = False
        self.calibration_samples = []
        self.calibration_sample_count = 50  # Collect 50 samples

        # --- Estimator & SLAM --- 
        self.ekf = EKF()
        self.ekf_pose = np.array([0.0, 0.0, 0.0])
        self.prev_ekf_pose = np.array([0.0, 0.0, 0.0])
        
        # --- ICP SLAM Position ---
        self.icp_pose = np.array([0.0, 0.0, 0.0])
        
        # --- Keyframe Management ---
        self.keyframes = []
        self.last_keyframe_pose = np.array([0.0, 0.0, 0.0])
        
        # Keyframe thresholds
        self.keyframe_distance_threshold = 0.20  # 20cm
        self.keyframe_angle_threshold = np.deg2rad(10)  # 10 degrees
        self.min_keyframe_interval = 0.2  # 200ms minimum
        self.last_keyframe_time = 0.0
        
        # --- Store point cloud ---
        self.current_scan = None
        self.local_map_point_cloud = np.array([]).reshape(0, 2)
        
        # --- Map management ---
        self.max_map_size = 15000
        self.icp_map_size = 3000  # Use up to 3000 points for ICP
        self.local_map_radius = 6.0
        
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
        
        # Jump detection
        self.last_valid_pose = np.array([0.0, 0.0, 0.0])
        self.jump_threshold_dist = 0.5  # 50cm sudden jump is suspicious
        self.jump_threshold_angle = np.deg2rad(30)  # 30° sudden jump is suspicious
        self.jump_count = 0
        
        # Point cloud quality tracking
        self.min_scan_points = 30  # Minimum points in scan after filtering
        self.min_local_map_points = 100  # Minimum points in local map for ICP
        self.poor_scan_count = 0
        
        self.enable_icp = True
        self.max_consecutive_failures = 20
        
        # Initial condition
        self.joint_init = False
        self.imu_init = False 

        self.get_logger().info('='*60)
        self.get_logger().info('ICP SLAM NODE - JUMP PREVENTION')
        self.get_logger().info('='*60)
        self.get_logger().info('Starting IMU calibration...')
        self.get_logger().info('Please keep the robot STATIONARY!')
        self.get_logger().info('='*60)

    def create_occupancy_grid(self, timestamp):
        """Convert accumulated keyframes to occupancy grid with proper ray tracing"""
        if len(self.keyframes) == 0:
            return None
        
        # Create empty grid (unknown = -1)
        grid = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)
        
        # Process each keyframe
        for kf in self.keyframes:
            # Get robot position when this scan was taken
            robot_world_x = kf.pose[0]
            robot_world_y = kf.pose[1]
            
            robot_grid_x = int((robot_world_x - self.grid_origin_x) / self.grid_resolution)
            robot_grid_y = int((robot_world_y - self.grid_origin_y) / self.grid_resolution)
            
            # Get points in map frame for this keyframe
            scan_points = kf.scan_in_map
            
            # Process each point in this scan
            for i in range(0, len(scan_points), 2):  # Sample every 2nd point for efficiency
                point = scan_points[i]
                
                # Convert world coordinates to grid coordinates
                grid_x = int((point[0] - self.grid_origin_x) / self.grid_resolution)
                grid_y = int((point[1] - self.grid_origin_y) / self.grid_resolution)
                
                # Check if within bounds
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    # Mark obstacle
                    grid[grid_y, grid_x] = 100
                    
                    # Ray trace from robot position to obstacle
                    cells = self.bresenham_line(robot_grid_x, robot_grid_y, grid_x, grid_y)
                    
                    # Mark all cells except the last one as free
                    for cx, cy in cells[:-1]:
                        if 0 <= cx < self.grid_width and 0 <= cy < self.grid_height:
                            if grid[cy, cx] == -1:  # Only mark if unknown
                                grid[cy, cx] = 0  # Free space
        
        # Create OccupancyGrid message
        occ_grid = OccupancyGrid()
        occ_grid.header.stamp = timestamp
        occ_grid.header.frame_id = 'odom'
        
        occ_grid.info.resolution = self.grid_resolution
        occ_grid.info.width = self.grid_width
        occ_grid.info.height = self.grid_height
        occ_grid.info.origin.position.x = self.grid_origin_x
        occ_grid.info.origin.position.y = self.grid_origin_y
        occ_grid.info.origin.position.z = 0.0
        occ_grid.info.origin.orientation.w = 1.0
        
        # Flatten grid (row-major order)
        occ_grid.data = grid.flatten().tolist()
        
        return occ_grid

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm to get all cells between two points"""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            cells.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return cells

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
        
        # use random
        if local_points.shape[0] > self.icp_map_size:
            indices = np.random.choice(local_points.shape[0], self.icp_map_size, replace=False)
            local_points = local_points[indices]
        
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
        
        # Filtering
        if self.local_map_point_cloud.shape[0] > self.max_map_size:
            self.get_logger().warn(f'Map too large ({self.local_map_point_cloud.shape[0]} points), filtering...')
            self.local_map_point_cloud = remove_outliers(self.local_map_point_cloud, 0.15, 5)
            self.local_map_point_cloud = voxel_grid_filter(self.local_map_point_cloud, leaf_size=0.08)
            self.get_logger().info(f'Map filtered to {self.local_map_point_cloud.shape[0]} points')
        
        self.last_keyframe_pose = pose.copy()
        self.last_keyframe_time = timestamp
        
        if self.keyframe_count % 5 == 0:
            self.get_logger().info(
                f'KF #{self.keyframe_count}: ({pose[0]:.2f}, {pose[1]:.2f}, {np.rad2deg(pose[2]):.1f}°), '
                f'Map: {self.local_map_point_cloud.shape[0]}pts'
            )

    def detect_jump(self, new_pose, delta_ekf):
        """Detect if pose jumped unreasonably"""
        # Calculate actual change
        actual_delta = new_pose - self.last_valid_pose
        actual_dist = np.sqrt(actual_delta[0]**2 + actual_delta[1]**2)
        actual_angle = abs(wrap(actual_delta[2]))
        
        # Compare to expected EKF delta
        expected_dist = np.sqrt(delta_ekf[0]**2 + delta_ekf[1]**2)
        
        # Check if jump is way larger than expected
        dist_ratio = actual_dist / max(expected_dist, 0.01)  # Avoid division by zero
        
        is_jump = (actual_dist > self.jump_threshold_dist or 
                  actual_angle > self.jump_threshold_angle or
                  dist_ratio > 10.0)  # Moved 10x more than expected
        
        if is_jump:
            self.jump_count += 1
            self.get_logger().error(
                f'JUMP DETECTED #{self.jump_count}! '
                f'Moved {actual_dist:.3f}m/{np.rad2deg(actual_angle):.1f}° '
                f'(expected ~{expected_dist:.3f}m), ratio={dist_ratio:.1f}x'
            )
        
        return is_jump

    def laser_scan_callback(self, msg):
        
        if not self.joint_init or not self.imu_init or not self.imu_calibrated: 
            return 
        
        self.total_scans += 1
        
        # ========================================
        # STEP 1: Get and validate scan
        # ========================================
        point_cloud = scan_to_pointcloud(msg)
        initial_points = point_cloud.shape[0]
        
        if initial_points < 25:
            self.get_logger().warn(f'Scan #{self.total_scans}: Only {initial_points} raw points - SKIPPING')
            return
        
        # Preprocessing
        # point_cloud = remove_outliers(point_cloud, 0.50, 5)
        # point_cloud = voxel_grid_filter(point_cloud, leaf_size=0.08)
        final_points = point_cloud.shape[0]
        
        # Check if we lost too many points in filtering
        if final_points < self.min_scan_points:
            self.poor_scan_count += 1
            self.get_logger().warn(
                f'Scan #{self.total_scans}: Filtered {initial_points} -> {final_points} points '
                f'(< {self.min_scan_points} minimum) - USING ODOMETRY ONLY'
            )
            # Skip ICP, just use odometry
            if self.prev_ekf_pose is None:
                self.prev_ekf_pose = self.ekf_pose.copy()
            
            delta_ekf = self.ekf_pose - self.prev_ekf_pose
            delta_ekf[2] = wrap(delta_ekf[2])
            self.prev_ekf_pose = self.ekf_pose.copy()
            
            self.icp_pose += delta_ekf
            self.icp_pose[2] = wrap(self.icp_pose[2])
            
            # Still publish
            now = self.get_clock().now().to_msg()
            self.broadcast_tf(self.icp_pose[0], self.icp_pose[1], self.icp_pose[2], "odom", "base_link", now)
            self.publish_path(self.icp_path_pub, self.icp_path_msg, 
                             self.icp_pose[0], self.icp_pose[1], self.icp_pose[2], now)
            return
        
        self.current_scan = point_cloud
        current_time = self.get_clock().now().seconds_nanoseconds()[0] + \
                      self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        
        # ========================================
        # STEP 2: Get EKF prediction
        # ========================================
        if self.prev_ekf_pose is None:
            self.prev_ekf_pose = self.ekf_pose.copy()
        
        delta_ekf = self.ekf_pose - self.prev_ekf_pose
        delta_ekf[2] = wrap(delta_ekf[2])
        self.prev_ekf_pose = self.ekf_pose.copy()

        # Validate EKF delta
        delta_dist = np.sqrt(delta_ekf[0]**2 + delta_ekf[1]**2)
        if delta_dist > 0.5:
            self.get_logger().error(f'HUGE EKF DELTA: {delta_dist:.3f}m - SKIPPING this scan!')
            return
        
        predicted_icp_pose = self.icp_pose + delta_ekf
        predicted_icp_pose[2] = wrap(predicted_icp_pose[2])

        # ========================================
        # STEP 3: Run ICP if map exists
        # ========================================
        icp_correction_applied = False
        
        if self.enable_icp and len(self.keyframes) > 0:
            local_map = self.get_local_map_for_icp(predicted_icp_pose)
            
            # Log map statistics
            if self.total_scans % 50 == 0:
                self.get_logger().info(
                    f'Scan #{self.total_scans}: {final_points} scan pts, '
                    f'{local_map.shape[0]} local map pts, '
                    f'{self.local_map_point_cloud.shape[0]} total map pts'
                )
            
            if local_map.shape[0] >= self.min_local_map_points:
                # Run ICP
                x, y, theta, count, error, success = icp_matching(
                    previous_points=local_map,
                    current_points=self.current_scan,
                    init_x=predicted_icp_pose[0],
                    init_y=predicted_icp_pose[1],
                    init_theta=predicted_icp_pose[2],
                    MAX_ITERATION=40,
                    ERROR_BOUND=0.0005,
                    max_corr_dist=0.1
                )
                
                # Calculate correction
                correction_dist = np.sqrt((x - predicted_icp_pose[0])**2 + 
                                        (y - predicted_icp_pose[1])**2)
                correction_angle = abs(wrap(theta - predicted_icp_pose[2]))
                
                # Log periodically
                if self.total_scans % 10 == 0:
                    self.get_logger().info(
                        f'ICP: err={error:.4f}, iter={count}/{40}, '
                        f'corr={correction_dist:.3f}m/{np.rad2deg(correction_angle):.1f}°, '
                        f'success={success}'
                    )
                
                # Validate ICP result
                if success and error < 0.15:
                    # Accept correction if within bounds
                    if correction_dist < 0.15 and correction_angle < np.deg2rad(20):
                        # Check for jumps before accepting
                        candidate_pose = np.array([x, y, theta])
                        if not self.detect_jump(candidate_pose, delta_ekf):
                            self.icp_pose = candidate_pose
                            self.last_valid_pose = self.icp_pose.copy()
                            self.icp_success_count += 1
                            self.consecutive_failures = 0
                            icp_correction_applied = True
                        else:
                            # Jump detected - reject ICP, use prediction
                            self.get_logger().error('ICP result would cause JUMP - REJECTED')
                            self.icp_pose = predicted_icp_pose
                            self.icp_fail_count += 1
                            self.consecutive_failures += 1
                    else:
                        # Correction too large
                        self.icp_pose = predicted_icp_pose
                        self.icp_fail_count += 1
                        self.consecutive_failures += 1
                        if self.total_scans % 10 == 0:
                            self.get_logger().warn(
                                f'ICP correction too large: {correction_dist:.3f}m/{np.rad2deg(correction_angle):.1f}°'
                            )
                else:
                    # ICP failed or high error
                    self.icp_pose = predicted_icp_pose
                    self.icp_fail_count += 1
                    self.consecutive_failures += 1
                    if self.total_scans % 10 == 0:
                        self.get_logger().warn(f'ICP failed: success={success}, error={error:.4f}')
            else:
                # Not enough local map points
                self.icp_pose = predicted_icp_pose
                self.get_logger().warn(
                    f'Insufficient local map: {local_map.shape[0]} pts '
                    f'(need ≥{self.min_local_map_points})'
                )
        else:
            # No keyframes yet - use prediction
            self.icp_pose = predicted_icp_pose
        
        # Update last valid pose if no ICP correction (odometry only)
        if not icp_correction_applied:
            self.last_valid_pose = self.icp_pose.copy()
        
        # ========================================
        # STEP 4: Keyframe decision
        # ========================================
        if self.is_keyframe(self.icp_pose, current_time):
            self.add_keyframe(self.icp_pose, self.current_scan, current_time)
        
        # ========================================
        # STEP 5: Publish
        # ========================================
        now = self.get_clock().now().to_msg()
        
        self.broadcast_tf(self.icp_pose[0], self.icp_pose[1], self.icp_pose[2], "odom", "base_link", now)
        self.publish_path(self.icp_path_pub, self.icp_path_msg, 
                         self.icp_pose[0], self.icp_pose[1], self.icp_pose[2], now)
        
        # Publish current scan
        if self.current_scan is not None:
            scan_in_map = pointcloud_to_odom(
                self.current_scan,
                self.icp_pose[0],
                self.icp_pose[1],
                self.icp_pose[2]
            )
            self.publish_processed_pc(scan_in_map, now, "odom", self.current_scan_pub)
        
        # Publish occupancy grid and point cloud periodically
        if self.total_scans % 10 == 0 and len(self.keyframes) > 0:
            # Publish occupancy grid (now uses all keyframes with proper ray origins)
            occ_grid = self.create_occupancy_grid(now)
            if occ_grid is not None:
                self.icp_map_pub.publish(occ_grid)
            
            # Publish point cloud
            if self.local_map_point_cloud.shape[0] > 0:
                self.publish_processed_pc(self.local_map_point_cloud, now, "odom", self.point_cloud_pub)

    def joint_callback(self, msg):
        self.joint_pos, self.joint_vel = msg.position, msg.velocity
        self.joint_init = True

    def imu_callback(self, msg):
        euler = euler_from_quaternion([msg.orientation.x, msg.orientation.y, 
                                       msg.orientation.z, msg.orientation.w])
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
                # Still calibrating
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
        if not self.imu_calibrated:
            return
            
        self.ekf_pose = self.ekf.predict(self.joint_pos, [0, 0, self.calibrated_yaw])
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
        node.get_logger().info('FINAL STATISTICS:')
        node.get_logger().info(f'  Total scans: {node.total_scans}')
        node.get_logger().info(f'  Poor quality scans: {node.poor_scan_count} '
                             f'({100*node.poor_scan_count/max(node.total_scans,1):.1f}%)')
        node.get_logger().info(f'  Keyframes: {node.keyframe_count}')
        node.get_logger().info(f'  Map points: {node.local_map_point_cloud.shape[0]}')
        node.get_logger().info(f'  ICP success rate: {node.icp_success_count}/{node.total_scans} '
                             f'({100*node.icp_success_count/max(node.total_scans,1):.1f}%)')
        node.get_logger().info(f'  Jumps detected: {node.jump_count}')
        node.get_logger().info(f'  Final pose: ({node.icp_pose[0]:.2f}, {node.icp_pose[1]:.2f}, '
                             f'{np.rad2deg(node.icp_pose[2]):.1f}°)')
        node.get_logger().info('='*60)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()