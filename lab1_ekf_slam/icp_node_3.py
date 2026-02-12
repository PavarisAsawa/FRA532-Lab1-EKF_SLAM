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
from state_estimator.ekf2 import *

from state_estimator.MotionIMU import *
from state_estimator.icp import *
from state_estimator.scan import *

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class ICPNode(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # --- QoS profile for /scan ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Best effort communication (no guarantee of message delivery)
            history=HistoryPolicy.KEEP_LAST,  # Keep the last N messages
            depth=300  # Buffer size
        )

        # --- subscriber ---
        # Joint state sub
        self.joint_sub = self.create_subscription( JointState, '/joint_states', self.joint_callback, 10) # sensor_msgs/msg/JointState
        # IMU sub
        self.imu_sub = self.create_subscription( Imu, '/imu', self.imu_callback, 10)
        # Scan sub
        self.scan_sub = self.create_subscription( LaserScan, '/scan',  self.laser_scan_callback, qos_profile)
        
        
        # --- Publishers ---
        # self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.ekf_path_pub = self.create_publisher(Path, '/robot_path', 10) # Path Publisher for ICP
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)  # PointCloud2 Publisher
        self.tf_broadcaster = TransformBroadcaster(self)
        # Create publisher for map (OccupancyGrid)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 300)
        # --- Timer ---
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.timer_callback)
        # --- Buffer ---
        self.joint_pos, self.joint_vel = [0,0] , [0,0]
        self.euler = [0,0,0]
        self.laserscan = np.array([])

        # --- Estimator & SLAM --- 
        self.ekf = EKF()
        self.icp = ICP()

        # --- Position ---
        self.icp_x, self.icp_y, self.icp_theta = 0.0, 0.0, 0.0
        self.prev_ekf_pose = np.array([0.0, 0.0, 0.0]) # [x, y, theta]
        # --- Store point cloud
        self.point_cloud = None
        self.current_point_cloud = None  # Store the point cloud
        self.previous_point_cloud = None

        # --- Path
        self.icp_path = Path()
        self.icp_path.header.frame_id = 'map' # Must match RViz Fixed Frame
        self.ekf_path_msg = Path()
        self.ekf_path_msg.header.frame_id = 'odom' # Path นี้วาดบน Odom

        # --- Key frame
        self.last_keyframe_x = 0.0
        self.last_keyframe_y = 0.0
        self.last_keyframe_th = 0.0

    def laser_scan_callback(self, msg):
        # Prepocessing Point cloud
        point_cloud = scan_to_pointcloud(msg) # shape [2,n]

        if point_cloud.shape[1] < 50: return

        clear_outlier = remove_outliers(point_cloud,0.2 , 8) # Outlier treatment
        voxel_grid = voxel_grid_filter(clear_outlier, leaf_size=0.02) # down sampling with voxel filter,
        self.current_point_cloud = voxel_grid
        self.publish_processed_pc(voxel_grid, timestamp=self.get_clock().now().to_msg())
        # print(point_cloud.shape , clear_outlier.shape, voxel_grid.shape)

    def joint_callback(self, msg):
        '''name=['wheel_left_joint', 'wheel_right_joint']'''
        self.joint_pos, self.joint_vel = msg.position , msg.velocity

    def imu_callback(self, msg):
        self.euler = euler_from_quaternion([msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w])

    def timer_callback(self):
        # --- Set pub        
        now = self.get_clock().now().to_msg() # Set time 

        # -------------------------------------------------
        #  EKF
        # -------------------------------------------------
        ekf_x, ekf_y, ekf_th = self.ekf.predict(self.joint_pos, [0,0,self.euler[2]])
        self.broadcast_tf(ekf_x, ekf_y, ekf_th, "odom", "base_link",now)
        
        # -------------------------------------------------
        #  ICP
        # -------------------------------------------------
        # print("iiiiiiiiiiiiiiiiiiiiiiiiii")

        if self.current_point_cloud is not None and self.previous_point_cloud is not None:        
            print("-----------------")
            # --- ICP Matching
            R, T, success = self.icp.icp_matching(self.previous_point_cloud, self.current_point_cloud)
            if success:
                # print("sadasdasdsa")
                R_world = R
                T_world = T
                R_robot = R_world.T
                T_robot = -R_world.T @ T_world

                local_dx, local_dy = T_robot[0], T_robot[1]
                local_dtheta = np.arctan2(R_robot[1,0], R_robot[0,0])

                # local diff
                # local_dx = T[0]
                # local_dy = T[1]
                # local_dtheta = np.arctan2(R[1, 0], R[0, 0])

                # Transform to world frame
                cos_th = np.cos(self.icp_theta)
                sin_th = np.sin(self.icp_theta)
                global_dx = (local_dx * cos_th) - (local_dy * sin_th)
                global_dy = (local_dx * sin_th) + (local_dy * cos_th)

                # Update Global position
                self.icp_x += global_dx
                self.icp_y += global_dy
                self.icp_theta += local_dtheta
                self.icp_theta = np.arctan2(np.sin(self.icp_theta), np.cos(self.icp_theta))

                # -------------------------------------------------
                #  CORRECTION (Map -> Odom)
                # -------------------------------------------------
                def T_from_xyth(x,y,th):
                    c,s = np.cos(th), np.sin(th)
                    T = np.eye(3)
                    T[0:2,0:2] = [[c,-s],[s,c]]
                    T[0:2,2] = [x,y]
                    return T

                T_map_base  = T_from_xyth(self.icp_x, self.icp_y, self.icp_theta)
                T_odom_base = T_from_xyth(ekf_x, ekf_y, ekf_th)

                T_map_odom = T_map_base @ np.linalg.inv(T_odom_base)

                self.map_correction_x  = T_map_odom[0,2]
                self.map_correction_y  = T_map_odom[1,2]
                self.map_correction_th = np.arctan2(T_map_odom[1,0], T_map_odom[0,0])
        if self.current_point_cloud is not None:
            self.previous_point_cloud = self.current_point_cloud

        if hasattr(self, 'map_correction_x'):
            print("broadcast!")
            self.broadcast_tf(self.map_correction_x, self.map_correction_y, self.map_correction_th, 
                              "map", "odom", now)
        self.publish_path(self.ekf_path_pub, self.ekf_path_msg,ekf_x,ekf_y,ekf_th)    
        
    def publish_path(self, path_pub , path_msg,x, y, theta, timestamp):
        """Publish path"""
        # --- header
        pose = PoseStamped()
        pose.header.stamp = timestamp
        pose.header.frame_id = path_msg.header.frame_id
        pose.pose.position.x , pose.pose.position.y , pose.pose.position.z = float(x), float(y), 0.0

        q = quaternion_from_euler(0, 0, float(theta))
        pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w = q[0],q[1],q[2],q[3]

        path_msg.header.stamp = timestamp
        path_msg.poses.append(pose)

        if len(path_msg.poses) > 10000000: # if buffer is full
            path_msg.poses.pop(0)
        # print(f"Published path with {len(self.icp_path.poses)} poses.")
        path_pub.publish(path_msg)

    def publish_processed_pc(self, points, timestamp):
        """
        แปลง Numpy Array (2, N) หรือ (3, N) เป็น PointCloud2 แล้ว Publish
        """
        if points is None or points.shape[1] == 0:
            return

        points = points.T  #  (N, 2) 
        
        if points.shape[1] == 2:
            z_col = np.zeros((points.shape[0], 1))
            points_3d = np.hstack((points, z_col)) # ได้ (N, 3)
        else:
            points_3d = points

        header = Header()
        header.stamp = timestamp
        header.frame_id = 'base_link'
        pc_msg = pc2.create_cloud_xyz32(header, points_3d)
        self.point_cloud_pub.publish(pc_msg)

    def broadcast_tf(self, x, y, theta, head,child,timestamp):
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
    subscriber = ICPNode()
    rclpy.spin(subscriber)

    # np.save("result/ekf1.npy" , np.array([subscriber.xs,subscriber.ys]))

    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
