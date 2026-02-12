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
from state_estimator.icp2 import *
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
        self.icp_path_pub = self.create_publisher(Path, '/robot_path', 10) # Path Publisher for ICP
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)  # PointCloud2 Publisher
        self.tf_broadcaster = TransformBroadcaster(self)
        # Create publisher for map (OccupancyGrid)
        # self.map_pub = self.create_publisher(OccupancyGrid, '/map', 300)
        # --- Timer ---
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.pc_timer = self.create_timer(0.5, self.timer_pub_pc)
        # --- Buffer ---
        self.joint_pos, self.joint_vel = [0,0] , [0,0]
        self.euler = [0,0,0]
        self.laserscan = np.array([])

        # --- Estimator & SLAM --- 
        self.ekf = EKF()
        self.ekf_pose = np.array([0,0,0])
        self.prev_ekf_pose = np.array([0,0,0])
        self.icp_map_base = np.array([0.0, 0.0, 0.0])  # pose of base_link in map from ICP accumulation
        
        # --- Position ---
        self.icp_pose= np.array([0.0, 0.0, 0.0])
        self.prev_ekf_pose = np.array([0.0, 0.0, 0.0]) # [x, y, theta]
        # --- Store point cloud
        self.current_scan = None
        self.current_point_cloud = None  # Store the point cloud
        self.local_map_point_cloud = np.array([])
        self.odom_point = None

        # --- Path
        self.icp_path_msg = Path()
        self.icp_path_msg.header.frame_id = 'odom' # Must match RViz Fixed Frame
        self.ekf_path_msg = Path()
        self.ekf_path_msg.header.frame_id = 'odom' # Path นี้วาดบน Odom

    def laser_scan_callback(self, msg):
        # -------------------------------- 
        # Prepocessing Point cloud
        # --------------------------------
        point_cloud = scan_to_pointcloud(msg) # shape [2,n]
        # if point_cloud.shape[0] < 50: return
        # clear_outlier = remove_outliers(point_cloud,0.2 , 5) # Outlier treatment
        # voxel_grid = voxel_grid_filter(clear_outlier, leaf_size=0.02) # down sampling with voxel filter,        
        self.current_scan =  point_cloud #voxel_grid
        if point_cloud.shape[0] < 25: return
        # Transform frame
        self.current_point_cloud = pointcloud_to_odom(self.current_scan ,self.icp_pose[0],self.icp_pose[1],self.icp_pose[2])
        # -------------------------------- 
        # Apply EKF with 5 Hz
        # --------------------------------   

        if self.prev_ekf_pose is None:
            self.prev_ekf_pose = self.ekf_pose
        delta_ekf = self.ekf_pose - self.prev_ekf_pose
        delta_ekf[2] = wrap(delta_ekf[2])
        self.icp_pose += delta_ekf
        
        self.prev_ekf_pose = self.ekf_pose

        now = self.get_clock().now().to_msg() # Set time 
        self.broadcast_tf(self.icp_pose[0], self.icp_pose[1], self.icp_pose[2],"odom", "base_link", now)

        # --- ICP 
        if (self.local_map_point_cloud.shape[0] > 100) and self.current_point_cloud is not None:
            x,y,theta,count,error,success = icp_matching(previous_points=self.local_map_point_cloud, 
                                            current_points=self.current_point_cloud, 
                                            # init_x=self.icp_pose[0], 
                                            # init_y=self.icp_pose[1],
                                            # init_theta=self.icp_pose[2],
                                            init_x=delta_ekf[0],
                                            init_y=delta_ekf[1],
                                            init_theta=delta_ekf[2],
                                            MAX_ITERATION=30,
                                            )
            print(x , y , theta)
            # print("get it" , self.local_map_point_cloud)
            # if success:
            #    print("success")
            #     corr_t = np.sqrt((x - self.icp_pose[0])**2 + (y-self.icp_pose[1])**2)
            #     corr_theta = abs(np.arctan2(np.sin(theta - self.icp_pose[2]) , np.cos(theta - self.icp_pose[2])))
            #     print(f"cor T {corr_t} , cor theta {corr_theta}")
            #     if corr_t < 0.01 and corr_theta < np.deg2rad(10):
            #         print("HIIIIIIIIII")
            #         self.icp_pose += np.array([x,y,theta])
            #     else:
            #         print("booooooooooooooooooo")
            #     self.icp_pose = np.array([x,y,theta])

            
            # -------------------------------- 
            # Construct map
            # --------------------------------   
        if self.local_map_point_cloud.shape[0] == 0: self.local_map_point_cloud = self.current_point_cloud
        self.local_map_point_cloud = np.vstack((self.local_map_point_cloud , self.current_point_cloud))
        self.local_map_point_cloud = remove_outliers(self.local_map_point_cloud, 0.2 , 8)
        self.local_map_point_cloud = voxel_grid_filter(self.local_map_point_cloud , leaf_size=0.05)
        print(self.local_map_point_cloud.shape)
        self.publish_processed_pc(self.local_map_point_cloud  ,now,"odom")
        self.publish_path(self.icp_path_pub , self.icp_path_msg, self.icp_pose[0],self.icp_pose[1],self.icp_pose[2],now)
            # test = "a"
            # self.local_map_point_cloud


    def joint_callback(self, msg):
        '''name=['wheel_left_joint', 'wheel_right_joint']'''
        self.joint_pos, self.joint_vel = msg.position , msg.velocity

    def imu_callback(self, msg):
        self.euler = euler_from_quaternion([msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w])

    def timer_callback(self):
        # --- Set time        
        now = self.get_clock().now().to_msg() # Set time 

        self.ekf_pose = self.ekf.predict(self.joint_pos, [0,0,self.euler[2]])
        self.ekf_pose[2] = wrap(self.ekf_pose[2])     

    def timer_pub_pc(self):
        if self.odom_point is not None:
            # self.publish_processed_pc(self.odom_point, timestamp=self.get_clock().now().to_msg())
            pass

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

    def publish_processed_pc(self, points, timestamp,frame_id):
        if points is None or points.shape[0] == 0:
            return        
        if points.shape[1] == 2:
            z_col = np.zeros((points.shape[0], 1))
            points_3d = np.hstack((points, z_col)) # ได้ (N, 3)
        else:
            points_3d = points

        header = Header()
        header.stamp = timestamp
        header.frame_id = frame_id
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
