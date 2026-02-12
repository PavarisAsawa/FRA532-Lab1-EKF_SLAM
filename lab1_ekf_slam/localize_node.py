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

# from state_estimator.dead_reckoning import *
# from state_estimator.ekf import *
# from utils.plot import *
# from state_estimator.MotionIMU import *

from .state_estimator.dead_reckoning import *
from .state_estimator.ekf import *
from .state_estimator.MotionIMU import *

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class OdometryNode(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # Joint state sub
        self.joint_sub = self.create_subscription( JointState, '/joint_states', self.joint_callback, 10) # sensor_msgs/msg/JointState
        # IMU sub
        self.imu_sub = self.create_subscription( Imu, '/imu', self.imu_callback, 10)
        
        # --- Publishers ---
        # self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.path_pub = self.create_publisher(Path, '/ekf_path', 10) # Path Publisher
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # --- Path Data Container ---
        self.path = Path()
        self.path.header.frame_id = 'odom' # Must match RViz Fixed Frame

        # --- Wheel odom
        self.wheel_odom = DeadReckoning()
        self.wheel_imu = DiffDriveIMU()
        self.ekf = EKF()

        # --- Initial Condition
        self.joint_pos = [0,0]
        self.joint_vel = [0,0]
        self.x , self.y ,self.theta = 0,0,0 # robot state
        self.euler = [0,0,0]

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
            # for animation (GIF)
            self.animation_frames = []  # Store the frames for GIF
            self.anim = None


    def set_plot_axis(self):
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Live Odometry")
        self.ax.legend(fontsize=8,loc='lower left')
        # lim = 8.0
        # set 0
        # self.ax.set_xlim(-3.25, 18)
        # self.ax.set_ylim(-18, 3.25)
        # set 1
        self.ax.set_xlim(-5, 22.5)
        self.ax.set_ylim(-25.0, 12.5)
        # set 2
        # self.ax.set_xlim(-5, 22.5)
        # self.ax.set_ylim(-15.0, 22.5)

        self.ax.set_aspect('equal')
        self.ax.grid(which='both', linestyle='--', color='gray', alpha=0.5)

    def joint_callback(self, msg):
        '''
        name=['wheel_left_joint', 'wheel_right_joint']
        '''
        self.joint_pos, self.joint_vel = msg.position , msg.velocity

    def imu_callback(self, msg):
        quaternion = msg.orientation
        euler = euler_from_quaternion([quaternion.x,quaternion.y,quaternion.z,quaternion.w])
        self.euler = euler
        # print(euler)

    def timer_callback(self):
        # ---------------------------- Update estimator ---------------------------- #
        # print(self.euler[2])
        self.wheel_odom.predict(self.joint_pos)
        self.wheel_imu.predict(self.joint_pos, z_yaw=self.euler[2])
        self.x , self.y ,self.theta = self.ekf.predict(self.joint_pos, [0,0,self.euler[2]])
        # print(self.x , self.y)

        # ---------------------------- Set pub ------------------------------------- #        
        now = self.get_clock().now().to_msg() # Set time 
        self.publish_path(self.x, self.y, self.theta, now) # Publish path
        self.publish_odom(self.x, self.y, self.theta, now)
        # ---------------------------- Plot ---------------------------------------- #        
        # print(f"X : {self.x} || Y : {self.y}")
        # EKF



        if self.show_animation:
            self.xs.append(self.x)
            self.ys.append(self.y)
            # Wheel
            self.xa.append(self.wheel_odom.x)
            self.ya.append(self.wheel_odom.y)
            # Wheel IMU
            self.xb.append(self.wheel_imu.x)
            self.yb.append(self.wheel_imu.y)
            # self.set_plot_axis()
            self.wheel_line.set_data(self.xa, self.ya)
            self.ekf_line.set_data(self.xs, self.ys)
            self.wheel_imu_line.set_data(self.xb, self.yb)
            # plt.autoscale()
            plt.pause(0.005)
    
    def publish_path(self, x, y, theta, timestamp):
        """Publish path"""
        pose = PoseStamped()
        pose.header.stamp = timestamp
        pose.header.frame_id = 'odom'
        pose.pose.position.x , pose.pose.position.y , pose.pose.position.z = float(x), float(y), 0.0

        q = quaternion_from_euler(0, 0, float(theta))
        pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w = q[0],q[1],q[2],q[3]

        self.path.header.stamp = timestamp
        self.path.poses.append(pose)

        if len(self.path.poses) > 10000000: # if buffer is full
            self.path.poses.pop(0)

        self.path_pub.publish(self.path)

    def publish_odom(self, x, y, theta, timestamp):
        # 1. Create and Publish the Transform (TF)
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0

        q = quaternion_from_euler(0, 0, float(theta))
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

        # 2. Create and Publish the Odometry Message
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # Set the pose
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # Set the velocities
        # v = (self.wheel_odom.r / 2.0) * (self.joint_vel[1] + self.joint_vel[0])
        # omega = (self.wheel_odom.r / self.wheel_odom.L) * (self.joint_vel[1] - self.joint_vel[0])
        # odom.twist.twist.linear.x = float(v)
        # odom.twist.twist.angular.z = float(omega)

        self.odom_pub.publish(odom)
        
def main():
    rclpy.init(args=None)
    subscriber = OdometryNode()
    rclpy.spin(subscriber)

    # np.save("result/ekf1.npy" , np.array([subscriber.xs,subscriber.ys]))

    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
