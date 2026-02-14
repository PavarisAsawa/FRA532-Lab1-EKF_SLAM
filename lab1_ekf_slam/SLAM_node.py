#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import tf2_ros
import numpy as np

class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        # 1. Static TF for LiDAR
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.make_static_transform(0.029, 0.0, 0.192, 0.0, 'base_link', 'base_scan')

        # 2. Receive pose from SLAM and send Path to Rviz
        # By default, slam_toolbox sends values to /pose (type PoseWithCovarianceStamped)
        self.slam_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.slam_pose_callback, 10)
        
        # Publisher for showing the path (Path) in Rviz
        self.slam_path_pub = self.create_publisher(Path, '/slam_path', 10)
        self.slam_path = Path()
        self.slam_path.header.frame_id = 'map' # SLAM references map always

        self.get_logger().info('SLAM Path Node Started')

    def make_static_transform(self, x, y, z, yaw, parent_frame, child_frame):
        t = TransformStamped()
        # Use current time of Sim/Bag
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.z = np.sin(yaw / 2.0)
        t.transform.rotation.w = np.cos(yaw / 2.0)
        self.tf_static_broadcaster.sendTransform(t)

    def slam_pose_callback(self, msg):
        """ Receive pose from SLAM and send Path to Rviz """
        # Create PoseStamped for inserting into Path
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose

        # Add position to Path list
        self.slam_path.poses.append(pose_stamped)
        self.slam_path.header.stamp = msg.header.stamp

        # Send Path
        self.slam_path_pub.publish(self.slam_path)
        
        # Log x, y to check
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.get_logger().info(f'SLAM Pose: x={x:.3f}, y={y:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()