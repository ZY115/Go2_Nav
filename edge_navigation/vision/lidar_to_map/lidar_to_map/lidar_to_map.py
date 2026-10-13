#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
import numpy as np
import struct

import tf2_ros
from scipy.spatial.transform import Rotation as R

class LidarToLegodom(Node):
    def __init__(self):
        super().__init__("lidar_to_legodom")

        self.declare_parameter('target_frame', 'leg_odom')
        self.declare_parameter('input_topic', '/scan/points')
        self.declare_parameter('output_topic', '/registered_scan')
        self.declare_parameter('timeout', 0.1)  # TF Search Timeout in seconds
        
        self.target_frame = self.get_parameter('target_frame').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.tf_timeout = self.get_parameter('timeout').value

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointCloud2, 
            input_topic, 
            self.callback, 
            10
        )

        self.pub = self.create_publisher(
            PointCloud2, 
            output_topic, 
            10
        )

        self.success_count = 0
        self.fail_count = 0
        self.last_log_time = self.get_clock().now()

        self.get_logger().info(
            f'The lidar_to_map node is running: {input_topic} → {output_topic} '
            f'(Target Frame: {self.target_frame})'
        )

    def parse_gazebo_cloud(self, msg):
        """Parse Gazebo's PointCloud2 into an Nx3 xyz numpy array"""
        step = msg.point_step
        data = msg.data
        N = msg.width

        if N == 0:
            return np.array([]).reshape(0, 3)

        pts = np.zeros((N, 3), dtype=np.float32)

        # XYZ offsets (Assume XYZRGB or XYZ format, adjust if different)
        ox, oy, oz = 0, 4, 8

        for i in range(N):
            base = i * step
            pts[i, 0] = struct.unpack_from('f', data, base + ox)[0]
            pts[i, 1] = struct.unpack_from('f', data, base + oy)[0]
            pts[i, 2] = struct.unpack_from('f', data, base + oz)[0]

        return pts

    def transform_to_matrix(self, transform):
        """Turn TransformStamped to 4x4 Homogeneous Matrix"""
        t = transform.transform.translation
        q = transform.transform.rotation
        
        # Use scipy's Rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        rot_matrix = r.as_matrix()
        
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rot_matrix
        T[:3, 3] = [t.x, t.y, t.z]
        
        return T

    def callback(self, msg):
        # 1. Check for empty cloud
        if msg.width == 0 or len(msg.data) == 0:
            self.get_logger().warn("Received empty point cloud, skipping", throttle_duration_sec=5.0)
            return

        # 2. Parse the incoming PointCloud2 message
        try:
            pts = self.parse_gazebo_cloud(msg)
            if pts.shape[0] == 0:
                return
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"Failed to parse point cloud: {e}")
            return

        try:
            # Use the timestamp from the point cloud message for TF lookup
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,          # Target frame
                msg.header.frame_id,        # Source frame from the point cloud message
                msg.header.stamp,           # Use the timestamp from the point cloud message
                timeout=Duration(seconds=self.tf_timeout)
            )
        except tf2_ros.LookupException as e:
            self.fail_count += 1
            self.get_logger().warn(
                f"TF lookup failed: {e}",
                throttle_duration_sec=2.0
            )
            return
        except tf2_ros.ExtrapolationException as e:
            self.fail_count += 1
            self.get_logger().warn(
                f"TF extrapolation failed: {e}",
                throttle_duration_sec=2.0
            )
            return
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"TF lookup exception: {e}")
            return

        # 4. Calculate transformation matrix
        try:
            T = self.transform_to_matrix(transform)
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"Failed to calculate transformation matrix: {e}")
            return

        # 5. Apply transformation
        try:
            # Convert to homogeneous coordinates (N, 4)
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
            
            # Transform: (4, 4) @ (N, 4).T = (4, N) -> (N, 4) -> Get the first 3 columns
            pts_transformed = (T @ pts_h.T).T[:, :3]

            # pts_transformed[:, 0] = -pts_transformed[:, 0]  # X-axis flip for leg_odom coordinate system
            
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"Failed to transform point cloud: {e}")
            return

        # 6. Create PointCloud2 message
        try:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            # Create header, note that frame_id should be the target frame
            header = msg.header
            header.frame_id = 'map'

            cloud_msg = create_cloud(header, fields, pts_transformed.tolist())
            
            # Publish the transformed point cloud
            self.pub.publish(cloud_msg)
            
            self.success_count += 1
            
            # Output statistics every 5 seconds
            now = self.get_clock().now()
            if (now - self.last_log_time).nanoseconds > 5e9:  # 5 seconds
                self.get_logger().info(
                    f'The statistics - Success: {self.success_count}, '
                    f'Failed: {self.fail_count}, '
                    f'Current point count: {pts_transformed.shape[0]}'
                )
                self.last_log_time = now
                
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"Failed to create point cloud message: {e}")
            return

def main():
    rclpy.init()
    node = LidarToLegodom()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Node shutdown - Total success: {node.success_count}, '
            f'Total failed: {node.fail_count}'
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()