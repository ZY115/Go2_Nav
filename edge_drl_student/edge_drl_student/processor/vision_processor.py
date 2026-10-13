#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

"""
Vision Processor Module
Processes point cloud data to vision features
"""

import torch
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class VisionProcessor:
    """
    Processes LiDAR point cloud to vision features using polynomial features
    """
    
    def __init__(self, 
                 node: Node,
                 topic_name: str = '/registered_scan',
                 queue_size: int = 10,
                 device: str = 'cpu', 
                 feature_dim: int = 24,
                 print_interval: int = 30,
                 static_lidar_size: int = 1440
        ):
        """
        Initialize Vision Processor
        
        Args:
            device: torch device
            feature_dim: Vision feature dimension 
            print_interval: Interval for printing logs
        """
        self.node = node
        self.device = device
        self.feature_dim = feature_dim
        self.print_interval = print_interval
        self.static_lidar_size = static_lidar_size
        
        # Matrix for processing lidar
        self.mat_base_vec = torch.arange(0.1, self.static_lidar_size + .1, 0.1, dtype=torch.float32).to(device) * 50
        x = torch.log(self.mat_base_vec) * 0.0001
        
        # Polynomial feature matrix
        self.vis_feature_mat = torch.vander(x, N=8, increasing=True).T.to(self.device)  # (8, 1440)
        
        # Vision features
        self.curr_vision_feature = torch.zeros(feature_dim, device=device)
        self.last_vision_feature = torch.zeros(feature_dim, device=device)
        
        # Statistics
        self.frame_count = 0
        self.last_process_time = 0.0
        
        # Raw point cloud data
        self.points = None
        
        # Subscribe to point cloud topic
        self.node.create_subscription(PointCloud2, topic_name, self.pointcloud_callback, queue_size)
        
        self.node.get_logger().info(f'VisionProcessor initialized on topic: {topic_name}')
    
    def pointcloud_callback(self, msg: PointCloud2):
        """
        Callback function for point cloud subscription
        
        Args:
            msg: PointCloud2 message from LiDAR sensor
        """
        import time
        start_time = time.time()
        
        try:
            # Parse point cloud message
            self.points = self.parse_pointcloud(msg)
            
            if self.points is not None:
                # Process to vision features
                features = self.process_pointcloud(self.points)
                
                # Update statistics
                self.frame_count += 1
                self.last_process_time = time.time() - start_time
                
                # Log info periodically 
                if self.frame_count % self.print_interval == 0:
                    self.node.get_logger().info(
                        f'[VisionProcessor] Processed frame {self.frame_count} | '
                        f'Points: {len(self.points)} | '
                        f'Process time: {self.last_process_time*1000:.2f}ms | '
                        f'Feature norm: {torch.norm(features).item():.4f}'
                    )
            else:
                self.node.get_logger().warn(f'Failed to parse point cloud at frame {self.frame_count}')
                
        except Exception as e:
            self.node.get_logger().error(f'Error in pointcloud callback: {str(e)}')
            
            
    def parse_pointcloud(self, msg):
        """
        Parse PointCloud2 message to extract XYZ coordinates
        
        Args:
            msg: PointCloud2 message
            
        Returns:
            np.ndarray: Point cloud array (N, 3)
        """
        try:
            points_list = []
            
            # Read point cloud data (x, y, z fields)
            for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False):
                clean_point = np.nan_to_num([point[0], point[1], point[2]], 
                                           nan=0.0, posinf=0.0, neginf=0.0)
                points_list.append(clean_point.tolist())
            
            return np.array(points_list)
            
        except Exception as e:
            self.node.get_logger().error(f'Failed to parse point cloud data: {str(e)}')
            return None        
            
    def process_pointcloud(self, points):
        """
        Process point cloud to vision features
        
        Args:
            points: Point cloud array (N, 3)
            
        Returns:
            torch.Tensor: Vision features (feature_dim,)
        """
        if points is None:
            return torch.zeros(self.feature_dim, device=self.device)
        
        points_tensor = torch.tensor(points, dtype=torch.float32).to(self.device)
        
        # Ensure vis_feature_mat is the right shape for matrix multiplication
        pcd_dim = points_tensor.shape[0]
        pcd_base_vec = torch.arange(0.1, pcd_dim + 0.1, 1, dtype=torch.float32).to(self.device) * 50
        pcd_x = torch.log(pcd_base_vec) * 0.0001
        self.vis_feature_mat = torch.vander(pcd_x, N=8, increasing=True).T.to(self.device)
        
        res = self.vis_feature_mat @ points_tensor
        res *= 0.01
        
        # Update vision features
        self.last_vision_feature = self.curr_vision_feature.clone()  # Last frame
        self.curr_vision_feature = res.flatten()  # Current frame
        return self.curr_vision_feature

    def get_current_features(self):
        """
        Get current vision features
        
        Returns:
            torch.Tensor: Current vision features
        """
        return self.curr_vision_feature
    
    def get_last_features(self):
        """
        Get last vision features
        
        Returns:
            torch.Tensor: Last vision features
        """
        return self.last_vision_feature
    
    def get_feature_diff(self):
        """
        Get difference between current and last features
        
        Returns:
            torch.Tensor: Feature difference
        """
        return self.curr_vision_feature - self.last_vision_feature
    
    
if __name__ == "__main__":
    # Example usage
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = rclpy.create_node('vision_processor_node')
    vision_processor = VisionProcessor(node, device='cpu', feature_dim=24)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()