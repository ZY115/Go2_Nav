#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
import cv2
import yaml
import torch
import rclpy
import random
import numpy as np
from typing import Optional

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSHistoryPolicy

from edge_perception.visualization.visualizer import MaskVisualizer
from edge_perception.segmentation.panoptic_segment.algorithm import ModelGenerator


class PanopticSegmentNode(Node):
    def __init__(self):
        super().__init__('panoptic_segment_node')

        # ---------------- Load Model Config ----------------
        self.declare_parameter('config_file', '')
        self.declare_parameter('model_name', '')
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('overlay_topic', '/panoptic_segment/overlay')
        self.declare_parameter('mask_topic', '/panoptic_segment/mask')

        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        model_name = self.get_parameter('model_name').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        overlay_topic = self.get_parameter('overlay_topic').get_parameter_value().string_value
        mask_topic = self.get_parameter('mask_topic').get_parameter_value().string_value

        if not os.path.isfile(config_file):
            self.get_logger().error(f"Config file {config_file} does not exist")
            raise FileNotFoundError(f"Config file {config_file} does not exist")

        with open(config_file, 'r') as f:
            all_models = yaml.safe_load(f)

        if model_name not in all_models:
            self.get_logger().error(f"Model {model_name} not found in config file")
            raise KeyError(f"Model {model_name} not found in config file")

        self.model_cfg: dict = all_models[model_name]
        
        # Resolve model path
        from ament_index_python.packages import get_package_share_directory
        pkg_dir = get_package_share_directory('edge_perception')
        if 'model_path' in self.model_cfg:
            self.model_cfg['model_path'] = os.path.join(pkg_dir, self.model_cfg['model_path'])
            print(f"Resolved model path: {self.model_cfg['model_path']}")

        # ---------------- Device ----------------
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')

        # ---------------- Visualizer ----------------        
        self.visualizer = MaskVisualizer()

        # ---------------- Select Algorithm ----------------
        self.model_generator = ModelGenerator(**self.model_cfg, device=self.device)
        
        # ---------------- ROS2 ----------------
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(Image, image_topic, self.image_callback, qos)
        
        self.mask_pub = self.create_publisher(Image, mask_topic, 10)
        self.overlay_pub = self.create_publisher(Image, overlay_topic, 10)

        random.seed(42)
        self.get_logger().info(f'Subscribed to {image_topic}, publishing to {overlay_topic} and {mask_topic}')

    # ------------------------------------------------
    def image_callback(self, msg: Image):
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # ---------------- Model Inference ----------------
        try:
            mask_list = self.model_generator.model.infer(image_rgb)
        except Exception as e:
            self.get_logger().error(f'Model inference failed: {e}')
            return

        overlay = image_rgb.copy()
        np.random.seed(42)

        # Create combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        mask_id = 1

        # Overlay masks
        for m in mask_list:
            mask: np.ndarray = m.mask  # bool [H, W]

            # Add to combined mask
            combined_mask[mask] = mask_id
            mask_id += 1

            color = np.random.randint(0, 255, 3, dtype=np.uint8)

            # Semi-transparent fill
            self.visualizer.apply_mask(overlay, mask, color, inplace=True)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # Publish mask
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)
        except Exception as e:
            self.get_logger().warn(f'Publish mask failed: {e}')

        # Publish overlay
        try:
            out_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding='bgr8')
            out_msg.header = msg.header
            self.overlay_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f'Publish failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PanopticSegmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
