#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import struct
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import message_filters
import tf2_ros
from tf2_ros import TransformException
import onnxruntime as ort


class SemanticSegmentNode(Node):
    def __init__(self) -> None:
        super().__init__('semantic_segment_node')
        self.declare_parameter('model_path', '')
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('pointcloud_topic', '/points_raw')
        self.declare_parameter('mask_topic', '/semantic_mask')
        self.declare_parameter('colored_cloud_topic', '/semantic_cloud')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 360)
        self.declare_parameter('publish_colored_cloud', True)
        self.declare_parameter('sync_slop', 0.08)
        self.declare_parameter('tf_timeout', 0.15)

        self.model_path: str = self.get_parameter('model_path').get_parameter_value().string_value
        self.publish_colored_cloud: bool = self.get_parameter('publish_colored_cloud').get_parameter_value().bool_value
        self.input_width: int = self.get_parameter('input_width').get_parameter_value().integer_value
        self.input_height: int = self.get_parameter('input_height').get_parameter_value().integer_value
        self.sync_slop: float = self.get_parameter('sync_slop').get_parameter_value().double_value
        self.tf_timeout: float = self.get_parameter('tf_timeout').get_parameter_value().double_value

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        pointcloud_topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value
        mask_topic = self.get_parameter('mask_topic').get_parameter_value().string_value
        colored_cloud_topic = self.get_parameter('colored_cloud_topic').get_parameter_value().string_value

        # QoS for image/camera_info: use RELIABLE to match Gazebo bridge
        qos_reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)
        # QoS for point cloud: use BEST_EFFORT for high-frequency sensor data
        qos_sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)

        self.mask_pub = self.create_publisher(Image, mask_topic, 10)
        self.colored_cloud_pub = self.create_publisher(PointCloud2, colored_cloud_topic, 5)
        self.overlay_pub = self.create_publisher(Image, '/semantic_overlay', 10)

        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.session: Optional['ort.InferenceSession'] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self._init_session()

        self.palette = self._build_palette()

        # Subscribe to image only for testing
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_reliable
        )
        
        # message_filters synchronizer - use RELIABLE for image/camera_info
        # image_sub = message_filters.Subscriber(self, Image, image_topic, qos_profile=qos_reliable)
        # info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic, qos_profile=qos_reliable)
        # cloud_sub = message_filters.Subscriber(self, PointCloud2, pointcloud_topic, qos_profile=qos_sensor)
        # self.sync = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub, cloud_sub],
        #                                                         queue_size=5,
        #                                                         slop=self.sync_slop)
        # self.sync.registerCallback(self._sync_callback)
        
        self.get_logger().info(f'Subscribed to image topic: {image_topic}')
        self.get_logger().info(f'Waiting for synchronized messages from:')
        self.get_logger().info(f'  Image: {image_topic}')
        self.get_logger().info(f'  CameraInfo: {camera_info_topic}')
        self.get_logger().info(f'  PointCloud: {pointcloud_topic}')
        self.get_logger().info(f'Sync slop: {self.sync_slop}s')

    def _init_session(self) -> None:
        if self.model_path == '':
            self.get_logger().error('model_path is empty; set a valid ONNX model path.')
            return
        if ort is None:
            self.get_logger().error('onnxruntime is not installed. Install onnxruntime-gpu on Jetson.')
            return
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.get_logger().info(f'Loaded ONNX model: {self.model_path}')
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f'Failed to load model {self.model_path}: {exc}')
            self.session = None

    @staticmethod
    def _build_palette() -> np.ndarray:
        colors = [
            (128, 128, 128), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),  # 背景改成灰色
            (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35),
            (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
            (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
        ]
        return np.array(colors, dtype=np.uint8)

    def _preprocess(self, cv_image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(cv_image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # CHW
        tensor = np.expand_dims(tensor, axis=0)    # NCHW
        return tensor

    def _infer_mask(self, image_msg: Image) -> Optional[np.ndarray]:
        if self.session is None:
            return None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'cv_bridge failed: {exc}')
            return None
        
        # 保存原图用于叠加显示
        self.last_cv_image = cv_image.copy()
        
        input_tensor = self._preprocess(cv_image)
        try:
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        except Exception as exc:
            self.get_logger().warn(f'Inference failed: {exc}')
            return None
        logits = outputs[0]  # (1, C, H, W)
        mask_small = np.argmax(logits, axis=1).astype(np.uint8)[0]
        mask = cv2.resize(mask_small, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _publish_mask(self, mask: np.ndarray, header) -> None:
        # Debug: 打印mask的统计信息
        unique_labels = np.unique(mask)
        self.get_logger().info(f'Mask: unique labels={unique_labels}, shape={mask.shape}', throttle_duration_sec=3.0)
        
        color_mask = self.palette[np.clip(mask, 0, self.palette.shape[0]-1)]
        
        # 发布纯mask
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(color_mask, encoding='rgb8')
            mask_msg.header = header
            self.mask_pub.publish(mask_msg)
        except Exception as exc:
            self.get_logger().warn(f'Publish mask failed: {exc}')
        
        # 发布叠加图像（mask半透明叠加在原图上）
        if hasattr(self, 'last_cv_image'):
            overlay = cv2.addWeighted(self.last_cv_image, 0.6, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR), 0.4, 0)
            try:
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                overlay_msg.header = header
                self.overlay_pub.publish(overlay_msg)
            except Exception as exc:
                self.get_logger().warn(f'Publish overlay failed: {exc}')

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        # standard quaternion to rotation matrix
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        rot = np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),     2.0 * (xz + wy)],
            [2.0 * (xy + wz),     1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),     2.0 * (yz + wx),     1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float32)
        return rot

    def _project_and_color(self, mask: np.ndarray, info_msg: CameraInfo, cloud_msg: PointCloud2) -> None:
        if not self.publish_colored_cloud:
            return
        # camera model from info
        self.cam_model.fromCameraInfo(info_msg)

        cam_frame = info_msg.header.frame_id
        cloud_frame = cloud_msg.header.frame_id
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                cam_frame,
                cloud_frame,
                rclpy.time.Time.from_msg(cloud_msg.header.stamp),
                timeout=Duration(seconds=self.tf_timeout)
            )
        except TransformException as exc:
            self.get_logger().warn_throttle(2.0, f'TF lookup failed {cloud_frame}->{cam_frame}: {exc}')
            return

        trans = tf_msg.transform.translation
        rot = tf_msg.transform.rotation
        R = self._quat_to_rot(rot.x, rot.y, rot.z, rot.w)
        t = np.array([trans.x, trans.y, trans.z], dtype=np.float32)

        points = np.array(list(pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)), dtype=np.float32)
        if points.size == 0:
            return
        cam_points = (R @ points.T).T + t

        h, w = mask.shape
        colors: List[float] = []
        valid_points: List[List[float]] = []

        for idx in range(cam_points.shape[0]):
            x, y, z = cam_points[idx]
            if z <= 0.0:
                continue
            uv = self.cam_model.project3dToPixel((x, y, z))
            u, v = int(uv[0] + 0.5), int(uv[1] + 0.5)
            if u < 0 or v < 0 or u >= w or v >= h:
                continue
            label = int(mask[v, u])
            color = self.palette[label % self.palette.shape[0]]
            packed = struct.unpack('f', struct.pack('I', (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])))[0]
            valid_points.append(points[idx].tolist())
            colors.append(packed)

        if not valid_points:
            return

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_points = [valid_points[i] + [colors[i]] for i in range(len(valid_points))]
        colored_cloud = pc2.create_cloud(cloud_msg.header, fields, cloud_points)
        self.colored_cloud_pub.publish(colored_cloud)

    def _image_callback(self, image_msg: Image) -> None:
        self.get_logger().info('Received image, processing...', throttle_duration_sec=2.0)
        mask = self._infer_mask(image_msg)
        if mask is None:
            self.get_logger().warn('Inference returned None', throttle_duration_sec=2.0)
            return
        self.get_logger().info(f'Inference successful, mask shape: {mask.shape}', throttle_duration_sec=2.0)
        self._publish_mask(mask, image_msg.header)
        
    def _sync_callback(self, image_msg: Image, info_msg: CameraInfo, cloud_msg: PointCloud2) -> None:
        self.get_logger().info('Received synchronized messages, processing...', throttle_duration_sec=2.0)
        mask = self._infer_mask(image_msg)
        if mask is None:
            self.get_logger().warn('Inference returned None', throttle_duration_sec=2.0)
            return
        self.get_logger().info(f'Inference successful, mask shape: {mask.shape}', throttle_duration_sec=2.0)
        self._publish_mask(mask, image_msg.header)
        self._project_and_color(mask, info_msg, cloud_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticSegmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
