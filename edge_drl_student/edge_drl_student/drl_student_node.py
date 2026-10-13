#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

"""
DRL Student Node
"""

import os
import rclpy
import torch
from rclpy.node import Node

from edge_msgs.msg import EdgeState, EdgeAction
from edge_drl_student.common.logger import Logger
from edge_drl_student.utils.utils import ExperimentRunType
from edge_drl_student.common.reset_manager import ResetManager
from edge_drl_student.common.episode_handler import EpisodeManager
from edge_drl_student.processor.vision_processor import VisionProcessor
from edge_drl_student.runners.distributed_runner import DistributedRunner
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

torch.set_printoptions(sci_mode=False)

class DRLStudentNode(Node):

    @staticmethod
    def _find_latest_snapshot_dir(root_dir: str, required_files: tuple[str, ...]) -> str | None:
        """Find the newest directory containing all required files."""
        if not os.path.isdir(root_dir):
            return None

        candidates = []
        for dirpath, _, filenames in os.walk(root_dir):
            if all(required in filenames for required in required_files):
                candidates.append(dirpath)

        if not candidates:
            return None

        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    def _resume_previous_real_robot_training(self) -> None:
        """Resume from latest checkpoint and replay buffer when running on real robot."""
        checkpoints_root = os.path.join(CURRENT_DIR, "outputs", "checkpoints")
        buffers_root = os.path.join(CURRENT_DIR, "outputs", "buffers")

        latest_checkpoint = self._find_latest_snapshot_dir(
            checkpoints_root,
            ("actor.pt", "critic.pt", "target_actor.pt", "target_critic.pt"),
        )
        latest_buffer = self._find_latest_snapshot_dir(
            buffers_root,
            ("stu_buffer.pt", "tea_buffer.pt"),
        )

        loaded_any = False

        if latest_checkpoint is not None:
            try:
                self.runner.load_checkpoint(latest_checkpoint)
                self.get_logger().info(f"✅ Real-robot resume loaded checkpoint: {latest_checkpoint}")
                loaded_any = True
            except Exception as exc:
                self.get_logger().warning(f"⚠️ Failed to load checkpoint from {latest_checkpoint}: {exc}")
        else:
            self.get_logger().warning("⚠️ No previous checkpoint found for real-robot auto-resume")

        if latest_buffer is not None:
            try:
                self.runner.load_buffer_contents(latest_buffer)
                self.get_logger().info(f"✅ Real-robot resume loaded buffer: {latest_buffer}")
                loaded_any = True
            except Exception as exc:
                self.get_logger().warning(f"⚠️ Failed to load buffer from {latest_buffer}: {exc}")
        else:
            self.get_logger().warning("⚠️ No previous replay buffer found for real-robot auto-resume")

        if loaded_any:
            self.get_logger().info("🔁 Real-robot startup resume completed")

    def __init__(self):

        super().__init__("drl_student", namespace="edge")

        # ============================================================
        # Device
        # ============================================================
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # ============================================================
        # Parameters from launch
        # ============================================================
        self.run = self.declare_parameter("run", "cognitive").value
        self.task = self.declare_parameter("task", "navigation").value
        self.is_real_robot = self.declare_parameter("is_real_robot", False).value
        self.project_name = self.declare_parameter("project_name", "drl_student").value
        
        # Path for logs, models, and buffers
        self.log_path = os.path.join(CURRENT_DIR, "outputs/logs", self.project_name)
        self.model_path = os.path.join(CURRENT_DIR, "outputs/checkpoints", self.project_name)
        self.buffer_path = os.path.join(CURRENT_DIR, "outputs/buffers", self.project_name)

        # ============================================================
        # Control parameters
        # ============================================================
        self.declare_parameters(
            namespace="",
            parameters=[
                ("MATRIX_P", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("IDENTITY_MAT", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("SAFETY_ENVELOPE_P", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("RESIDUAL_GAIN", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("DEFAULT_KP", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("DEFAULT_KD", rclpy.Parameter.Type.DOUBLE_ARRAY),
            ]
        )

        RESIDUAL_GAIN = self.get_parameter("RESIDUAL_GAIN").value

        # ============================================================
        # Task configuration
        # ============================================================
        task_config = {
            "navigation": {
                "act_dim": 3,
                "edge_state_type": EdgeState.ROBOT_STATE,
                "edge_action_type": EdgeAction.CMD_VEL,
                "action_bound":(
                    torch.tensor([0.35, -0.1, -0.4], device=self.device),
                    torch.tensor([0.7, 0.1, 0.1], device=self.device))
            },
            "locomotion": {
                "act_dim": 6,
                "edge_state_type": EdgeState.TRACKING_ERROR,
                "edge_action_type": EdgeAction.BASE_ACCEL,
                "action_bound":(
                    torch.tensor([-10., -10., -3., -15., -15., -15.], device=self.device),
                    torch.tensor([10., 10., 3., 15., 15., 15.], device=self.device)
                )
            },
            "end2end": {
                "act_dim": 6,
                "edge_state_type": EdgeState.ROBOT_STATE,
                "edge_action_type": EdgeAction.BASE_ACCEL,
                "action_bound":(
                    torch.tensor([-10., -10., -3., -15., -15., -15.], device=self.device),
                    torch.tensor([10., 10., 3., 15., 15., 15.], device=self.device)
                )
            },
        }

        if self.task not in task_config:
            raise ValueError(f"Unsupported task: {self.task}")


        self.run_type = ExperimentRunType[self.run.upper()]      # Convert to enum for better type readability
        self.get_logger().info(f"🚀 Starting DRL Student Node with RUN={self.run_type}")
        cfg = task_config[self.task]

        self.act_dim = cfg["act_dim"]
        self.edge_state_type = cfg["edge_state_type"]
        self.edge_action_type = cfg["edge_action_type"]
        self.action_bound = cfg["action_bound"]

        self.get_logger().info(f"🧠 Task: {self.task}")

        # ============================================================
        # Dimensions
        # ============================================================
        self.body_state_dim = 12
        self.vision_feature_dim = 24
        
        if self.task == "locomotion":
            self.obs_dim = self.body_state_dim
        elif self.task in {"navigation", "end2end"}:
            self.obs_dim = self.body_state_dim + self.vision_feature_dim

        self.get_logger().info(
            f"📐 Dimensions: obs_dim={self.obs_dim}, act_dim={self.act_dim}"
        )

        # ============================================================
        # Logger
        # ============================================================
        self.logger = Logger(log_dir=self.log_path, device=self.device)

        self.get_logger().info(f"📊 Logger initialized: {self.log_path}")
        self.get_logger().info(f"📦 Model path: {self.model_path}")
        self.get_logger().info(f"💾 Buffer path: {self.buffer_path}")

        # ============================================================
        # Replay Buffer
        # ============================================================
        if self.run_type in {ExperimentRunType.RLM, ExperimentRunType.SHIELDING}:
            self.batch_size = 512
            buf_size = int(2e5)
            prefill_size = self.batch_size
            self.buffer = SoloReplayBuffer(
                buffer_size=buf_size,
                prefill_size=prefill_size,
                obs_dim=self.obs_dim,
                action_dim=self.act_dim,
                name="Buffer",
                device=self.device,
            )
            
        elif self.run_type in {ExperimentRunType.DUAL, ExperimentRunType.COGNITIVE}:
            
            self.batch_size = 512
            stu_buf_size = int(1e5) 
            tea_buf_size = int(1e5)
            teacher_prefill_size = 200
            
            # self.batch_size = 512
            # stu_buf_size = int(2e5) 
            # tea_buf_size = int(2e5)
            # teacher_prefill_size = 200

            stu_blending_weight = 0.0
            tea_blending_weight = 1.0
            
            if self.run_type == ExperimentRunType.DUAL:
                self.buffer = DualReplayBuffer(
                    student_buf_size=stu_buf_size,
                    student_prefill_size=self.batch_size,
                    teacher_buf_size=tea_buf_size,
                    teacher_prefill_size=teacher_prefill_size,
                    obs_dim=self.obs_dim,
                    action_dim=self.act_dim,
                    device=self.device,
                )
                
            else:
                self.buffer = CognitiveReplayBuffer(
                student_buf_size=stu_buf_size,
                student_prefill_size=self.batch_size,
                teacher_buf_size=tea_buf_size,
                teacher_prefill_size=teacher_prefill_size,
                stu_blending_weight=stu_blending_weight,
                tea_blending_weight=tea_blending_weight,
                obs_dim=self.obs_dim,
                action_dim=self.act_dim,
                device=self.device,
            )
                
        else:
            raise ValueError(f"Unsupported RUN type: {self.run_type}")

        self.get_logger().info(f"✅ Replay buffer initialized for {self.run_type}")

        # ============================================================
        # Vision Processor
        # ============================================================
        self.vision_processor = VisionProcessor(
            node=self,
            topic_name="/registered_scan",
            queue_size=10,
            device=self.device,
            feature_dim=self.vision_feature_dim,
        )

        # ============================================================
        # Episode Manager
        # ============================================================
        eval_interval = 500000

        self.episode_manager = EpisodeManager(
            logger=self.logger,
            # max_episode_length=int(3e5),
            max_episode_length=int(1e5),
            training_episode=10000,
            eval_during_train=True,
            eval_interval=eval_interval,
            moving_average_window=100,
            is_real_robot=self.is_real_robot,
        )

        # ============================================================
        # Reset Manager
        # ============================================================
        self.reset_manager = ResetManager(
            node=self,
            reset_timeout=15.0,
            logger=self.logger,
            reset_service_name="/edge/reset",
            gz_reset_service_name="/edge/reset/gz_handler",
        )

        # ============================================================
        # ROS2 Topics
        # ============================================================
        ros2_topic_name = {
            "transition_topic": "/edge/transition",
            "drl_action_topic": "/edge/drl_student_action",
            "edge_state_topic": "/edge/robot_state",
            "drl_learned_action_topic": "/edge/drl_student_learned_action",
            "fall_detection_topic": "/edge/is_fall",
        }

        ros2_service_name = {
            "manual_reset_service": "/edge/manual_reset",
        }

        # ============================================================
        # Training configuration
        # ============================================================
        training_cfg = {
            "log_interval": 100,
            "sync_interval": 5,
            "normalize_obs": True,
            "normalize_reward": True,
            "auto_reset_on_fall": True,
            "batch_size": self.batch_size,
            "save_interval": 500,
            "model_save_path": self.model_path,
            "buffer_save_path": self.buffer_path,
        }

        # ============================================================
        # Distributed Runner
        # ============================================================
        self.runner = DistributedRunner(
            node=self,
            logger=self.logger,
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            state_type=self.edge_state_type,
            action_type=self.edge_action_type,
            run_type=self.run_type,
            
            body_state_dim=self.body_state_dim,
            reset_manager=self.reset_manager,
            episode_manager=self.episode_manager,
            vision_processor=self.vision_processor,
            replay_buffer=self.buffer,
            action_bound=self.action_bound,
            is_real_robot=self.is_real_robot,
            # residual_gain=torch.tensor(RESIDUAL_GAIN, device=self.device).reshape(
            #     self.act_dim, self.body_state_dim
            # ),
            
            **ros2_topic_name,
            **ros2_service_name,
            **training_cfg,
        )

        self.get_logger().info("✅ Distributed Runner initialized")

        if self.is_real_robot:
            self._resume_previous_real_robot_training()

        # ============================================================
        # Start training loop
        # ============================================================
        self.runner.start()
        self.get_logger().info("🚀 DRL Student Node is up and running!")

    # ============================================================
    # Shutdown
    # ============================================================
    def destroy_node(self):

        self.get_logger().info("🛑 Shutting down DRL Student Node...")

        self.runner.shutdown()
        self.runner.save_final_checkpoint()

        super().destroy_node()


# ============================================================
# Main
# ============================================================
def main(args=None):

    print("\n🚀 Starting DRL Student Node\n")

    rclpy.init(args=args)

    node = DRLStudentNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("⌨️ Keyboard interrupt received")

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print("\n✅ DRL Student Node shutdown complete\n")


if __name__ == "__main__":
    main()