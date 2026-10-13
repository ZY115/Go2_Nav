#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
import time

import torch
import numpy as np
from rclpy.node import Node
from dataclasses import dataclass
from rclpy.qos import QoSProfile, ReliabilityPolicy
from typing import List, Union, Tuple, Dict, Any, Optional

from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from edge_msgs.msg import Transition, Vec12Msg, EdgeAction, EdgeReward, EdgeState

from edge_drl_student.common.noise import OUNoise
from edge_drl_student.common.logger import Logger
from edge_drl_student.common.reset_manager import ResetManager
from edge_drl_student.common.normalizer import EmpiricalNormalizer 
from edge_drl_student.common.episode_handler import EpisodeManager
from edge_drl_student.processor.vision_processor import VisionProcessor
from edge_drl_student.common.normalizer import RunningMeanStdNormalizer
from edge_drl_student.agents.distributed.ddpg_actor_learner import DDPGActorLearner
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer
from edge_drl_student.utils.utils import ActionMode, ResetReason, DRLMode, ExperimentRunType

RESIDUAL_GAIN = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]) * -0.1


class DistributedRunner:
    """
    Distributed Runner for interacting with the environment and learning
    """
    
    def __init__(
        self,
        node: Node,
        logger: Logger,
        obs_dim: int,
        act_dim: int,
        body_state_dim: int,
        state_type: int,
        action_type: int,
        run_type: ExperimentRunType,
        
        replay_buffer: Union[SoloReplayBuffer, DualReplayBuffer, CognitiveReplayBuffer],
        action_bound: Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]],
        reset_manager: Optional[ResetManager],
        episode_manager: Optional[EpisodeManager],
        vision_processor: Optional[VisionProcessor],

        is_real_robot: bool = False,
        residual_gain: torch.Tensor = RESIDUAL_GAIN,
        transition_topic: str = 'transition',
        drl_action_topic: str = 'drl_student_action',
        edge_state_topic: str = 'edge_state_stu',
        drl_learned_action_topic: str = 'drl_student_learned_action',
        fall_detection_topic: str = '/edge/is_fall',
        manual_reset_service: str = '/edge/manual_reset',

        # batch_size: int = 256,
        batch_size: int = 512,
        
        log_interval: int = 100,
        sync_interval: int = 5,
        save_interval: int = 500,
        
        policy_blending_weight: float = 0.3,
        action_frequency: float = 10.0,   # Hz
        learning_frequency: float = 2.0,  # Hz
        
        normalize_obs: bool = True,
        normalize_reward: bool = True,

        auto_reset_on_fall: bool = True,
        model_save_path: str = 'distributed_drl_model',
        buffer_save_path: str = 'distributed_drl_buffer',
        
        device: str = 'cpu',
        **agent_kwargs
    ):
        
        self.node = node
        self.logger = logger
        
        # Store dimensions
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.body_state_dim = body_state_dim
        self.run_type = run_type
        self.device = device
        
        self.batch_size = batch_size
        self.action_frequency = action_frequency
        self.learning_frequency = learning_frequency
        self.policy_blending_weight = policy_blending_weight
        self.residual_gain = residual_gain.to(device)
        
        self.node.get_logger().info(f"Residual Gain Matrix:\n{self.residual_gain}")
        
        # Training settings
        self.log_interval = log_interval
        self.sync_interval = sync_interval
        self.save_interval = save_interval
        
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        
        self.model_save_path = model_save_path
        self.buffer_save_path = buffer_save_path
        
        # Create Normalizers
        if normalize_obs:
            self.obs_normalizer = RunningMeanStdNormalizer(
                shape=(obs_dim,),
                device=device
            )
            self.node.get_logger().info(
                f"✅ Observation normalizer created (RunningMeanStd) with obs_dim={obs_dim}"
            )
        else:
            self.obs_normalizer = None
            self.node.get_logger().info("⏸️  Observation normalization disabled")

        if normalize_reward:
            self.reward_normalizer = RunningMeanStdNormalizer(
                shape=(1,), 
                device=device
            )
            self.node.get_logger().info(
                f"✅ Reward normalizer created (RunningMeanStd) with reward_dim={1}"
            )
        else:
            self.reward_normalizer = None
            self.node.get_logger().info("⏸️  Reward normalization disabled")

        
        # ⭐ Create DDPGActorLearner as a component
        self.actor_learner = DDPGActorLearner(
            obs_dim=obs_dim,
            act_dim=act_dim,
            replay_buffer=replay_buffer,
            action_bound=action_bound,
            sync_interval=sync_interval,
            model_save_path=model_save_path,
            device=device,
            **agent_kwargs
        )
        
        # Store components
        self.replay_buffer = replay_buffer
        self.vision_processor = vision_processor
        self.episode_manager = episode_manager
        self.reset_manager = reset_manager
        self.auto_reset_on_fall = auto_reset_on_fall
        self.is_real_robot = is_real_robot
        self.save_flag = False
        self.is_real_robot_ready = False
        
        # ============================================================
        # Edge State/Action Initialization
        # ============================================================
        self.state_type = state_type
        self.action_type = action_type

        if state_type == EdgeState.ROBOT_STATE:
            self.state_input: List[float] = [0.0] * self.body_state_dim
            self.node.get_logger().info("🧠 Using ROBOT_STATE as EdgeState")

        elif state_type == EdgeState.TRACKING_ERROR:
            self.state_input: List[float] = [0.0] * self.body_state_dim
            self.node.get_logger().info("🧠 Using TRACKING_ERROR as EdgeState")

        else:
            raise ValueError(f"Unsupported state_type: {state_type}")
        
        # Fall detection state
        self.fall_count: int = 0
        self.last_fall_state: bool = False
        self.fall_reset_time: float = 0.0
        
        # Statistics
        self.total_actor_loss = 0.0
        self.total_critic_loss = 0.0
        self.transition_count = 0
        self.action_count = 0
        self.action_time = 0.0
        self.learn_time = 0.0
        
        # ⭐ Episode reward accumulator (for reward normalization)
        self.episode_rewards = []
        
        # ========== ROS2 Publishers ==========
        self.drl_action_pub = self.node.create_publisher(EdgeAction, drl_action_topic, 10)
        self.drl_learned_action_pub = self.node.create_publisher(EdgeAction, drl_learned_action_topic, 10)
        
        # ========== ROS2 Subscribers ==========
        transition_qos = QoSProfile(depth=10)
        transition_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.node.create_subscription(Transition, transition_topic, self.transition_callback, transition_qos)
        self.node.create_subscription(EdgeState, edge_state_topic, self.edge_state_callback, 10)
        self.node.create_subscription(Bool, fall_detection_topic, self.fall_detection_callback, 10)
        self.node.create_subscription(Bool, "/edge/is_real_robot_ready", self.real_robot_ready_callback, 10)
        
        # ========== ROS2 Services ==========
        self.node.create_service(Trigger, manual_reset_service, self.manual_reset_callback)
        self.node.get_logger().info(
            f"✅ Manual reset service created: {manual_reset_service}\n"
            f"   Usage: ros2 service call {manual_reset_service} std_srvs/srv/Trigger"
        )
        
        # ========== ROS2 Timers (not started yet) ==========
        self.action_timer = None
        self.learning_timer = None
        
        # Log initialization
        self._log_initialization()
    
    
    def start(self):
        """Start acting and learning loops"""
        if self.action_timer is not None or self.learning_timer is not None:
            self.node.get_logger().info("⚠️ Loops already started")
            return
        
        self.action_timer = self.node.create_timer(1.0 / self.action_frequency, self.action_loop)
        self.learning_timer = self.node.create_timer(1.0 / self.learning_frequency, self.learning_loop)
        self.is_action_running = True
        self.is_learning_running = True
        
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"🚀 Loops started\n"
            f"{'='*70}\n"
            f"   Action Loop:   {self.action_frequency}Hz\n"
            f"   Learning Loop: {self.learning_frequency}Hz\n"
            f"{'='*70}"
        )
    
    def pause(self):
        """Pause acting and learning loops"""
        self.pause_action_loop()
        self.pause_learning_loop()
        
    def resume(self):
        """Resume acting and learning loops"""
        self.resume_action_loop()
        self.resume_learning_loop()
    
    def shutdown(self):
        """Shutdown acting and learning loops"""
        if self.action_timer is not None:
            self.action_timer.cancel()
            self.action_timer = None
        
        if self.learning_timer is not None:
            self.learning_timer.cancel()
            self.learning_timer = None
        
        self.node.get_logger().info("🛑 Loops shutdown")
    
    # ==================== ROS2 Callbacks ====================
    def edge_state_callback(self, msg: EdgeState):
        """Edge State callback"""
        
        if msg.state_type == EdgeState.ROBOT_STATE:
            self.state_input: List[float] = list(msg.state_value)
            # print(f"Received ROBOT_STATE: {self.state_input}")
            self.node.get_logger().debug(f"Received ROBOT_STATE: {self.state_input}")
            
        elif msg.state_type == EdgeState.TRACKING_ERROR:
            self.state_input: List[float] = list(msg.state_value)
            
            tracking_error_data = list(msg.state_value)
            if len(tracking_error_data) > 0:
                tracking_error_data[0] = 0.0
            if len(tracking_error_data) > 1:
                tracking_error_data[1] = 0.0

            # self.tracking_error_ = Vec12Msg(data=tracking_error_data)

            tracking_errors = torch.tensor(tracking_error_data, dtype=torch.float32)
            self.logger.writer.add_scalar('Control/tracking_error/mean', torch.mean(torch.abs(tracking_errors)), self.transition_count)
            self.logger.writer.add_scalar('Control/tracking_error/max', torch.max(torch.abs(tracking_errors)), self.transition_count)
            self.logger.writer.add_scalar('Control/tracking_error/std', torch.std(tracking_errors), self.transition_count)
            
            self.node.get_logger().debug(f"Received TRACKING_ERROR: {tracking_error_data}")
        else:
            self.node.get_logger().error(f"Unknown EdgeState type: {msg.state_type}")

    
    def manual_reset_callback(self, request: Trigger.Request, response: Trigger.Response):
        """
        Manual reset service callback
        
        Usage:
            ros2 service call /edge/manual_reset std_srvs/srv/Trigger
        """
        self.node.get_logger().info('🔄 Manual reset triggered by service call')
        
        if self.reset_manager.is_resetting:
            response.success = False
            response.message = 'Reset already in progress, please wait'
            return response
        
        ep_mode = self.episode_manager.episode_mode
        
        # Handle episode done with MANUAL reason 
        self._handle_episode_done(success=False, reason=ResetReason.MANUAL, ep_mode=ep_mode)
        
        response.success = True
        response.message = (
            f'Manual reset successful. '
            f'Episode #{self.episode_manager.episode_count_train}.'
        )
        return response


    def real_robot_ready_callback(self, msg: Bool):
        """Callback for real robot readiness signal"""
        is_real_robot_ready = msg.data
        self.is_real_robot_ready = is_real_robot_ready
    
    def fall_detection_callback(self, msg: Bool) -> None:
        """Fall detection callback with rising edge detection"""
        is_fall: bool = msg.data
        
        # Detect rising edge (False → True)
        if is_fall and not self.last_fall_state:
            self.fall_count += 1
            
            self.node.get_logger().info(
                f"\n{'='*70}\n"
                f"⚠️  FALL DETECTED #{self.fall_count}\n"
                f"{'='*70}"
            )
            
            # self.logger.writer.add_scalar('Fall/total_count', self.fall_count, self.transition_count)

            if self.auto_reset_on_fall:
                self.node.get_logger().info('🔄 Auto reset triggered by fall detection')
                
                ep_mode = self.episode_manager.episode_mode
                
                # ⭐ Handle episode done with FALL_DETECTED reason
                self._handle_episode_done(success=False, reason=ResetReason.FALL_DETECTED, ep_mode=ep_mode)
                
                # Update fall state
                self.last_fall_state = False
                self.fall_reset_time = time.time()
            else:
                self.node.get_logger().info('⏸️  Auto reset on fall is disabled')
                self.last_fall_state = is_fall
        
        elif not is_fall and self.last_fall_state:
            self.node.get_logger().info("✅ Robot recovered from fall state")
            self.last_fall_state = False
        
        else:
            self.last_fall_state = is_fall

    
    # ==================== Transition Processing ====================
    def _process_transition(self, msg: Transition, stepwise_total_reward: float) -> bool:
        """
        Process transition: create observations, normalize, and store in buffer
        
        Args:
            msg: Transition message from environment
            stepwise_total_reward: Total reward for this step
            
        Returns:
            success: Whether processing and storing transition was successful
        """
        # Create observation (body_state + vision_feature)
        observation = torch.cat((
            torch.tensor(
                np.array(msg.obs, dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ),
            self.vision_processor.get_last_features()
        ), dim=0)
        
        next_observation = torch.cat((
            torch.tensor(
                np.array(msg.next_obs, dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ),
            self.vision_processor.get_current_features()
        ), dim=0)
        # observation = torch.tensor(
        #     np.array(msg.obs, dtype=np.float32),
        #     dtype=torch.float32,
        #     device=self.device
        # )
        
        # next_observation = torch.tensor(
        #     np.array(msg.next_obs, dtype=np.float32),
        #     dtype=torch.float32,
        #     device=self.device
        # )
        
        # Normalize observations
        if self.obs_normalizer is not None:
            # Update normalizer statistics
            self.obs_normalizer.update(observation)
            self.obs_normalizer.update(next_observation)
            
            # Normalize observations
            obs_norm = self.obs_normalizer.normalize(observation)
            next_obs_norm = self.obs_normalizer.normalize(next_observation)
        else:
            obs_norm = observation
            next_obs_norm = next_observation
        
        # Create transition
        t = self.replay_buffer.to_transition(
            obs=obs_norm,
            action=msg.action.action_value,
            reward=stepwise_total_reward,
            next_obs=next_obs_norm,
            done=msg.done
        )
        action_mode = ActionMode(msg.action.action_mode)
        
        # Model Predictive Shielding (MPS) logic
        if self.run_type == ExperimentRunType.SHIELDING:
            if action_mode == ActionMode.STUDENT:
                self.replay_buffer.add_transition(t, action_mode=ActionMode.STUDENT)
                self.node.get_logger().info(
                    f"[Transition] Received student transition. "
                    f"Reward: {stepwise_total_reward:.3f}, "
                    f"Done: {msg.done}"
                )
            else:
                self.node.get_logger().warning(
                    f"Received non-student transition in SHIELDING run, ignoring."
                )
                return False
            
        # Runtime Learning Machine (RLM) logic
        elif self.run_type == ExperimentRunType.RLM:
            if action_mode in [ActionMode.STUDENT, ActionMode.TEACHER]:
                self.replay_buffer.add_transition(t, action_mode=action_mode)
                self.node.get_logger().info(
                    f"[Transition] Received {action_mode.name.lower()} transition. "
                    f"Reward: {stepwise_total_reward:.3f}, "
                    f"Done: {msg.done}"
                )
            else:
                self.node.get_logger().warning(
                    f"Received uncertain transition in RLM run, ignoring."
                )
                return False
        
        # Dual and Cognitive logic (store in respective buffers)
        elif self.run_type in [ExperimentRunType.COGNITIVE, ExperimentRunType.DUAL]:
        
            # Add transition to the appropriate buffer
            if action_mode == ActionMode.STUDENT:
                self.replay_buffer.add_transition(t, action_mode=ActionMode.STUDENT)
                if not self.replay_buffer.stu_buf.is_ready:
                    self.node.get_logger().info(
                        f"[Transition] Added to student buffer. "
                        f"Size: {self.replay_buffer.stu_buf_size}"
                    )
            elif action_mode == ActionMode.TEACHER:
                self.replay_buffer.add_transition(t, action_mode=ActionMode.TEACHER)
                if not self.replay_buffer.tea_buf.is_ready:
                    self.node.get_logger().info(
                        f"[Transition] Added to teacher buffer. "
                        f"Size: {self.replay_buffer.tea_buf_size}"
                    )
            else:
                self.node.get_logger().error(f'Unknown action mode: {msg.action.action_mode}')
                return False
            
        else:
            self.node.get_logger().error(f"Unsupported run type: {self.run_type}")
            return False
        
        return True
    
    # ==================== Transition Callback ====================
    def transition_callback(self, msg: Transition):
        """Transition callback - collect experiences with normalization. 
           ...
           Also update episode statistics (interaction-level, not learning)
        """
        
        # Reset handling
        if self.reset_manager.is_resetting:
            self.node.get_logger().info('⏸️  Skipping transition during reset')
            
            if self.is_learning_running:
                self.node.get_logger().info('⏸️  Learning loop paused during reset')
                self.pause_learning_loop()
            
            if self.is_action_running:
                self.node.get_logger().info('⏸️  Action loop paused during reset')
                self.pause_action_loop()
            
            return
        else:
            if self.episode_manager.episode_mode == DRLMode.EVAL:
                self.node.get_logger().info('⏸️  Skipping transition during evaluation episode')
                self.pause_learning_loop()
                
            else:
                if not self.is_learning_running:
                    self.node.get_logger().info('▶️ Resuming learning loop after reset')
                    self.resume_learning_loop()
                
                if not self.is_action_running:
                    self.node.get_logger().info('▶️ Resuming action loop after reset')
                    self.resume_action_loop()
        
        # Sanity check for action type
        if msg.action.action_type != self.action_type:
            self.node.get_logger().error(
                f"Received action type {msg.action.action_type} does not match expected {self.action_type}"
            )
            return
        
        # Handle special case for SHIELDING run: ignore teacher transitions for reward calculation
        if self.run_type == ExperimentRunType.SHIELDING:
            if msg.action.action_mode == ActionMode.TEACHER:
                self.node.get_logger().warning(
                    f"Received teacher transition in SHIELDING run, ignoring for reward calculation."
                )
                if self.is_learning_running:
                    self.node.get_logger().info('⏸️  Learning loop paused during teacher transition in SHIELDING run')
                    self.pause_learning_loop()
                return
            elif msg.action.action_mode == ActionMode.STUDENT:
                if not self.is_learning_running:
                    self.node.get_logger().info('▶️ Resuming learning loop after teacher transition in SHIELDING run')
                    self.resume_learning_loop()
                self.node.get_logger().info(
                    f"[Transition] Received student transition in SHIELDING run. "
                )

        # Extract reward components
        rew_comp: List[EdgeReward] = msg.reward_components   
        reward_components = {c.name: c.value for c in rew_comp}
                
        # Calculate stepwise total reward
        stepwise_total_reward = sum(reward_components.values())
        
        # Accumulate episode rewards (for reward normalization)
        self.episode_rewards.append(stepwise_total_reward)
        
        # Determine ep_mode and action_mode
        ep_mode = self.episode_manager.episode_mode
        action_mode = ActionMode(msg.action.action_mode)
        
        # Process transition and store in buffer (if in training mode) 
        if ep_mode == DRLMode.TRAIN:
            is_success = self._process_transition(msg, stepwise_total_reward)
            if is_success:
                self.transition_count += 1
            # print(f"Processed transition #{self.transition_count}")
            
        # Update episode stats (training mode)
        self.episode_manager.update(reward_components, action_mode, ep_mode) 
        
        # Log stepwise statistics
        self._log_stepwise_stats(reward_components, ep_mode)
        
        # Check episode done
        if msg.done or self.episode_manager.is_episode_done:
            reason = (
                ResetReason.TIMEOUT if self.episode_manager.is_episode_done
                else ResetReason.TASK_DONE
            )
            success = True if msg.done and reason == ResetReason.TASK_DONE else False
            self._handle_episode_done(success=success, reason=reason, ep_mode=ep_mode)

    # ==================== Action Loop ====================
    def compute_output_action(self, mode: DRLMode) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute output action with residual control and observation normalization
        Args:
            mode: DRLMode enum value ('train' or 'eval')
        
        Returns:
            drl_student_action: Final action to send to robot [act_dim]
            learned_action: Pure learned action [act_dim]
        """
        
        # Determine if exploration noise should be added
        explore = (mode == DRLMode.TRAIN) and not isinstance(self.replay_buffer, CognitiveReplayBuffer)
        
        # Get current observation
        observation = torch.cat((
            torch.tensor(
                np.array(self.state_input, dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ),
            self.vision_processor.curr_vision_feature
        ), dim=0)
        # observation = torch.tensor(
        #         np.array(self.state_input, dtype=np.float32),
        #         dtype=torch.float32,
        #         device=self.device
        # )
        
        # Normalize observation
        if self.obs_normalizer is not None:
            obs_norm = self.obs_normalizer.normalize(observation)
        else:
            obs_norm = observation
        
        # Get policy action from actor_learner's acting_agent (using normalized obs)
        policy_action = self.actor_learner.acting_agent.act(
            obs=obs_norm.unsqueeze(0),
            explore=explore
        ).squeeze(0)

        # policy_action[2] *= -1          
        # Compute residual action
        # tracking_error = torch.tensor(
        #     self.tracking_error_.data,
        #     dtype=torch.float32,
        #     device=self.devicef
        # )
        # residual_action = (self.residual_gain @ tracking_error.unsqueeze(-1)).flatten()
    

        # Blend with teacher action (for cognitive buffer)
        if isinstance(self.replay_buffer, CognitiveReplayBuffer):
            blended_action = self.replay_buffer.get_weighted_blended_action()
            learned_action = 0.5 * policy_action + 0.5 * blended_action
            # learned_action = 0.0 * policy_action + 0.8 * blended_action
            # learned_action = 1.0 * policy_action + 0.0 * blended_action
            
            # print(f"blended_action: {blended_action}")
            # print(f"policy_action: {policy_action}")

            # a1 = 0.5 * policy_action
            # a2 = 0.2 * blended_action

            # print("aaaaaaaaaaaa",a1)
            # print("bbbbbbbbbbbb",a2)

        else:
            learned_action = policy_action
        
        # learned_action[2] -= 0.03

        # Final output action
        drl_student_action = learned_action
        
        return drl_student_action, learned_action

    
    def action_loop(self):
        """Action generation loop (high frequency)"""
        
        if not self.is_action_running:
            self.node.get_logger().info('⏸️  Action loop is paused')
            return
        
        run_mode = DRLMode.TRAIN
        
        if self.episode_manager.episode_mode == DRLMode.EVAL:
            self.node.get_logger().info('ActionLoop: Evaluation episode - switching to EVAL mode')
            run_mode = DRLMode.EVAL
        
        action_start = time.time()
        
        # Compute actions
        drl_student_action, learned_action = self.compute_output_action(mode=run_mode)
        
        # Publish drl-student action
        edge_drl_action_msg = EdgeAction()
        edge_drl_action_msg.header.stamp = self.node.get_clock().now().to_msg()
        edge_drl_action_msg.action_value = drl_student_action.detach().cpu().numpy().flatten().tolist()
        edge_drl_action_msg.action_type = self.action_type
        edge_drl_action_msg.action_mode = ActionMode.STUDENT.value
        self.drl_action_pub.publish(edge_drl_action_msg)
        
        # Publish learned action
        edge_policy_action_msg = EdgeAction()
        edge_policy_action_msg.header.stamp = self.node.get_clock().now().to_msg()
        edge_policy_action_msg.action_value = learned_action.detach().cpu().numpy().flatten().tolist()
        edge_policy_action_msg.action_type = self.action_type
        edge_policy_action_msg.action_mode = ActionMode.STUDENT.value
        self.drl_learned_action_pub.publish(edge_policy_action_msg)
        
        self.action_count += 1
        self.action_time = time.time() - action_start
        
        # Log
        self.logger.writer.add_scalar('Perf/action_generation_ms', self.action_time * 1000, self.action_count)
        
        # Warn if slow
        # expected_time = 1.0 / self.action_frequency
        # if self.action_time > expected_time * 1.5:
        #     self.node.get_logger().info(
        #         f"⚠️ Slow action generation: {self.action_time*1000:.2f}ms "
        #         f"(expected <{expected_time*1000:.2f}ms)"
        #     )
    
    # ==================== Learning Loop ====================
    def learning_loop(self):
        """Learning loop (low frequency)"""
        if not self.is_learning_running:
            self.node.get_logger().info('⏸️  Learning loop is paused')
            return
        
        if not self.is_real_robot_ready:
            self.node.get_logger().info('⏸️  Waiting for real robot to be ready before learning')
            return
        
        if self.save_flag:
            self.node.get_logger().info("⏸️  Save flag is set, skipping learning to save checkpoint")
            return
        
        # Log buffer sizes
        if self.run_type in {ExperimentRunType.SHIELDING, ExperimentRunType.RLM}:
            self.logger.writer.add_scalar('Learn/buffer_size/solo_buf', self.replay_buffer.buffer_size, self.actor_learner.update_count)
        
        else:
            self.logger.writer.add_scalar('Learn/buffer_size/student', self.replay_buffer.stu_buf_size, self.actor_learner.update_count)
            self.logger.writer.add_scalar('Learn/buffer_size/teacher', self.replay_buffer.tea_buf_size, self.actor_learner.update_count)

        # Check if buffers are ready
        if self.run_type in {ExperimentRunType.SHIELDING, ExperimentRunType.RLM}:
            if not self.replay_buffer.is_ready:
                self.node.get_logger().info(
                    f"Replay buffer not ready. "
                    f"Size: {self.replay_buffer.size}/{self.replay_buffer.prefill_size}"
                )
                return
        
        else:
            if not self.replay_buffer.stu_buf.is_ready:
                self.node.get_logger().info(
                    f"Student Buffer not ready. "
                    f"Size: {self.replay_buffer.stu_buf_size}/{self.replay_buffer.stu_buf.prefill_size}"
                )
                return
            
            if not self.replay_buffer.tea_buf.is_ready:
                self.node.get_logger().info(
                    f"Teacher Buffer not ready. "
                    f"Size: {self.replay_buffer.tea_buf_size}/{self.replay_buffer.tea_buf.prefill_size}"
                )
                return

        learn_start = time.time()
        
        # ⭐ Update policy using actor_learner (auto sync)
        losses: Dict[str, float] = self.actor_learner.update(batch_size=self.batch_size)
        
        # Update statistics
        self.total_actor_loss += losses['actor_loss']
        self.total_critic_loss += losses['critic_loss']
        self.learn_time = time.time() - learn_start
        
        # Log metrics
        self._log_learning_metrics(losses)
        
        # Save models periodically
        if self.actor_learner.update_count % self.save_interval == 0:
            self.save_checkpoint(model_name=f"update_{self.actor_learner.update_count}")
    
    # ==================== Buffer Management ====================
    def save_buffer_contents(self, save_path: str):
        """Save replay buffer contents to disk"""
        target_path = save_path if os.path.isabs(save_path) else f"{self.buffer_save_path}/{save_path}"
        self.replay_buffer.dump(target_path)
        self.node.get_logger().info(f"💾 Replay buffers saved to {target_path}")
    
    def load_buffer_contents(self, load_path: str):
        """Load replay buffer contents from disk"""
        target_path = load_path if os.path.isabs(load_path) else f"{self.buffer_save_path}/{load_path}"
        self.replay_buffer.load(target_path)
        self.node.get_logger().info(f"✅ Replay buffers loaded from {target_path}")
    
    # ==================== Model Management ====================
    def save_checkpoint(self, model_name: str = ''):
        """Save model checkpoint including normalizers"""
        
        save_path = f"{self.model_save_path}/{model_name}"
        
        # Save actor-learner models
        self.actor_learner.save_models(save_path)
        
        # ⭐ Save normalizers
        if self.obs_normalizer is not None:
            torch.save(
                self.obs_normalizer.state_dict(),
                os.path.join(save_path, "obs_normalizer.pt")
            )
        
        if self.reward_normalizer is not None:
            torch.save(
                self.reward_normalizer.state_dict(),
                os.path.join(save_path, "reward_normalizer.pt")
            )
        
        self.node.get_logger().info(f"💾 Models and normalizers saved to {save_path}")
    
    
    def save_final_checkpoint(self):
        """Save final model checkpoint"""
        import os
        save_path = f"{self.model_save_path}/update_count_final"
        os.makedirs(save_path, exist_ok=True)
        
        # Save actor-learner models
        self.actor_learner.save_models(save_path)
        
        # Save normalizers
        if self.obs_normalizer is not None and self.normalize_obs:
            torch.save(
                self.obs_normalizer.state_dict(),
                os.path.join(save_path, "obs_normalizer.pt")
            )
        
        if self.reward_normalizer is not None and self.normalize_reward:
            torch.save(
                self.reward_normalizer.state_dict(),
                os.path.join(save_path, "reward_normalizer.pt")
            )
        
        self.node.get_logger().info(f"💾 Final models and normalizers saved to {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """⭐ Load model checkpoint including normalizers"""
        import os
        
        # Load actor-learner models
        self.actor_learner.load_models(load_path)
        
        # Load normalizers
        obs_norm_path = os.path.join(load_path, "obs_normalizer.pt")
        if self.obs_normalizer is not None and os.path.exists(obs_norm_path):
            self.obs_normalizer.load_state_dict(torch.load(obs_norm_path))
            self.node.get_logger().info("✅ Observation normalizer loaded")
        
        reward_norm_path = os.path.join(load_path, "reward_normalizer.pt")
        if self.reward_normalizer is not None and os.path.exists(reward_norm_path):
            self.reward_normalizer.load_state_dict(torch.load(reward_norm_path))
            self.node.get_logger().info("✅ Reward normalizer loaded")
        
        self.node.get_logger().info(f"✅ Models and normalizers loaded from {load_path}")
    
    # ==================== Episode Management ====================
    def _handle_episode_done(self, success: Optional[bool],reason: ResetReason = ResetReason.TASK_DONE, ep_mode: DRLMode = DRLMode.TRAIN):
        """
        Handle end of episode: log stats, update normalizers, reset environment
        
        Args:
            reason: Reset reason (TASK_DONE, FALL_DETECTED, TIMEOUT, etc.)
        """
        
        if ep_mode == DRLMode.EVAL:
            self.node.get_logger().info("Evaluation episode ended, saving checkpoint...")
            self.save_checkpoint(model_name=f'eval_episode_{self.episode_manager.episode_count_train}')
        
        # Log episode end and get stats
        stats = self.episode_manager.log_episode_end(success=success, reason=reason, ep_mode=ep_mode)
        self.episode_manager.reset_episode()    # Reset episode state
        
        # Update reward normalizer
        if self.reward_normalizer is not None and self.normalize_reward:
            if len(self.episode_rewards) > 0:
                episode_rewards_tensor = torch.tensor(
                    self.episode_rewards,
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Update normalizer with episode rewards
                for reward in episode_rewards_tensor:
                    self.reward_normalizer.update(reward.unsqueeze(0))
                
                # Log normalization stats
                reward_mean = self.reward_normalizer.get_stats()['mean']
                reward_std = self.reward_normalizer.get_stats()['std']
                
                self.logger.writer.add_scalar('Learn/reward_norm/mean', reward_mean, self.episode_manager.episode_count_train)
                self.logger.writer.add_scalar('Learn/reward_norm/std', reward_std, self.episode_manager.episode_count_train)
                
                # self.node.get_logger().info(
                #     f"📊 Reward normalization updated: "
                #     f"mean={reward_mean:.4f}, std={reward_std:.4f}"
                # )
        
        # Clear episode rewards
        self.episode_rewards.clear()
        
        if hasattr(self, 'discounted_return'):
            self.discounted_return = 0.0
        
        # ========== Step 4: Log Episode Summary ==========
        # self.node.get_logger().info(
        #     f"\n{'='*70}\n"
        #     f"📊 Episode {self.episode_manager.episode_count_train} ended\n"
        #     f"{'─'*70}\n"
        #     f"   Reason:           {reason.value}\n"
        #     f"   Total Reward:     {stats['reward']:.2f}\n"
        #     f"   Episode Length:   {stats['length']}\n"
        #     f"   Activation Ratio: {stats['activation_ratio']:.2%}\n"
        #     f"{'='*70}"
        # )
        
        if self.is_real_robot:
            if self.save_flag:
                self.node.get_logger().info("Already saved checkpoint for this episode, skipping...")
            
            else:
                self.save_flag = True
                self.replay_buffer.dump(f"{self.buffer_save_path}/episode_{self.episode_manager.episode_count_train}")
                self.save_checkpoint(model_name=f"episode_{self.episode_manager.episode_count_train}")
                self.node.get_logger().info(
                    f"\n{'='*70}\n"
                    f"📊 Episode {self.episode_manager.episode_count_train} ended\n"
                    f"{'─'*70}\n"
                    f"   Reason:           {reason.value}\n"
                    f"   Total Reward:     {stats.episode_return:.2f}\n"
                    f"   Episode Length:   {stats.episode_length}\n"
                    f"   Activation Ratio: {stats.activation_ratio:.2%}\n"
                    f"{'='*70}"
            )
            return # Skip reset for real robot to allow manual intervention
        
        # Call reset service
        self.node.get_logger().info(f'🔄 Triggering reset (reason: {reason.value})')
        success, message = self.reset_manager.call_reset_service(reason)
        
        if success:
            self.node.get_logger().info(f'✅ Reset completed: {message}')
        else:
            self.node.get_logger().error(f'❌ Reset failed: {message}')

    
    # ==================== Statistics ====================
    @property
    def agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        if self.actor_learner.update_count == 0:
            return {
                'avg_actor_loss': 0.0,
                'avg_critic_loss': 0.0,
                'update_count': 0,
                'transition_count': self.transition_count,
                'action_count': self.action_count,
            }
        
        return {
            'avg_actor_loss': self.total_actor_loss / self.actor_learner.update_count,
            'avg_critic_loss': self.total_critic_loss / self.actor_learner.update_count,
            'update_count': self.actor_learner.update_count,
            'transition_count': self.transition_count,
            'action_count': self.action_count,
        }
    
    @property
    def performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'action_time_ms': self.action_time * 1000,
            'learn_time_ms': self.learn_time * 1000,
            'action_frequency': self.action_frequency,
            'learning_frequency': self.learning_frequency,
        }
    
    @property
    def network_info(self) -> Dict[str, Any]:
        """Get network information"""
        def count_parameters(model: torch.nn.Module) -> int:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'actor_params': count_parameters(self.actor_learner.learning_agent.actor),
            'critic_params': count_parameters(self.actor_learner.learning_agent.critic),
            'target_actor_params': count_parameters(self.actor_learner.learning_agent.target_actor),
            'target_critic_params': count_parameters(self.actor_learner.learning_agent.target_critic),
            'total_params': (
                count_parameters(self.actor_learner.learning_agent.actor) +
                count_parameters(self.actor_learner.learning_agent.critic) +
                count_parameters(self.actor_learner.learning_agent.target_actor) +
                count_parameters(self.actor_learner.learning_agent.target_critic)
            ),
        }
    
    @property
    def fall_stats(self) -> Dict[str, Any]:
        """Get fall detection statistics"""
        return {
            'fall_count': self.fall_count,
            'last_fall_state': self.last_fall_state,
            'auto_reset_on_fall': self.auto_reset_on_fall,
        }
    
    @property
    def reset_stats(self) -> Dict[str, Any]:
        """Get reset statistics"""
        return self.reset_manager.get_statistics()
    
    @property
    def sync_stats(self) -> Dict[str, Any]:
        """Get sync statistics"""
        return self.actor_learner.get_sync_stats()
    
    @property
    def normalization_stats(self) -> Dict[str, Any]:
        """
        Get current normalization statistics
        
        Returns:
            Dictionary containing normalization statistics
        """
        stats = {}
        
        # ========== Observation Normalizer ==========
        if self.obs_normalizer is not None:
            obs_stats = self.obs_normalizer.get_stats()
            
            # Extract statistics
            stats['obs_mean'] = obs_stats['mean'].mean().item()
            stats['obs_std'] = obs_stats['std'].mean().item()
            stats['obs_count'] = obs_stats['count']
            
            # Optional: Add per-dimension statistics
            stats['obs_mean_per_dim'] = obs_stats['mean'].cpu().numpy().tolist()
            stats['obs_std_per_dim'] = obs_stats['std'].cpu().numpy().tolist()
        
        # ========== Reward Normalizer ==========
        if self.reward_normalizer is not None:
            reward_stats = self.reward_normalizer.get_stats()
            
            # Extract statistics (reward is typically scalar)
            if reward_stats['mean'].numel() == 1:
                stats['reward_mean'] = reward_stats['mean'].item()
                stats['reward_std'] = reward_stats['std'].item()
            else:
                stats['reward_mean'] = reward_stats['mean'].mean().item()
                stats['reward_std'] = reward_stats['std'].mean().item()
            
            stats['reward_count'] = reward_stats['count']
        
        return stats

    
    # ==================== Logging ====================
    def _log_initialization(self):
        """Log initialization information"""
        
        self.node.get_logger().info("\n" + "="*70)
        self.node.get_logger().info("📊 DistributedRunner Initialization Summary")
        self.node.get_logger().info("="*70)
        
        # ========== Configuration ==========
        self.node.get_logger().info("\n⚙️  Configuration:")
        self.node.get_logger().info(
            f"   Device:               {self.device}\n"
            f"   Observation Dim:      {self.obs_dim}\n"
            f"   Action Dim:           {self.act_dim}\n"
            f"   Body State Dim:       {self.body_state_dim}"
        )
        
        # ========== Training Config ==========
        self.node.get_logger().info("\n🎓 Training Configuration:")
        self.node.get_logger().info(
            f"   Batch Size:           {self.batch_size}\n"
            f"   Save Interval:        {self.save_interval}\n"
            f"   Log Interval:         {self.log_interval}"
        )
        
        # ========== Actor-Learner Config ==========
        self.node.get_logger().info("\n🔄 Actor-Learner:")
        self.node.get_logger().info(
            f"   Sync Interval:        every {self.actor_learner.sync_interval} updates\n"
            f"   Only Sync Actor:      {self.actor_learner.only_sync_actor}"
        )
        
        # ========== Normalization ==========
        self.node.get_logger().info("\n🔢 Normalization:")
        self.node.get_logger().info(
            f"   Obs Normalization:    {'Enabled' if self.normalize_obs else 'Disabled'}\n"
            f"   Reward Normalization: {'Enabled' if self.normalize_reward else 'Disabled'}"
        )
        
        # Get current normalization statistics
        if self.obs_normalizer is not None:
            obs_stats = self.obs_normalizer.get_stats()
            self.node.get_logger().info(
                f"   Obs Stats:            mean={obs_stats['mean'].mean().item():.4f}, "
                f"std={obs_stats['std'].mean().item():.4f}, "
                f"count={obs_stats['count']:.0f}"
            )
        
        if self.reward_normalizer is not None:
            reward_stats = self.reward_normalizer.get_stats()
            # Handle scalar vs vector rewards
            if reward_stats['mean'].numel() == 1:
                mean_val = reward_stats['mean'].item()
                std_val = reward_stats['std'].item()
            else:
                mean_val = reward_stats['mean'].mean().item()
                std_val = reward_stats['std'].mean().item()
            
            self.node.get_logger().info(
                f"   Reward Stats:         mean={mean_val:.4f}, "
                f"std={std_val:.4f}, "
                f"count={reward_stats['count']:.0f}"
            )
        
        # ========== Agent Info ==========
        self.node.get_logger().info("\n🤖 Agent Configuration:")
        agent_info = self.actor_learner.get_agent_info()
        for key, value in agent_info.items():
            self.node.get_logger().info(f"   {key}: {value}")
        
        # ========== Buffer Info ==========
        # self.node.get_logger().info("\n💾 Replay Buffer:")
        # self.node.get_logger().info(
        #     f"   Type:                 {type(self.replay_buffer).__name__}\n"
        #     # f"   Capacity:             {self.replay_buffer.capacity}\n"
        #     f"   Current Size:         {len(self.replay_buffer)}\n"
        #     f"   Prefill Size:         {self.replay_buffer.prefill_size}\n"
        #     f"   Is Ready:             {self.replay_buffer.is_ready()}"
        # )
        
        # ========== Training Status ==========
        self.node.get_logger().info("\n📈 Training Status:")
        self.node.get_logger().info(
            f"   Episode Count:        {self.episode_manager.episode_count_train}\n"
            # f"   Total Steps:          {self.episode_manager.total_steps}\n"
            f"   Update Count:         {self.actor_learner.update_count}\n"
            f"   Sync Count:           {self.actor_learner.sync_count}"
        )
        
        self.node.get_logger().info("\n" + "="*70 + "\n")

    # def _log_solo_stepwise_stats(self, reward_components: Dict[str, float]):
    #     """Log stepwise statistics"""
    #     step = self.episode_manager.current_episode_length
    #     stepwise_total_reward = sum(reward_components.values())
        
    #     self.logger.writer.add_scalar('Stepwise/total_reward', stepwise_total_reward, step)
    #     self.logger.writer.add_scalar('Stepwise/count/student_action', self.episode_manager.stu_action_count, step)
    #     self.logger.writer.add_scalar('Stepwise/count/teacher_action', self.episode_manager.tea_action_count, step)
    #     for name, value in reward_components.items():
    #         self.logger.writer.add_scalar(f'Stepwise/reward/{name}', value, step)
    
    def _log_stepwise_stats(self, reward_components: Dict[str, float], mode=DRLMode.TRAIN):
        """Log stepwise statistics"""
        
        if mode == DRLMode.TRAIN:
            totalstep = self.episode_manager.total_steps
        elif mode == DRLMode.EVAL and self.episode_manager.eval_during_train:
            totalstep = self.episode_manager.eval_steps
        else:
            raise ValueError(f"Invalid mode for logging stepwise stats: {mode}")
        stepwise_total_reward = sum(reward_components.values())
        
        mode_str = mode.name.capitalize()
        
        self.logger.writer.add_scalar(f'{mode_str}-Stepwise/total_reward', stepwise_total_reward, totalstep)
        self.logger.writer.add_scalar(f'{mode_str}-Stepwise/count/student_action', self.episode_manager.stu_action_count, totalstep)
        self.logger.writer.add_scalar(f'{mode_str}-Stepwise/count/teacher_action', self.episode_manager.tea_action_count, totalstep)
        for name, value in reward_components.items():
            self.logger.writer.add_scalar(f'{mode_str}-Stepwise/reward/{name}', value, totalstep)
    
    
    def _log_learning_metrics(self, losses: Dict[str, float]):
        """Log learning metrics"""
        update_count = self.actor_learner.update_count
        
        # Log losses
        self.logger.writer.add_scalar('Learn/loss/actor', losses['actor_loss'], update_count)
        self.logger.writer.add_scalar('Learn/loss/critic', losses['critic_loss'], update_count)
        
        # Log learning_rate
        actor_lr = self.actor_learner.learning_agent.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.actor_learner.learning_agent.critic_optimizer.param_groups[0]['lr']
        self.logger.writer.add_scalar('Learn/learning_rate/actor', actor_lr, update_count)
        self.logger.writer.add_scalar('Learn/learning_rate/critic', critic_lr, update_count)
        
        # Log noise
        if isinstance(self.actor_learner.acting_agent.noise, OUNoise):
            sigma = self.actor_learner.acting_agent.noise.sigma
            self.logger.writer.add_scalar('Learn/noise_sigma', sigma, update_count)
        
        # Log learning performance
        self.logger.writer.add_scalar('Perf/learn_ms', self.learn_time * 1000, update_count)
        
        # Log sync count
        # self.logger.writer.add_scalar('Learn/sync_count', self.actor_learner.sync_count, update_count)
        
        # Log sync time if it happened
        if 'sync_time' in losses:
            self.logger.writer.add_scalar('Perf/parameter_sync_ms', losses['sync_time'] * 1000, self.actor_learner.sync_count)
            self.node.get_logger().info(
                f"🔄 [Sync #{self.actor_learner.sync_count}] "
                f"Parameters synchronized in {losses['sync_time']*1000:.2f}ms"
            )
        
        # Log average statistics
        if update_count % self.log_interval == 0:
            stats = self.agent_stats
            perf_stats = self.performance_stats
            sync_stats = self.sync_stats
            reset_stats = self.reset_stats
            fall_stats = self.fall_stats
            norm_stats = self.normalization_stats
            
            self.logger.writer.add_scalar('Learn/moving_avg_loss/actor', stats['avg_actor_loss'], update_count)
            self.logger.writer.add_scalar('Learn/moving_avg_loss/critic', stats['avg_critic_loss'], update_count)
            # self.logger.writer.add_scalar('Fall/total_count', fall_stats['fall_count'], update_count)
            
            # Console log
            norm_info = ""
            if 'obs_mean' in norm_stats:
                norm_info += f"   Obs Mean/Std:    {norm_stats['obs_mean']:.4f} / {norm_stats['obs_std']:.4f}\n"
            if 'reward_mean' in norm_stats:
                norm_info += f"   Reward Mean/Std: {norm_stats['reward_mean']:.4f} / {norm_stats['reward_std']:.4f}\n"
            
            self.node.get_logger().info(
                f"\n{'='*70}\n"
                f"📊 Training Statistics (Update #{update_count})\n"
                f"{'='*70}\n"
                f"📉 Losses:\n"
                f"   Actor Loss:      {losses['actor_loss']:.4f}\n"
                f"   Critic Loss:     {losses['critic_loss']:.4f}\n"
                f"   Avg Actor Loss:  {stats['avg_actor_loss']:.4f}\n"
                f"   Avg Critic Loss: {stats['avg_critic_loss']:.4f}\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"📊 Normalization:\n"
                f"{norm_info}"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"⚡ Performance:\n"
                f"   Learn Time:      {perf_stats['learn_time_ms']:.2f}ms\n"
                f"   Action Time:     {perf_stats['action_time_ms']:.2f}ms\n"
                f"   Avg Sync Time:   {sync_stats['avg_sync_time']*1000:.2f}ms\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"🔢 Counters:\n"
                f"   Updates:         {stats['update_count']}\n"
                f"   Syncs:           {self.actor_learner.sync_count}\n"
                f"   Transitions:     {stats['transition_count']}\n"
                f"   Actions:         {stats['action_count']}\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"🔄 Reset Statistics:\n"
                f"   Total Resets:    {reset_stats['total']}\n"
                f"   Success Rate:    {reset_stats['success_rate']:.1%}\n"
                f"   Task Done:       {reset_stats['reasons']['task_done']}\n"
                f"   Fall Detected:   {reset_stats['reasons']['fall_detected']}\n"
                f"   Timeout:         {reset_stats['reasons']['timeout']}\n"
                f"   External:        {reset_stats['reasons']['external']}\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"⚠️  Fall Detection:\n"
                f"   Total Falls:     {fall_stats['fall_count']}\n"
                f"   Auto Reset:      {'Enabled' if fall_stats['auto_reset_on_fall'] else 'Disabled'}\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                # f"💾 Buffer Status:\n"
                # f"   Student Size:    {len(self.replay_buffer.stu_buf)}/{self.replay_buffer.stu_buf.capacity}\n"
                # f"   Teacher Size:    {len(self.replay_buffer.tea_buf)}/{self.replay_buffer.tea_buf.capacity}\n"
                f"{'='*70}"
            )
    
    def pause_action_loop(self):
        """Pause action loop"""
        self.is_action_running = False
        self.node.get_logger().info("⏸️  Action loop paused")
        
    def resume_action_loop(self):
        """Resume action loop"""
        self.is_action_running = True
        self.node.get_logger().info("▶️  Action loop resumed")
        
    def pause_learning_loop(self):
        """Pause learning loop"""
        self.is_learning_running = False
        self.node.get_logger().info("⏸️  Learning loop paused")
        
    def resume_learning_loop(self):
        """Resume learning loop"""
        self.is_learning_running = True
        self.node.get_logger().info("▶️  Learning loop resumed")
    
    # ==================== Utility Methods ====================
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"DistributedRunner(\n"
            f"  obs_dim={self.obs_dim},\n"
            f"  act_dim={self.act_dim},\n"
            f"  body_state_dim={self.body_state_dim},\n"
            f"  vision_feature_dim={self.obs_dim - self.body_state_dim},\n"
            f"  action_frequency={self.action_frequency}Hz,\n"
            f"  learning_frequency={self.learning_frequency}Hz,\n"
            f"  sync_interval={self.sync_interval},\n"
            f"  normalize_obs={self.normalize_obs},\n"
            f"  normalize_reward={self.normalize_reward},\n"
            f"  auto_reset_on_fall={self.auto_reset_on_fall},\n"
            f"  device='{self.device}'\n"
            f")"
        )
