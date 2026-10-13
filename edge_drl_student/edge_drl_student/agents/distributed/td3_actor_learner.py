#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

"""
TD3 Actor-Learner Implementation

Implements the actor-learner pattern specifically for TD3 (Twin Delayed DDPG) algorithm.

Reference:
    Addressing Function Approximation Error in Actor-Critic Methods (ICML 2018)
    https://arxiv.org/abs/1802.09477
"""

from typing import Tuple, Optional

from edge_drl_student.agents.off_policy.td3_agent import TD3Agent
from edge_drl_student.agents.distributed.actor_learner import ActorLearner
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer


class TD3ActorLearner(ActorLearner):
    """
    TD3 Actor-Learner Agent
    
    Implements IMPALA-style actor-learner architecture for TD3:
        - Acting Agent: Fast action generation (high frequency)
        - Learning Agent: Slow policy updates (low frequency)
        - Periodic parameter synchronization
    
    TD3-Specific Components:
        ✅ Actor network
        ✅ Twin Critic networks (Critic1, Critic2)
        ✅ Target Actor network
        ✅ Twin Target Critic networks (Target Critic1, Target Critic2)
        ✅ Policy delay mechanism
        ✅ Target policy smoothing
        ✅ Actor normalizer (optional)
        ✅ Critic normalizer (optional)
        ✅ Reward normalizer (optional)
    
    TD3 Key Features:
        1. Twin Q-networks: Reduce overestimation bias
        2. Delayed policy updates: Update actor less frequently than critics
        3. Target policy smoothing: Add noise to target actions
    
    Example:
        >>> from edge_drl_student.agents.distributed.td3_actor_learner import TD3ActorLearner
        >>> 
        >>> actor_learner = TD3ActorLearner(
        ...     obs_dim=36,
        ...     act_dim=6,
        ...     replay_buffer=buffer,
        ...     device='cuda:0',
        ...     action_bound=(-10.0, 10.0),
        ...     actor_lr=3e-4,
        ...     critic_lr=3e-4,
        ...     gamma=0.99,
        ...     tau=0.005,
        ...     policy_noise=0.2,
        ...     noise_clip=0.5,
        ...     policy_delay=2,
        ...     hidden_dim=256
        ... )
        >>> 
        >>> # Update policy
        >>> losses = actor_learner.update(batch_size=256, sync_interval=5)
        >>> print(f"Actor loss: {losses.get('actor_loss', 'N/A')}")
        >>> print(f"Critic1 loss: {losses['critic1_loss']:.4f}")
        >>> print(f"Critic2 loss: {losses['critic2_loss']:.4f}")
        >>> 
        >>> # Get statistics
        >>> sync_stats = actor_learner.get_sync_stats()
        >>> print(f"Avg sync time: {sync_stats['avg_sync_time']*1000:.2f}ms")
    """
    
    # Type hints for better IDE support
    acting_agent: TD3Agent
    learning_agent: TD3Agent
    
    def __init__(
        self,
        # Core parameters
        obs_dim: int,
        act_dim: int,
        replay_buffer: BaseBuffer,
        device: str,
        action_bound: Tuple[float, float],
        # TD3 hyperparameters
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_dim: int = 256,
        # TD3-specific parameters
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        # Normalization options
        actor_normalizer: bool = False,
        critic_normalizer: bool = False,
        reward_normalizer: bool = False,
        # Noise options
        exploration_noise: float = 0.1,
        # Optional parameters
        actor_hidden_layers: Optional[int] = None,
        critic_hidden_layers: Optional[int] = None,
    ):
        """
        Initialize TD3 Actor-Learner
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            replay_buffer: Replay buffer instance (shared between agents)
            device: PyTorch device ('cpu' or 'cuda:0')
            action_bound: Action bounds (min, max)
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            hidden_dim: Hidden layer dimension
            policy_noise: Target policy smoothing noise std
            noise_clip: Target policy smoothing noise clipping range
            policy_delay: Policy update delay (update actor every N critic updates)
            actor_normalizer: Use normalization for actor inputs
            critic_normalizer: Use normalization for critic inputs
            reward_normalizer: Use reward normalization
            exploration_noise: Exploration noise scale
            actor_hidden_layers: Number of hidden layers in actor (default: 2)
            critic_hidden_layers: Number of hidden layers in critics (default: 2)
        
        TD3 Parameter Guidelines:
            - policy_noise: Typically 0.2 (20% of action range)
            - noise_clip: Typically 0.5 (50% of action range)
            - policy_delay: Typically 2 (update actor every 2 critic updates)
        
        Example:
            >>> actor_learner = TD3ActorLearner(
            ...     obs_dim=36,
            ...     act_dim=6,
            ...     replay_buffer=buffer,
            ...     device='cuda:0',
            ...     action_bound=(-10.0, 10.0),
            ...     actor_lr=3e-4,
            ...     critic_lr=3e-4,
            ...     gamma=0.99,
            ...     tau=0.005,
            ...     policy_noise=0.2,
            ...     noise_clip=0.5,
            ...     policy_delay=2
            ... )
        """
        # Store TD3-specific parameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.replay_buffer = replay_buffer
        self.device = device
        self.action_bound = action_bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.actor_normalizer = actor_normalizer
        self.critic_normalizer = critic_normalizer
        self.reward_normalizer = reward_normalizer
        self.exploration_noise = exploration_noise
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers
        
        # Log initialization
        print(f"\n{'='*70}")
        print(f"🔧 Initializing TD3ActorLearner")
        print(f"{'='*70}")
        print(f"📐 Architecture:")
        print(f"   Observation Dim:    {obs_dim}")
        print(f"   Action Dim:         {act_dim}")
        print(f"   Hidden Dim:         {hidden_dim}")
        print(f"   Action Bound:       {action_bound}")
        print(f"─────────────────────────────────────────────────────────────────────")
        print(f"🎓 Hyperparameters:")
        print(f"   Actor LR:           {actor_lr}")
        print(f"   Critic LR:          {critic_lr}")
        print(f"   Gamma:              {gamma}")
        print(f"   Tau:                {tau}")
        print(f"   Exploration Noise:  {exploration_noise}")
        print(f"─────────────────────────────────────────────────────────────────────")
        print(f"🎯 TD3-Specific:")
        print(f"   Policy Noise:       {policy_noise}")
        print(f"   Noise Clip:         {noise_clip}")
        print(f"   Policy Delay:       {policy_delay}")
        print(f"─────────────────────────────────────────────────────────────────────")
        print(f"🔧 Normalization:")
        print(f"   Actor Normalizer:   {actor_normalizer}")
        print(f"   Critic Normalizer:  {critic_normalizer}")
        print(f"   Reward Normalizer:  {reward_normalizer}")
        print(f"─────────────────────────────────────────────────────────────────────")
        print(f"💻 Device: {device}")
        print(f"{'='*70}\n")
        
        # Create agents BEFORE calling super().__init__()
        print(f"🔧 Creating TD3 acting agent...")
        self.acting_agent = self._create_agent()
        
        print(f"🔧 Creating TD3 learning agent...")
        self.learning_agent = self._create_agent()
        
        # Initialize base class (sets up sync statistics)
        super().__init__()
        
        print(f"\n{'='*70}")
        print(f"✅ TD3ActorLearner initialized successfully")
        print(f"   Twin Critics:       ✓")
        print(f"   Policy Delay:       ✓")
        print(f"   Target Smoothing:   ✓")
        print(f"{'='*70}\n")
    
    def _create_agent(self) -> TD3Agent:
        """
        Create TD3 agent instance
        
        Returns:
            TD3Agent instance with all parameters configured
        """
        return TD3Agent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            replay_buffer=self.replay_buffer,
            action_bound=self.action_bound,
            device=self.device,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            gamma=self.gamma,
            tau=self.tau,
            hidden_dim=self.hidden_dim,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
            policy_delay=self.policy_delay,
            actor_normalizer=self.actor_normalizer,
            critic_normalizer=self.critic_normalizer,
            reward_normalizer=self.reward_normalizer,
            exploration_noise=self.exploration_noise,
            actor_hidden_layers=self.actor_hidden_layers,
            critic_hidden_layers=self.critic_hidden_layers,
        )
    
    def _sync_agent_parameters(self):
        """
        Synchronize TD3-specific parameters
        
        Synchronization order:
            1. Online networks (Actor, Critic1, Critic2)
            2. Target networks (Target Actor, Target Critic1, Target Critic2)
            3. Normalizers (if exist)
        
        Note:
            TD3 has twin critics, so we need to sync both critic networks
            and their corresponding target networks.
        """
        # ========== 1. Online Networks ==========
        
        # Actor network
        self.acting_agent.actor.load_state_dict(
            self.learning_agent.actor.state_dict()
        )
        
        # Twin Critic networks
        self.acting_agent.critic1.load_state_dict(
            self.learning_agent.critic1.state_dict()
        )
        self.acting_agent.critic2.load_state_dict(
            self.learning_agent.critic2.state_dict()
        )
        
        # ========== 2. Target Networks ==========
        
        # Target Actor
        self.acting_agent.target_actor.load_state_dict(
            self.learning_agent.target_actor.state_dict()
        )
        
        # Twin Target Critics
        self.acting_agent.target_critic1.load_state_dict(
            self.learning_agent.target_critic1.state_dict()
        )
        self.acting_agent.target_critic2.load_state_dict(
            self.learning_agent.target_critic2.state_dict()
        )
        
        # ========== 3. Normalizers (Optional) ==========
        
        # Actor normalizer
        if (hasattr(self.learning_agent, 'actor_normalizer') and 
            self.learning_agent.actor_normalizer is not None):
            self.acting_agent.actor_normalizer.load_state_dict(
                self.learning_agent.actor_normalizer.state_dict()
            )
        
        # Critic normalizer
        if (hasattr(self.learning_agent, 'critic_normalizer') and 
            self.learning_agent.critic_normalizer is not None):
            self.acting_agent.critic_normalizer.load_state_dict(
                self.learning_agent.critic_normalizer.state_dict()
            )
        
        # Reward normalizer
        if (hasattr(self.learning_agent, 'reward_normalizer') and 
            self.learning_agent.reward_normalizer is not None):
            self.acting_agent.reward_normalizer.load_state_dict(
                self.learning_agent.reward_normalizer.state_dict()
            )
    
    # ==================== Additional Methods ====================
    
    def get_agent_info(self) -> dict:
        """
        Get TD3-specific agent information
        
        Returns:
            Dictionary containing agent configuration
        
        Example:
            >>> info = actor_learner.get_agent_info()
            >>> print(f"Policy Delay: {info['policy_delay']}")
            >>> print(f"Policy Noise: {info['policy_noise']}")
        """
        return {
            'class_name': self.__class__.__name__,
            'acting_agent': self.acting_agent.__class__.__name__,
            'learning_agent': self.learning_agent.__class__.__name__,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'action_bound': self.action_bound,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dim': self.hidden_dim,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'policy_delay': self.policy_delay,
            'device': self.device,
            'actor_normalizer': self.actor_normalizer,
            'critic_normalizer': self.critic_normalizer,
            'reward_normalizer': self.reward_normalizer,
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"TD3ActorLearner(\n"
            f"  obs_dim={self.obs_dim},\n"
            f"  act_dim={self.act_dim},\n"
            f"  actor_lr={self.actor_lr},\n"
            f"  critic_lr={self.critic_lr},\n"
            f"  gamma={self.gamma},\n"
            f"  tau={self.tau},\n"
            f"  policy_noise={self.policy_noise},\n"
            f"  noise_clip={self.noise_clip},\n"
            f"  policy_delay={self.policy_delay},\n"
            f"  device='{self.device}',\n"
            f"  update_count={self.update_count},\n"
            f"  sync_count={self.sync_count}\n"
            f")"
        )
