#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

"""
SAC Actor-Learner Implementation

Implements the actor-learner pattern specifically for SAC (Soft Actor-Critic) algorithm.

Reference:
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor (ICML 2018)
    https://arxiv.org/abs/1801.01290
    
    Soft Actor-Critic Algorithms and Applications (2019)
    https://arxiv.org/abs/1812.05905
"""

from typing import Tuple, Optional

from edge_drl_student.agents.off_policy.sac_agent import SACAgent
from edge_drl_student.agents.distributed.actor_learner import ActorLearner
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer


class SACActorLearner(ActorLearner):
    """
    SAC Actor-Learner Agent
    
    Implements IMPALA-style actor-learner architecture for SAC:
        - Acting Agent: Fast action generation (high frequency)
        - Learning Agent: Slow policy updates (low frequency)
        - Periodic parameter synchronization
    
    SAC-Specific Components:
        ✅ Stochastic Actor network (outputs mean and log_std)
        ✅ Twin Critic networks (Critic1, Critic2)
        ✅ Twin Target Critic networks (Target Critic1, Target Critic2)
        ✅ Entropy temperature (alpha)
        ✅ Automatic entropy tuning (optional)
        ✅ Actor normalizer (optional)
        ✅ Critic normalizer (optional)
        ✅ Reward normalizer (optional)
    
    SAC Key Features:
        1. Maximum entropy RL: Encourage exploration through entropy maximization
        2. Stochastic policy: Actor outputs a distribution (Gaussian)
        3. Automatic temperature tuning: Dynamically adjust entropy coefficient
        4. Twin Q-networks: Reduce overestimation bias
        5. No target actor: Only critics have target networks
    
    Example:
        >>> from edge_drl_student.agents.distributed.sac_actor_learner import SACActorLearner
        >>> 
        >>> actor_learner = SACActorLearner(
        ...     obs_dim=36,
        ...     act_dim=6,
        ...     replay_buffer=buffer,
        ...     device='cuda:0',
        ...     action_bound=(-10.0, 10.0),
        ...     actor_lr=3e-4,
        ...     critic_lr=3e-4,
        ...     alpha_lr=3e-4,
        ...     gamma=0.99,
        ...     tau=0.005,
        ...     auto_entropy_tuning=True,
        ...     target_entropy=-6.0,
        ...     hidden_dim=256
        ... )
        >>> 
        >>> # Update policy
        >>> losses = actor_learner.update(batch_size=256, sync_interval=5)
        >>> print(f"Actor loss: {losses['actor_loss']:.4f}")
        >>> print(f"Critic1 loss: {losses['critic1_loss']:.4f}")
        >>> print(f"Critic2 loss: {losses['critic2_loss']:.4f}")
        >>> print(f"Alpha: {losses['alpha']:.4f}")
        >>> 
        >>> # Get statistics
        >>> sync_stats = actor_learner.get_sync_stats()
        >>> print(f"Avg sync time: {sync_stats['avg_sync_time']*1000:.2f}ms")
    """
    
    # Type hints for better IDE support
    acting_agent: SACAgent
    learning_agent: SACAgent
    
    def __init__(
        self,
        # Core parameters
        obs_dim: int,
        act_dim: int,
        replay_buffer: BaseBuffer,
        device: str,
        action_bound: Tuple[float, float],
        # SAC hyperparameters
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_dim: int = 256,
        # SAC-specific parameters
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        init_alpha: float = 0.2,
        # Normalization options
        actor_normalizer: bool = False,
        critic_normalizer: bool = False,
        reward_normalizer: bool = False,
        # Optional parameters
        actor_hidden_layers: Optional[int] = None,
        critic_hidden_layers: Optional[int] = None,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        """
        Initialize SAC Actor-Learner
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            replay_buffer: Replay buffer instance (shared between agents)
            device: PyTorch device ('cpu' or 'cuda:0')
            action_bound: Action bounds (min, max)
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            alpha_lr: Entropy temperature learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            hidden_dim: Hidden layer dimension
            auto_entropy_tuning: Enable automatic entropy tuning
            target_entropy: Target entropy (default: -act_dim)
            init_alpha: Initial entropy temperature
            actor_normalizer: Use normalization for actor inputs
            critic_normalizer: Use normalization for critic inputs
            reward_normalizer: Use reward normalization
            actor_hidden_layers: Number of hidden layers in actor (default: 2)
            critic_hidden_layers: Number of hidden layers in critics (default: 2)
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        
        SAC Parameter Guidelines:
            - auto_entropy_tuning: Recommended to set True
            - target_entropy: Typically -act_dim (e.g., -6 for 6D action)
            - init_alpha: Typically 0.2 (will be tuned automatically if auto_entropy_tuning=True)
            - alpha_lr: Same as actor_lr typically works well
        
        Example:
            >>> actor_learner = SACActorLearner(
            ...     obs_dim=36,
            ...     act_dim=6,
            ...     replay_buffer=buffer,
            ...     device='cuda:0',
            ...     action_bound=(-10.0, 10.0),
            ...     actor_lr=3e-4,
            ...     critic_lr=3e-4,
            ...     alpha_lr=3e-4,
            ...     gamma=0.99,
            ...     tau=0.005,
            ...     auto_entropy_tuning=True,
            ...     target_entropy=-6.0
            ... )
        """
        # Store SAC-specific parameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.replay_buffer = replay_buffer
        self.device = device
        self.action_bound = action_bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.auto_entropy_tuning = auto_entropy_tuning
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim
        self.init_alpha = init_alpha
        self.actor_normalizer = actor_normalizer
        self.critic_normalizer = critic_normalizer
        self.reward_normalizer = reward_normalizer
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Log initialization
        print(f"\n{'='*70}")
        print(f"🔧 Initializing SACActorLearner")
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
        print(f"   Alpha LR:           {alpha_lr}")
        print(f"   Gamma:              {gamma}")
        print(f"   Tau:                {tau}")
        print(f"─────────────────────────────────────────────────────────────────────")
        print(f"🎯 SAC-Specific:")
        print(f"   Auto Entropy Tune:  {auto_entropy_tuning}")
        print(f"   Target Entropy:     {self.target_entropy}")
        print(f"   Init Alpha:         {init_alpha}")
        print(f"   Log Std Range:      [{log_std_min}, {log_std_max}]")
        print(f"─────────────────────────────────────────────────────────────────────")
        print(f"🔧 Normalization:")
        print(f"   Actor Normalizer:   {actor_normalizer}")
        print(f"   Critic Normalizer:  {critic_normalizer}")
        print(f"   Reward Normalizer:  {reward_normalizer}")
        print(f"─────────────────────────────────────────────────────────────────────")
        print(f"💻 Device: {device}")
        print(f"{'='*70}\n")
        
        # Create agents BEFORE calling super().__init__()
        print(f"🔧 Creating SAC acting agent...")
        self.acting_agent = self._create_agent()
        
        print(f"🔧 Creating SAC learning agent...")
        self.learning_agent = self._create_agent()
        
        # Initialize base class (sets up sync statistics)
        super().__init__()
        
        print(f"\n{'='*70}")
        print(f"✅ SACActorLearner initialized successfully")
        print(f"   Stochastic Policy:  ✓")
        print(f"   Twin Critics:       ✓")
        print(f"   Entropy Tuning:     {'✓ (Auto)' if auto_entropy_tuning else '✓ (Manual)'}")
        print(f"{'='*70}\n")
    
    def _create_agent(self) -> SACAgent:
        """
        Create SAC agent instance
        
        Returns:
            SACAgent instance with all parameters configured
        """
        return SACAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            replay_buffer=self.replay_buffer,
            action_bound=self.action_bound,
            device=self.device,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            alpha_lr=self.alpha_lr,
            gamma=self.gamma,
            tau=self.tau,
            hidden_dim=self.hidden_dim,
            auto_entropy_tuning=self.auto_entropy_tuning,
            target_entropy=self.target_entropy,
            init_alpha=self.init_alpha,
            actor_normalizer=self.actor_normalizer,
            critic_normalizer=self.critic_normalizer,
            reward_normalizer=self.reward_normalizer,
            actor_hidden_layers=self.actor_hidden_layers,
            critic_hidden_layers=self.critic_hidden_layers,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )
    
    def _sync_agent_parameters(self):
        """
        Synchronize SAC-specific parameters
        
        Synchronization order:
            1. Online networks (Actor, Critic1, Critic2)
            2. Target networks (Target Critic1, Target Critic2)
            3. Entropy temperature (log_alpha)
            4. Normalizers (if exist)
        
        Note:
            SAC has NO target actor network (unlike DDPG/TD3).
            Only critics have target networks.
            
            SAC uses a stochastic policy, so we need to sync the entire
            actor network that outputs both mean and log_std.
        """
        # ========== 1. Online Networks ==========
        
        # Stochastic Actor network (outputs mean and log_std)
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
        
        # Twin Target Critics (NO target actor in SAC)
        self.acting_agent.target_critic1.load_state_dict(
            self.learning_agent.target_critic1.state_dict()
        )
        self.acting_agent.target_critic2.load_state_dict(
            self.learning_agent.target_critic2.state_dict()
        )
        
        # ========== 3. Entropy Temperature (log_alpha) ==========
        
        # Sync log_alpha (learnable entropy temperature)
        if hasattr(self.learning_agent, 'log_alpha'):
            self.acting_agent.log_alpha.data.copy_(
                self.learning_agent.log_alpha.data
            )
        
        # ========== 4. Normalizers (Optional) ==========
        
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
        Get SAC-specific agent information
        
        Returns:
            Dictionary containing agent configuration
        
        Example:
            >>> info = actor_learner.get_agent_info()
            >>> print(f"Auto Entropy Tuning: {info['auto_entropy_tuning']}")
            >>> print(f"Target Entropy: {info['target_entropy']}")
            >>> print(f"Current Alpha: {info['current_alpha']}")
        """
        # Get current alpha value
        current_alpha = None
        if hasattr(self.learning_agent, 'log_alpha'):
            import torch
            current_alpha = torch.exp(self.learning_agent.log_alpha).item()
        
        return {
            'class_name': self.__class__.__name__,
            'acting_agent': self.acting_agent.__class__.__name__,
            'learning_agent': self.learning_agent.__class__.__name__,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'action_bound': self.action_bound,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'alpha_lr': self.alpha_lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dim': self.hidden_dim,
            'auto_entropy_tuning': self.auto_entropy_tuning,
            'target_entropy': self.target_entropy,
            'init_alpha': self.init_alpha,
            'current_alpha': current_alpha,
            'device': self.device,
            'actor_normalizer': self.actor_normalizer,
            'critic_normalizer': self.critic_normalizer,
            'reward_normalizer': self.reward_normalizer,
        }
    
    def get_current_alpha(self) -> float:
        """
        Get current entropy temperature value
        
        Returns:
            Current alpha value (exp(log_alpha))
        
        Example:
            >>> alpha = actor_learner.get_current_alpha()
            >>> print(f"Current entropy temperature: {alpha:.4f}")
        """
        if hasattr(self.learning_agent, 'log_alpha'):
            import torch
            return torch.exp(self.learning_agent.log_alpha).item()
        return self.init_alpha
    
    def __repr__(self) -> str:
        """String representation"""
        current_alpha = self.get_current_alpha()
        return (
            f"SACActorLearner(\n"
            f"  obs_dim={self.obs_dim},\n"
            f"  act_dim={self.act_dim},\n"
            f"  actor_lr={self.actor_lr},\n"
            f"  critic_lr={self.critic_lr},\n"
            f"  alpha_lr={self.alpha_lr},\n"
            f"  gamma={self.gamma},\n"
            f"  tau={self.tau},\n"
            f"  auto_entropy_tuning={self.auto_entropy_tuning},\n"
            f"  target_entropy={self.target_entropy},\n"
            f"  current_alpha={current_alpha:.4f},\n"
            f"  device='{self.device}',\n"
            f"  update_count={self.update_count},\n"
            f"  sync_count={self.sync_count}\n"
            f")"
        )
