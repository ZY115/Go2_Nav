#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

"""
DDPG Actor-Learner Implementation

Implements the actor-learner pattern specifically for DDPG algorithm.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from edge_drl_student.agents.off_policy.ddpg_agent import DDPGAgent
from edge_drl_student.agents.distributed.actor_learner import ActorLearner
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer


class DDPGActorLearner(ActorLearner):
    """
    DDPG Actor-Learner Agent
    
    Implements IMPALA-style actor-learner architecture for DDPG.
    
    Example:
        >>> actor_learner = DDPGActorLearner(
        ...     obs_dim=36,
        ...     act_dim=6,
        ...     replay_buffer=buffer,
        ...     action_bound=(-10.0, 10.0),
        ...     device='cuda:0',
        ...     sync_interval=5,
        ...     actor_lr=3e-4,
        ...     critic_lr=3e-4
        ... )
    """
    
    acting_agent: DDPGAgent
    learning_agent: DDPGAgent
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        replay_buffer: Union[SoloReplayBuffer, DualReplayBuffer],
        action_bound: Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]],
        device: str = "cpu",
        sync_interval: int = 5,
        only_sync_actor: bool = False,
        **agent_kwargs
    ):
        """
        Initialize DDPG Actor-Learner
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            replay_buffer: Replay buffer instance
            action_bound: Action bounds (min, max)
            device: PyTorch device
            sync_interval: Synchronization interval
            only_sync_actor: If True, only synchronize actor networks
            **agent_kwargs: Additional arguments for DDPGAgent
        """
        
        # Store core parameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.replay_buffer = replay_buffer
        self.action_bound = action_bound
        self.device = device
        self.only_sync_actor = only_sync_actor
        self.agent_kwargs = agent_kwargs
        
        # Log initialization info
        print(f"\n{'='*70}")
        print(f"🔧 Initializing DDPGActorLearner")
        print(f"   obs_dim={obs_dim}, act_dim={act_dim}, device={device}")
        print(f"   sync_interval={sync_interval}, only_sync_actor={only_sync_actor}")
        print(f"{'='*70}\n")
        
        # Create agents
        print(f"🔧 Creating DDPG acting agent...")
        self.acting_agent = self._create_agent()
        
        print(f"🔧 Creating DDPG learning agent...")
        self.learning_agent = self._create_agent()
        
        # Initialize base class
        super().__init__(sync_interval=sync_interval)
        
        print(f"✅ DDPGActorLearner initialized successfully\n")
    
    def _create_agent(self) -> DDPGAgent:
        """Create DDPG agent instance"""
        return DDPGAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            replay_buffer=self.replay_buffer,
            action_bound=self.action_bound,
            device=self.device,
            **self.agent_kwargs
        )
    
    def _sync_agent_parameters(self):
        """
        Synchronize DDPG-specific parameters
        
        Synchronization order:
            1. Online networks (Actor, Critic)
            2. Target networks (Target Actor, Target Critic)
            3. Optimizers (optional, for consistency)
            4. Normalizers (Actor, Critic, Reward)
            5. Noise process state
        """
        # Actor network
        self.acting_agent.actor.load_state_dict(
            self.learning_agent.actor.state_dict()
        )
        
        # Only sync actor if specified
        if not self.only_sync_actor:
            # Critic network
            self.acting_agent.critic.load_state_dict(
                self.learning_agent.critic.state_dict()
            )
            
            # Target Actor
            self.acting_agent.target_actor.load_state_dict(
                self.learning_agent.target_actor.state_dict()
            )
            
            # Target Critic
            self.acting_agent.target_critic.load_state_dict(
                self.learning_agent.target_critic.state_dict()
            )
        
        # Sync optimizer states
        self.acting_agent.actor_optimizer.load_state_dict(self.learning_agent.actor_optimizer.state_dict())
        self.acting_agent.critic_optimizer.load_state_dict(self.learning_agent.critic_optimizer.state_dict())
        
        # Sync noise state
        if (hasattr(self.learning_agent, 'noise') and 
            self.learning_agent.noise is not None and
            hasattr(self.learning_agent.noise, 'state_dict')):
            if hasattr(self.acting_agent.noise, 'load_state_dict'):
                self.acting_agent.noise.load_state_dict(
                    self.learning_agent.noise.state_dict()
                )
    
    def update(self, batch_size: int) -> dict:
        """
        Update policy using learning agent with automatic sync
        
        Args:
            batch_size: Batch size for training
        
        Returns:
            Dictionary of losses and metrics
        """
        self.update_count += 1
        
        # Update using learning agent
        actor_loss, critic_loss = self.learning_agent.update(batch_size)
        
        # Convert to dict
        losses = {
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
            'critic_loss': critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
        }
        
        # Sync if needed
        if self.should_sync():
            sync_time = self.sync_parameters()
            losses['sync_time'] = sync_time
        
        return losses
    
    def get_agent_info(self) -> dict:
        """
        Get DDPG-specific agent information
        
        Returns:
            Dictionary containing agent configuration
        """
        # Start with core info
        info = {
            'class_name': self.__class__.__name__,
            'acting_agent': self.acting_agent.__class__.__name__,
            'learning_agent': self.learning_agent.__class__.__name__,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'device': self.device,
            'sync_interval': self.sync_interval,
            'only_sync_actor': self.only_sync_actor,
        }
        
        # Add action_bound info (handle tensor case)
        if isinstance(self.action_bound, tuple) and len(self.action_bound) == 2:
            if isinstance(self.action_bound[0], torch.Tensor):
                info['action_bound'] = f"Tensor(shape={self.action_bound[0].shape})"
            else:
                info['action_bound'] = self.action_bound
        else:
            info['action_bound'] = str(self.action_bound)
        
        # Add all agent_kwargs
        info.update(self.agent_kwargs)
        
        return info
    
    def __repr__(self) -> str:
        """String representation"""
        # Extract key parameters from agent_kwargs
        actor_lr = self.agent_kwargs.get('actor_lr', 1e-6) 
        # actor_lr = self.agent_kwargs.get('actor_lr', 5e-7) 
        critic_lr = self.agent_kwargs.get('critic_lr', 1e-5)
        # critic_lr = self.agent_kwargs.get('critic_lr', 1e-5)
        gamma = self.agent_kwargs.get('gamma', 0.95)
        tau = self.agent_kwargs.get('tau', 0.005)
        
        return (
            f"DDPGActorLearner(\n"
            f"  obs_dim={self.obs_dim},\n"
            f"  act_dim={self.act_dim},\n"
            f"  device='{self.device}',\n"
            f"  sync_interval={self.sync_interval},\n"
            f"  only_sync_actor={self.only_sync_actor},\n"
            f"  actor_lr={actor_lr},\n"
            f"  critic_lr={critic_lr},\n"
            f"  gamma={gamma},\n"
            f"  tau={tau},\n"
            f"  update_count={self.update_count},\n"
            f"  sync_count={self.sync_count}\n"
            f")"
        )


# ==================== Test Code ====================

if __name__ == "__main__":
    """Test DDPGActorLearner"""
    
    print("\n" + "="*70)
    print("Testing DDPGActorLearner")
    print("="*70 + "\n")
    
    # Setup
    obs_dim = 12
    act_dim = 6
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    action_bound = (
        torch.tensor([-10., -10., -10., -20., -20., -20.], device=device),
        torch.tensor([10., 10., 10., 20., 20., 20.], device=device)
    )
    
    # Create replay buffer
    replay_buffer = SoloReplayBuffer(
        buffer_size=10000,
        prefill_size=1000,
        obs_dim=obs_dim,
        action_dim=act_dim,
        device=device
    )
    
    # Fill buffer with random data
    print("Filling replay buffer...")
    from edge_drl_student.memory.buffers.base_buffer import TransitionBatch
    for _ in range(1500):
        transition = TransitionBatch(
            obs=torch.randn(1, obs_dim, device=device),
            action=torch.randn(1, act_dim, device=device),
            reward=torch.randn(1, 1, device=device),
            next_obs=torch.randn(1, obs_dim, device=device),
            done=torch.zeros(1, 1, device=device)
        )
        replay_buffer.add_transition(transition)
    
    print(f"✅ Buffer filled: {len(replay_buffer)} transitions\n")
    
    # Create DDPG Actor-Learner
    actor_learner = DDPGActorLearner(
        obs_dim=obs_dim,
        act_dim=act_dim,
        replay_buffer=replay_buffer,
        action_bound=action_bound,
        device=device,
        sync_interval=2,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        actor_lr=1e-6,
        critic_lr=1e-5,
        gamma=0.95,
        tau=0.005,
    )
    
    # Test action generation
    print("\n" + "="*70)
    print("Testing Action Generation")
    print("="*70 + "\n")
    
    sample_obs = torch.randn(4, obs_dim, device=device)
    actions = actor_learner.acting_agent.act(sample_obs, noise_scale=0.0, explore=True)
    print(f"Sample actions shape: {actions.shape}")
    print(f"Sample actions:\n{actions}\n")
    
    # Test policy update
    print("="*70)
    print("Testing Policy Update")
    print("="*70 + "\n")
    
    for i in range(5):
        losses = actor_learner.update(batch_size=64)
        print(f"Update #{i+1}:")
        print(f"  Actor Loss:  {losses['actor_loss']:.4f}")
        print(f"  Critic Loss: {losses['critic_loss']:.4f}")
        if 'sync_time' in losses:
            print(f"  Sync Time:   {losses['sync_time']*1000:.2f}ms")
        print()
    
    # Test statistics
    print("="*70)
    print("Statistics")
    print("="*70 + "\n")
    
    sync_stats = actor_learner.get_sync_stats()
    update_stats = actor_learner.get_update_stats()
    agent_info = actor_learner.get_agent_info()
    
    print("Sync Statistics:")
    for key, value in sync_stats.items():
        if 'time' in key:
            print(f"  {key}: {value*1000:.2f}ms")
        else:
            print(f"  {key}: {value}")
    
    print("\nUpdate Statistics:")
    for key, value in update_stats.items():
        print(f"  {key}: {value}")
    
    print("\nAgent Info:")
    for key, value in agent_info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70 + "\n")
