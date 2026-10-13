#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

"""
Base Actor-Learner Architecture Module
Implements IMPALA-style distributed RL architecture (algorithm-agnostic)

Reference:
    IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
    https://arxiv.org/abs/1802.01561
    
Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                   ActorLearner                          │
    │                                                         │
    │    ┌──────────────────┐         ┌──────────────────┐    │
    │    │  Acting Agent    │         │  Learning Agent  │    │
    │    │ (Fast Inference) │  ◄────  │  (Slow Training) │    │
    │    └──────────────────┘         └──────────────────┘    │
    │                                                         │
    │   Responsibilities:                                     │
    │     • Manage two agents (acting & learning)             │
    │     • Synchronize parameters periodically               │
    │     • Track sync/update statistics                      │
    │     • Provide unified interface                         │
    └─────────────────────────────────────────────────────────┘

Design Philosophy:
    - Base class is algorithm-agnostic (no algorithm-specific parameters)
    - Subclasses handle their own initialization parameters
    - Clear separation of concerns (sync logic vs algorithm logic)
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from edge_drl_student.agents.base_agent import BaseAgent


class ActorLearner(ABC):
    """
    Abstract Actor-Learner Architecture Base Class
    
    This class implements the core actor-learner pattern without knowing
    about specific RL algorithms (DDPG, TD3, SAC, etc.).
    
    Key Features:
        ✅ Algorithm-agnostic design
        ✅ Thread-safe parameter synchronization
        ✅ Automatic sync scheduling (based on sync_interval)
        ✅ Comprehensive statistics tracking
        ✅ Clean separation of concerns
    
    Subclass Responsibilities:
        1. Store algorithm-specific parameters in __init__
        2. Implement _create_agent() to instantiate agents
        3. Implement _sync_agent_parameters() to sync networks
        4. (Optional) Override update() to convert return format
    
    Example Subclass:
        >>> class DDPGActorLearner(ActorLearner):
        ...     def __init__(self, obs_dim, act_dim, sync_interval=5, ...):
        ...         # 1. Store DDPG-specific parameters
        ...         self.obs_dim = obs_dim
        ...         self.act_dim = act_dim
        ...         self.actor_lr = actor_lr
        ...         ...
        ...         
        ...         # 2. Create agents
        ...         self.acting_agent = self._create_agent()
        ...         self.learning_agent = self._create_agent()
        ...         
        ...         # 3. Initialize base class
        ...         super().__init__(sync_interval=sync_interval)
        ...     
        ...     def _create_agent(self):
        ...         return DDPGAgent(self.obs_dim, self.act_dim, ...)
        ...     
        ...     def _sync_agent_parameters(self):
        ...         # Sync DDPG networks
        ...         self.acting_agent.actor.load_state_dict(...)
        ...         ...
    """
    
    # Type hints (to be set by subclasses)
    acting_agent: BaseAgent
    learning_agent: BaseAgent
    
    def __init__(self, sync_interval: int = 5):
        """
        Initialize ActorLearner base class
        
        Args:
            sync_interval: Synchronization interval (number of updates)
        
        Note:
            This should be called AFTER subclass creates both agents.
            
        Example:
            >>> class MyActorLearner(ActorLearner):
            ...     def __init__(self, obs_dim, act_dim, sync_interval=5, ...):
            ...         # 1. Store parameters
            ...         self.obs_dim = obs_dim
            ...         self.act_dim = act_dim
            ...         
            ...         # 2. Create agents
            ...         self.acting_agent = self._create_agent()
            ...         self.learning_agent = self._create_agent()
            ...         
            ...         # 3. Initialize base class
            ...         super().__init__(sync_interval=sync_interval)
        """
        # Validate that agents are created
        if not hasattr(self, 'acting_agent') or not hasattr(self, 'learning_agent'):
            raise RuntimeError(
                f"{self.__class__.__name__} must create 'acting_agent' and "
                f"'learning_agent' before calling super().__init__()"
            )
        
        # Store configuration
        self.sync_interval = sync_interval
        
        # Thread safety for parameter synchronization
        self.sync_lock = threading.Lock()
        
        # Synchronization statistics
        self.sync_count: int = 0
        self.last_sync_time: float = 0.0
        self.total_sync_time: float = 0.0
        
        # Update statistics
        self.update_count: int = 0
        
        print(f"✅ {self.__class__.__name__} initialized")
        print(f"   Acting Agent:   {self.acting_agent.__class__.__name__}")
        print(f"   Learning Agent: {self.learning_agent.__class__.__name__}")
        print(f"   Sync Interval:  every {sync_interval} updates")
    
    @abstractmethod
    def _create_agent(self) -> BaseAgent:
        """
        Create algorithm-specific agent instance
        
        This method is called by subclass __init__ to create both
        acting_agent and learning_agent.
        
        Returns:
            Agent instance (e.g., DDPGAgent, TD3Agent, SACAgent)
        
        Implementation Guidelines:
            - Use self.* to access parameters stored in subclass __init__
            - Create identical agents for acting and learning
            - Return a fully initialized agent
        
        Example (DDPG):
            >>> def _create_agent(self) -> DDPGAgent:
            ...     return DDPGAgent(
            ...         obs_dim=self.obs_dim,
            ...         act_dim=self.act_dim,
            ...         replay_buffer=self.replay_buffer,
            ...         action_bound=self.action_bound,
            ...         device=self.device,
            ...         actor_lr=self.actor_lr,
            ...         critic_lr=self.critic_lr,
            ...         gamma=self.gamma,
            ...         tau=self.tau
            ...     )
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _create_agent()"
        )
    
    @abstractmethod
    def _sync_agent_parameters(self):
        """
        Synchronize algorithm-specific parameters
        
        Copy all necessary parameters from learning_agent to acting_agent.
        This method is called within a thread lock, so thread safety is guaranteed.
        
        What to Sync:
            ✅ Actor network(s)
            ✅ Critic network(s)
            ✅ Target networks
            ✅ Optimizers (optional, for consistency)
            ✅ Normalizers (if any)
            ✅ Algorithm-specific parameters (e.g., log_alpha in SAC)
            ✅ Noise process state (if stateful)
        
        Implementation Guidelines:
            - Use load_state_dict() for PyTorch modules
            - Check if optional components exist before syncing
            - Sync in logical order (online networks → target networks → others)
        
        Example (DDPG):
            >>> def _sync_agent_parameters(self):
            ...     # Online networks
            ...     self.acting_agent.actor.load_state_dict(
            ...         self.learning_agent.actor.state_dict()
            ...     )
            ...     self.acting_agent.critic.load_state_dict(
            ...         self.learning_agent.critic.state_dict()
            ...     )
            ...     
            ...     # Target networks
            ...     self.acting_agent.target_actor.load_state_dict(
            ...         self.learning_agent.target_actor.state_dict()
            ...     )
            ...     self.acting_agent.target_critic.load_state_dict(
            ...         self.learning_agent.target_critic.state_dict()
            ...     )
            ...     
            ...     # Optimizers
            ...     self.acting_agent.actor_optimizer.load_state_dict(
            ...         self.learning_agent.actor_optimizer.state_dict()
            ...     )
            ...     
            ...     # Normalizers (optional)
            ...     if (hasattr(self.learning_agent, 'actor_normalizer') and 
            ...         self.learning_agent.actor_normalizer is not None):
            ...         self.acting_agent.actor_normalizer.load_state_dict(
            ...             self.learning_agent.actor_normalizer.state_dict()
            ...         )
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _sync_agent_parameters()"
        )
    
    # ==================== Core Methods ====================
    
    def sync_parameters(self) -> float:
        """
        Synchronize parameters from learning agent to acting agent
        
        This method is thread-safe and can be called from any thread.
        It uses a lock to ensure only one sync happens at a time.
        
        Returns:
            Synchronization time in seconds
        
        Example:
            >>> sync_time = actor_learner.sync_parameters()
            >>> print(f"Sync took {sync_time*1000:.2f}ms")
        """
        sync_start = time.time()
        
        # Thread-safe parameter synchronization
        with self.sync_lock:
            self._sync_agent_parameters()
        
        # Update statistics
        sync_time = time.time() - sync_start
        self.last_sync_time = sync_time
        self.total_sync_time += sync_time
        self.sync_count += 1
        
        # Log to console (every sync_interval syncs)
        if self.sync_count % self.sync_interval == 0:
            avg_sync_time = self.total_sync_time / self.sync_count
            print(
                f"🔄 Sync #{self.sync_count}: "
                f"{sync_time*1000:.2f}ms "
                f"(avg: {avg_sync_time*1000:.2f}ms)"
            )
        return sync_time
    
    def should_sync(self) -> bool:
        """
        Check if should synchronize parameters
        
        Uses self.sync_interval and self.update_count to determine
        if synchronization should happen.
        
        Returns:
            True if current update_count is a multiple of sync_interval
        
        Example:
            >>> if actor_learner.should_sync():
            ...     actor_learner.sync_parameters()
        """
        return self.update_count % self.sync_interval == 0
    
    def update(self, batch_size: int) -> Dict[str, float]:
        """
        Update policy using learning agent
        
        This method:
            1. Increments update counter
            2. Calls learning_agent.update()
            3. Syncs parameters if needed (based on self.sync_interval)
            4. Returns losses with sync time
        
        Args:
            batch_size: Batch size for training
        
        Returns:
            Dictionary of losses and metrics, e.g.:
            {
                'actor_loss': 0.123,
                'critic_loss': 0.456,
                'q_value': 1.234,
                'sync_time': 0.002  # (if synced)
            }
        
        Note:
            Subclasses may override this method to convert the return format
            from their specific agent's update() method (e.g., tuple to dict).
        
        Example:
            >>> losses = actor_learner.update(batch_size=256)
            >>> print(f"Actor loss: {losses['actor_loss']:.4f}")
            >>> if 'sync_time' in losses:
            ...     print(f"Synced in {losses['sync_time']*1000:.2f}ms")
        """
        self.update_count += 1
        
        # Update using learning agent
        losses = self.learning_agent.update(batch_size)
        
        # Check if should synchronize (uses self.sync_interval)
        if self.should_sync():
            sync_time = self.sync_parameters()
            if isinstance(losses, dict):
                losses['sync_time'] = sync_time
        
        return losses
    
    # ==================== Model Management ====================
    
    def save_models(self, path: str):
        """
        Save learning agent models to disk
        
        Args:
            path: Directory path to save models
        
        Example:
            >>> actor_learner.save_models('./models/checkpoint_1000')
        """
        self.learning_agent.save_models(path)
        print(f"💾 Models saved to: {path}")
    
    def load_models(self, path: str):
        """
        Load models from disk to both agents
        
        Args:
            path: Directory path to load models from
        
        Example:
            >>> actor_learner.load_models('./models/checkpoint_1000')
        """
        self.learning_agent.load_models(path)
        self.acting_agent.load_models(path)
        print(f"📂 Models loaded from: {path}")
    
    # ==================== Statistics ====================
    def get_sync_stats(self) -> Dict[str, Any]:
        """
        Get synchronization statistics
        
        Returns:
            Dictionary containing:
            - sync_count: Total number of synchronizations
            - last_sync_time: Last synchronization time (seconds)
            - total_sync_time: Total synchronization time (seconds)
            - avg_sync_time: Average synchronization time (seconds)
            - sync_interval: Synchronization interval
        
        Example:
            >>> stats = actor_learner.get_sync_stats()
            >>> print(f"Total syncs: {stats['sync_count']}")
            >>> print(f"Avg sync time: {stats['avg_sync_time']*1000:.2f}ms")
            >>> print(f"Sync interval: {stats['sync_interval']}")
        """
        return {
            'sync_count': self.sync_count,
            'last_sync_time': self.last_sync_time,
            'total_sync_time': self.total_sync_time,
            'avg_sync_time': self.total_sync_time / max(self.sync_count, 1),
            'sync_interval': self.sync_interval,
        }
    
    def get_update_stats(self) -> Dict[str, Any]:
        """
        Get update statistics
        
        Returns:
            Dictionary containing:
            - update_count: Total number of updates
            - sync_count: Total number of synchronizations
            - updates_per_sync: Average updates per synchronization
            - sync_interval: Synchronization interval
        
        Example:
            >>> stats = actor_learner.get_update_stats()
            >>> print(f"Total updates: {stats['update_count']}")
            >>> print(f"Updates per sync: {stats['updates_per_sync']:.2f}")
        """
        return {
            'update_count': self.update_count,
            'sync_count': self.sync_count,
            'updates_per_sync': self.update_count / max(self.sync_count, 1),
            'sync_interval': self.sync_interval,
        }
    
    # ==================== Utility Methods ====================
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"{self.__class__.__name__}(\n"
            f"  acting_agent={self.acting_agent.__class__.__name__},\n"
            f"  learning_agent={self.learning_agent.__class__.__name__},\n"
            f"  sync_interval={self.sync_interval},\n"
            f"  update_count={self.update_count},\n"
            f"  sync_count={self.sync_count}\n"
            f")"
        )
