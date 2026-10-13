#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

"""
Reset Manager Module
Centralized reset management - receives all reset requests and forwards to Reset Coordinator
"""

import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from typing import Tuple, Dict, Optional
from edge_drl_student.common.logger import Logger
from edge_drl_student.utils.utils import ResetReason



class ResetManager:
    """
    Centralized Reset Manager
    
    Receives ALL reset requests (from agent, other nodes, command line)
    and forwards them to Reset Coordinator for execution.
    
    Architecture:
        ┌─────────────────────────────────────────────────┐
        │  Agent / Other Nodes / Command Line             │
        └────────────────────┬────────────────────────────┘
                             │ call /edge/reset
                             ▼
        ┌─────────────────────────────────────────────────┐
        │    DRL Student Node (ResetManager)              │
        │    ┌────────────────────────────────────────┐   │
        │    │  Service Server: /edge/reset           │   │
        │    │  - Receives ALL reset requests         │   │
        │    │  - Logs reason and statistics          │   │
        │    │  - Manages reset state                 │   │
        │    └──────────────┬─────────────────────────┘   │
        │                   │                             │
        │    ┌──────────────▼─────────────────────────┐   │
        │    │ Service Client: /edge/reset/gz_handler │   │
        │    │  - Forwards to Gz Reset Handler        │   │
        │    └──────────────┬─────────────────────────┘   │
        └───────────────────┼─────────────────────────────┘
                            │ call /edge/reset/gz_handler
                            ▼
        ┌─────────────────────────────────────────────────┐
        │         Gz Reset Handler (gazebo_resetter)      │
        │  1. Reset Controller                            │
        │  2. Reset Gazebo (pause→reset→unpause)          │
        │  3. Start Controller                            │
        └─────────────────────────────────────────────────┘
    
    Example:
        >>> # In DRL Student Node
        >>> manager = ResetManager(
        ...     node=self,
        ...     reset_timeout=20.0,
        ...     logger=logger,
        ...     reset_service_name='/edge/reset',
        ...     gz_reset_service_name='/edge/reset/gz_handler',
        ... )
        >>> 
        >>> # Agent calls internally with enum
        >>> success, msg = manager.call_reset_service(ResetReason.FALL_DETECTED)
        >>> success, msg = manager.call_reset_service(ResetReason.EPISODE_DONE)
        >>> 
        >>> # Or with string (auto-converted)
        >>> success, msg = manager.call_reset_service('fall_detected')
        >>> 
        >>> # Other nodes call via service
        >>> # ros2 service call /edge/reset std_srvs/srv/Trigger
    """
    
    def __init__(
        self,
        node: Node,
        reset_timeout: float,
        logger: Logger,
        reset_service_name: str = '/edge/reset',
        gz_reset_service_name: str = '/edge/reset/gz_handler',
    ) -> None:
        """
        Initialize Reset Manager
        
        Args:
            node: ROS2 node instance
            reset_timeout: Service call timeout in seconds
            logger: Logger instance
            reset_service_name: Service name for receiving reset requests (server)
            gz_reset_service_name: Service name for calling Gz Reset Handler (client)
        """
        self.node: Node = node
        self.reset_timeout: float = reset_timeout
        self.logger: Logger = logger
        self.reset_service_name: str = reset_service_name
        self.gz_reset_service_name: str = gz_reset_service_name
        
        # ========== Service Server (Receive all reset requests) ==========
        node.create_service(Trigger, reset_service_name, self._reset_service_callback)
        
        # ========== Service Client (Call Gz Reset Handler) ==========
        self.gz_reset_client = node.create_client(Trigger, gz_reset_service_name)
        
        # Statistics
        self.reset_count: int = 0
        self.reset_success_count: int = 0
        self.reset_failure_count: int = 0
        self.is_resetting: bool = False
        
        # Reset reason statistics (using Enum)
        self.reset_reasons: Dict[ResetReason, int] = {
            reason: 0 for reason in ResetReason
        }
        
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"✅ Reset Manager initialized\n"
            f"{'='*70}\n"
            f"📡 Services:\n"
            f"   Server (receive):  {reset_service_name}\n"
            f"   Client (forward):  {gz_reset_service_name}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"⚙️  Configuration:\n"
            f"   Timeout: {reset_timeout}s\n"
            f"{'='*70}\n"
            f"✅ Ready to receive reset requests on {reset_service_name}\n"
            f"{'='*70}"
        )
    
    # ==================== Service Server Callback ====================
    def _reset_service_callback(
        self,
        request: Trigger.Request,
        response: Trigger.Response
    ) -> Trigger.Response:
        """
        Service callback for reset requests from external nodes
        
        This is called when other nodes call /edge/reset
        
        Args:
            request: Trigger request (empty)
            response: Trigger response
            
        Returns:
            Trigger.Response: Response with success status and message
        """
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"🔔 External reset request received via {self.reset_service_name}\n"
            f"{'='*70}"
        )
        
        # Call internal reset with 'external' reason
        success: bool
        message: str
        success, message = self._perform_reset(reason=ResetReason.EXTERNAL)
        
        response.success = success
        response.message = message
        
        return response
    
    # ==================== Internal Reset Methods ====================
    def call_reset_service(
        self, 
        reason: 'ResetReason | str' = ResetReason.UNKNOWN
    ) -> Tuple[bool, str]:
        """
        Call reset service (internal use by agent)
        
        This is called internally by DRL Student for episode_done, fall_detected, etc.
        
        Args:
            reason: Reset reason (ResetReason enum or string)
            
        Returns:
            Tuple[bool, str]: (success, message)
        
        Example:
            >>> # Using enum (recommended)
            >>> success, msg = manager.call_reset_service(ResetReason.EPISODE_DONE)
            >>> success, msg = manager.call_reset_service(ResetReason.FALL_DETECTED)
            >>> 
            >>> # Using string (auto-converted)
            >>> success, msg = manager.call_reset_service('episode_done')
            >>> success, msg = manager.call_reset_service('fall_detected')
        """
        # Convert string to enum if needed
        if isinstance(reason, str):
            reason = ResetReason.from_string(reason)
        
        return self._perform_reset(reason=reason)
    
    def _perform_reset(self, reason: ResetReason) -> Tuple[bool, str]:
        """
        Perform reset by calling Gz Reset Handler service
        
        Args:
            reason: Reset reason (ResetReason enum)
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if self.is_resetting:
            warning_msg = "Reset already in progress, skipping..."
            self.node.get_logger().warning(f'⚠️  {warning_msg}')
            return False, warning_msg
        
        self.is_resetting = True
        self.reset_count += 1
        
        # Update reason statistics (using enum as key)
        self.reset_reasons[reason] += 1
        
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"🔄 [Reset #{self.reset_count}] Starting reset sequence\n"
            f"{'─'*70}\n"
            f"   Reason: {reason.name} ({reason.value})\n"
            f"   Target: {self.gz_reset_service_name}\n"
            f"   Timeout: {self.reset_timeout}s\n"
            f"{'='*70}"
        )
        
        reset_start: float = time.time()
        
        # Wait for Reset Coordinator service
        self.node.get_logger().info('⏳ Waiting for Gz Reset Handler service...')
        if not self.gz_reset_client.wait_for_service(timeout_sec=2.0):
            error_msg: str = f'Gz Reset Handler service not available after 2.0s'
            self.node.get_logger().error(f'❌ {error_msg}')
            self.reset_failure_count += 1
            self.is_resetting = False

            elapsed_time: float = time.time() - reset_start
            self._log_reset_result(success=False, reason=reason, elapsed_time=elapsed_time)
            return False, error_msg
        
        # Create and send request
        request = Trigger.Request()
        
        try:
            self.node.get_logger().info('📞 Calling Gz Reset Handler...')
            future = self.gz_reset_client.call_async(request)
            
            # Polling with timeout and progress display
            start_time: float = time.time()
            last_progress_time: float = start_time
            
            while not future.done():
                elapsed: float = time.time() - start_time
                
                if elapsed > self.reset_timeout:
                    error_msg = f'Gz Reset Handler timeout after {elapsed:.2f}s'
                    self.node.get_logger().error(f'❌ {error_msg}')
                    self.reset_failure_count += 1
                    elapsed_time: float = time.time() - reset_start
                    self._log_reset_result(success=False, reason=reason, elapsed_time=elapsed_time)
                    self.is_resetting = False
                    return False, error_msg
                
                # Show progress every 2 seconds
                if time.time() - last_progress_time >= 2.0:
                    self.node.get_logger().info(
                        f'   ⏳ Waiting... ({elapsed:.1f}s / {self.reset_timeout:.1f}s)'
                    )
                    last_progress_time = time.time()
                
                time.sleep(0.01)
            
            elapsed_time: float = time.time() - reset_start
            
            # Get result
            response: Trigger.Response = future.result()
            
            if response is None:
                error_msg = 'Reset Coordinator returned None response'
                self.node.get_logger().error(f'❌ {error_msg}')
                self.reset_failure_count += 1
                self._log_reset_result(success=False, reason=reason, elapsed_time=elapsed_time)
                self.is_resetting = False
                return False, error_msg
            
            # Handle success/failure
            if response.success:
                self.reset_success_count += 1
                success_rate = self.reset_success_count / self.reset_count * 100
                
                self.node.get_logger().info(
                    f"\n{'='*70}\n"
                    f"✅ [Reset #{self.reset_count}] SUCCESS\n"
                    f"{'─'*70}\n"
                    f"   Reason: {reason.name}\n"
                    f"   Duration: {elapsed_time:.3f}s\n"
                    f"   Message: {response.message}\n"
                    f"   Success Rate: {self.reset_success_count}/{self.reset_count} ({success_rate:.1f}%)\n"
                    f"{'='*70}\n"
                )
                self._log_reset_result(success=True, reason=reason, elapsed_time=elapsed_time)
                self.is_resetting = False
                return True, response.message
            else:
                self.reset_failure_count += 1
                success_rate = self.reset_success_count / self.reset_count * 100
                
                self.node.get_logger().error(
                    f"\n{'='*70}\n"
                    f"❌ [Reset #{self.reset_count}] FAILED\n"
                    f"{'─'*70}\n"
                    f"   Reason: {reason.name}\n"
                    f"   Duration: {elapsed_time:.3f}s\n"
                    f"   Error: {response.message}\n"
                    f"   Success Rate: {self.reset_success_count}/{self.reset_count} ({success_rate:.1f}%)\n"
                    f"{'='*70}\n"
                )
                self._log_reset_result(success=False, reason=reason, elapsed_time=elapsed_time)
                self.is_resetting = False
                return False, response.message
                
        except Exception as e:
            error_msg = f'Exception during Reset Coordinator call: {type(e).__name__}: {str(e)}'
            self.node.get_logger().error(f'❌ {error_msg}')

            self._log_reset_result(success=False, reason=reason, elapsed_time=elapsed_time)
            self.is_resetting = False
            
            # Print traceback
            import traceback
            self.node.get_logger().error('Traceback:')
            for line in traceback.format_exc().split('\n'):
                if line:
                    self.node.get_logger().error(f'  {line}')
            
            return False, error_msg
    
    # ====================  Logging  ====================
    def _log_reset_result(
        self, 
        success: bool, 
        reason: ResetReason,
        elapsed_time: Optional[float] = None
    ) -> None:
        """
        Log reset result (success or failure)
        
        Args:
            success: Whether reset was successful
            reason: Reset reason (ResetReason enum)
            elapsed_time: Time taken for reset (only for successful resets)
        
        Example:
            >>> # Log success
            >>> self._log_reset_result(True, ResetReason.EPISODE_DONE, elapsed_time=1.5)
            >>> 
            >>> # Log failure
            >>> self._log_reset_result(False, ResetReason.FALL_DETECTED)
        """
        # Log success/failure
        self.logger.writer.add_scalar('Reset/success', 1 if success else 0, self.reset_count)
        
        # Log elapsed time
        if elapsed_time is not None:
            self.logger.writer.add_scalar('Reset/time', elapsed_time, self.reset_count)
        
        # Log reason
        # self.logger.writer.add_scalar(f'Reset/reason_{reason.name.lower()}', 1, self.reset_count)
        
        # Calculate and log rates
        success_rate = self.reset_success_count / self.reset_count if self.reset_count > 0 else 0.0

        self.logger.writer.add_scalar('Reset/success_rate', success_rate, self.reset_count)
        
        # Log reason counts
        for r, count in self.reset_reasons.items():
            self.logger.writer.add_scalar(f'Reset/count/{r.name.lower()}', count, self.reset_count)
    
    # ==================== Statistics ====================
    def get_statistics(self) -> Dict[str, any]:
        """
        Get reset statistics
        
        Returns:
            Dict[str, any]: Statistics dictionary containing:
                - total: Total reset count
                - success: Successful reset count
                - failure: Failed reset count
                - success_rate: Success rate (0.0 to 1.0)
                - reasons: Dictionary of reset counts by reason name
        
        Example:
            >>> stats = manager.get_statistics()
            >>> print(f"Total resets: {stats['total']}")
            >>> print(f"Success rate: {stats['success_rate']:.2%}")
            >>> print(f"Reasons: {stats['reasons']}")
            >>> # Output:
            >>> # Total resets: 10
            >>> # Success rate: 100.00%
            >>> # Reasons: {'episode_done': 5, 'fall_detected': 3, 'timeout': 2, ...}
        """
        success_rate: float = (
            self.reset_success_count / self.reset_count 
            if self.reset_count > 0 else 0.0
        )
        
        # ⭐ Convert enum keys to string for easier access
        reasons_dict = {
            reason.name.lower(): count 
            for reason, count in self.reset_reasons.items()
        }
        
        return {
            'total': self.reset_count,
            'success': self.reset_success_count,
            'failure': self.reset_failure_count,
            'success_rate': success_rate,
            'reasons': reasons_dict
        }
    
    def reset_statistics(self) -> None:
        """
        Reset all statistics counters
        
        Example:
            >>> manager.reset_statistics()
            >>> stats = manager.get_statistics()
            >>> assert stats['total'] == 0
        """
        self.reset_count = 0
        self.reset_success_count = 0
        self.reset_failure_count = 0
        
        # ⭐ Reset all enum-based counters
        self.reset_reasons = {reason: 0 for reason in ResetReason}
        
        self.node.get_logger().info("📊 Reset statistics cleared")
    
    def print_statistics(self) -> None:
        """
        Print detailed statistics
        
        Example:
            >>> manager.print_statistics()
        """
        stats = self.get_statistics()
        
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"📊 Reset Statistics\n"
            f"{'='*70}\n"
            f"Total Resets: {stats['total']}\n"
            f"  ✅ Success: {stats['success']}\n"
            f"  ❌ Failure: {stats['failure']}\n"
            f"  📈 Success Rate: {stats['success_rate']*100:.1f}%\n"
            f"{'─'*70}\n"
            f"Reset Reasons:\n"
            f"  Episode Done:   {stats['reasons']['episode_done']}\n"
            f"  Fall Detected:  {stats['reasons']['fall_detected']}\n"
            f"  Manual:         {stats['reasons']['manual']}\n"
            f"  Timeout:        {stats['reasons']['timeout']}\n"
            f"  External:       {stats['reasons']['external']}\n"
            f"  Unknown:        {stats['reasons']['unknown']}\n"
            f"{'='*70}"
        )
    
    def get_reason_count(self, reason: 'ResetReason | str') -> int:
        """
        Get count for specific reset reason
        
        Args:
            reason: Reset reason (enum or string)
            
        Returns:
            int: Count for this reason
            
        Example:
            >>> count = manager.get_reason_count(ResetReason.FALL_DETECTED)
            >>> count = manager.get_reason_count('fall_detected')
        """
        if isinstance(reason, str):
            reason = ResetReason.from_string(reason)
        
        return self.reset_reasons.get(reason, 0)
    
    def __repr__(self) -> str:
        """String representation"""
        stats = self.get_statistics()
        return (
            f"ResetManager(\n"
            f"  server='{self.reset_service_name}',\n"
            f"  gz_reset_handler='{self.gz_reset_service_name}',\n"
            f"  timeout={self.reset_timeout}s,\n"
            f"  statistics={stats}\n"
            f")"
        )