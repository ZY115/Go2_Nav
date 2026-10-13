#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
import time
import rclpy
import subprocess
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Empty, Trigger
from pathlib import Path  


class GzResetHandlerNode(Node):
    """
    Gz Reset Handler Node
    
    Receives reset request from DRL student via /edge/reset,
    then coordinates reset of both Controller and Gazebo,
    and optionally starts the controller after reset.
    """
    
    def __init__(self):
        super().__init__('gz_reset_handler_node')
        
        # Use reentrant callback group
        self.callback_group = ReentrantCallbackGroup()
        
        # ========== Parameter Declaration ==========
        # Gazebo parameters
        self.declare_parameter('world_name', 'openwild_outdoor.world')
        self.declare_parameter('robot_name', 'go2')
        self.declare_parameter('reset_position_x', 0.0)
        self.declare_parameter('reset_position_y', 0.0)
        self.declare_parameter('reset_position_z', 0.3)
        self.declare_parameter('reset_orientation_x', 0.0)
        self.declare_parameter('reset_orientation_y', 0.0)
        self.declare_parameter('reset_orientation_z', 0.0)
        self.declare_parameter('reset_orientation_w', 1.0)
        self.declare_parameter('service_timeout', 3000)
        self.declare_parameter('reset_world', False)
        self.declare_parameter('reset_robot_pose', True)
        self.declare_parameter('wait_after_reset', 0.3)
        
        self.declare_parameter('reset_attempts', 3)  # Increase attempts for pose reset
        self.declare_parameter('wait_between_attempts', 0.3)
        self.declare_parameter('clear_velocities', True)
        self.declare_parameter('pause_physics_during_reset', True)
        self.declare_parameter('wait_after_pause', 0.2)
        self.declare_parameter('wait_after_world_reset', 0.3)
        self.declare_parameter('wait_after_pose_reset', 0.2)
        
        # Controller parameters
        self.declare_parameter('controller_reset_service', '/edge/reset/controller')
        self.declare_parameter('controller_start_service', '/edge/launch/controller') 
        self.declare_parameter('controller_reset_timeout', 5.0)
        self.declare_parameter('controller_start_timeout', 5.0)  
        self.declare_parameter('reset_controller', True)
        self.declare_parameter('reset_order', 'controller_first')
        
        # Start behavior parameters
        self.declare_parameter('start_after_reset', True)
        self.declare_parameter('delay_after_reset', 1.0)
        
        # Get parameters
        world_name_full = self.get_parameter('world_name').value 
        self.world_name = Path(world_name_full).stem
        self.robot_name = self.get_parameter('robot_name').value
        self.reset_pos_x = self.get_parameter('reset_position_x').value
        self.reset_pos_y = self.get_parameter('reset_position_y').value
        self.reset_pos_z = self.get_parameter('reset_position_z').value
        self.reset_ori_x = self.get_parameter('reset_orientation_x').value
        self.reset_ori_y = self.get_parameter('reset_orientation_y').value
        self.reset_ori_z = self.get_parameter('reset_orientation_z').value
        self.reset_ori_w = self.get_parameter('reset_orientation_w').value
        self.timeout = self.get_parameter('service_timeout').value
        self.reset_world_flag = self.get_parameter('reset_world').value
        self.reset_robot_pose_flag = self.get_parameter('reset_robot_pose').value
        self.wait_after_reset = self.get_parameter('wait_after_reset').value
        
        self.reset_attempts = self.get_parameter('reset_attempts').value
        self.wait_between_attempts = self.get_parameter('wait_between_attempts').value
        self.clear_velocities = self.get_parameter('clear_velocities').value
        self.pause_physics = self.get_parameter('pause_physics_during_reset').value
        self.wait_after_pause = self.get_parameter('wait_after_pause').value
        self.wait_after_world_reset = self.get_parameter('wait_after_world_reset').value
        self.wait_after_pose_reset = self.get_parameter('wait_after_pose_reset').value
        
        self.controller_reset_service_name = self.get_parameter('controller_reset_service').value
        self.controller_start_service_name = self.get_parameter('controller_start_service').value
        self.controller_reset_timeout = self.get_parameter('controller_reset_timeout').value
        self.controller_start_timeout = self.get_parameter('controller_start_timeout').value
        self.reset_controller_flag = self.get_parameter('reset_controller').value
        self.reset_order = self.get_parameter('reset_order').value
        
        self.start_after_reset = self.get_parameter('start_after_reset').value
        self.delay_after_reset = self.get_parameter('delay_after_reset').value
        
        # ========== Create Main Service ==========
        self.edge_reset_srv = self.create_service(
            Trigger,
            '/edge/reset/gz_handler',
            self.edge_reset_callback,
            callback_group=self.callback_group
        )
        
        # ========== Create Service Clients ==========
        self.controller_reset_client = self.create_client(
            Trigger,
            self.controller_reset_service_name,
            callback_group=self.callback_group
        )
        
        self.controller_start_client = self.create_client(
            Trigger,
            self.controller_start_service_name,
            callback_group=self.callback_group
        )
        
        self.gz_reset_client = self.create_client(
            Empty,
            '/reset_world',
            callback_group=self.callback_group
        )
        
        # ========== Statistics ==========
        self.reset_count = 0
        self.last_reset_time = None
        self.failed_resets = 0
        self.controller_reset_failures = 0
        self.gazebo_reset_failures = 0
        self.controller_start_failures = 0
        
        # ========== Log Output ==========
        self.get_logger().info('=' * 70)
        self.get_logger().info('🎯 Reset Coordinator Node Initialized')
        self.get_logger().info('=' * 70)
        self.get_logger().info('📡 Services:')
        self.get_logger().info('  Main: /edge/reset/gz_handler (Trigger)')
        self.get_logger().info(f'  Controller Reset: {self.controller_reset_service_name}')
        self.get_logger().info(f'  Controller Start: {self.controller_start_service_name}')
        self.get_logger().info('─' * 70)
        self.get_logger().info('⚙️  Configuration:')
        self.get_logger().info(f'  Reset Controller: {self.reset_controller_flag}')
        self.get_logger().info(f'  Reset Order: {self.reset_order}')
        self.get_logger().info(f'  Start After Reset: {self.start_after_reset}')
        self.get_logger().info(f'  Delay After Reset: {self.delay_after_reset}s')
        self.get_logger().info(f'  Reset Attempts: {self.reset_attempts}')
        self.get_logger().info('=' * 70)
        self.get_logger().info('✅ Ready to receive reset requests on /edge/reset/gz_handler')
        self.get_logger().info('=' * 70)

    def edge_reset_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Main reset service callback"""
        self.get_logger().info("===== RESET CALLBACK ENTERED =====")

        self.get_logger().info('')
        self.get_logger().info('═' * 70)
        self.get_logger().info('🔄 [RESET REQUEST] Received from DRL student on /edge/reset/gz_handler')
        self.get_logger().info('═' * 70)
        
        success, message = self._perform_coordinated_reset()
        
        response.success = success
        response.message = message
        
        if success:
            self.get_logger().info('═' * 70)
            self.get_logger().info(f'✅ [RESET SUCCESS] {message}')
            self.get_logger().info('═' * 70)
        else:
            self.get_logger().error('═' * 70)
            self.get_logger().error(f'❌ [RESET FAILED] {message}')
            self.get_logger().error('═' * 70)
        
        self.get_logger().info('')
        return response
    
    def _perform_coordinated_reset(self):
        """
        Coordinate reset of controller and Gazebo, then optionally start controller
        
        Flow:
        1. Reset Controller
        2. Reset Gazebo 
        3. Wait delay_after_reset
        4. Start Controller (if start_after_reset is True)
        """
        start_time = time.time()
        self.reset_count += 1
        
        controller_reset_duration = 0.0
        gazebo_duration = 0.0
        controller_start_duration = 0.0
        
        try:
            # ========== Step 1 & 2: Reset Controller and Gazebo ==========
            if self.reset_order == 'controller_first':
                # Reset Controller
                if self.reset_controller_flag:
                    ctrl_start = time.time()
                    controller_success, controller_message = self._reset_controller()
                    controller_reset_duration = time.time() - ctrl_start
                    
                    if not controller_success:
                        self.controller_reset_failures += 1
                        self.failed_resets += 1
                        return False, f'Controller reset failed: {controller_message}'
                
                time.sleep(5)  # Small delay between resets
                
                # Reset Gazebo
                gz_start = time.time()
                gazebo_success, gazebo_message = self._reset_gazebo()
                gazebo_duration = time.time() - gz_start
                
                if not gazebo_success:
                    self.gazebo_reset_failures += 1
                    self.failed_resets += 1
                    return False, f'Gazebo reset failed: {gazebo_message}'
            
            else:  # gazebo_first
                # Reset Gazebo
                gz_start = time.time()
                gazebo_success, gazebo_message = self._reset_gazebo()
                gazebo_duration = time.time() - gz_start
                
                if not gazebo_success:
                    self.gazebo_reset_failures += 1
                    self.failed_resets += 1
                    return False, f'Gazebo reset failed: {gazebo_message}'
                
                # Reset Controller
                if self.reset_controller_flag:
                    ctrl_start = time.time()
                    controller_success, controller_message = self._reset_controller()
                    controller_reset_duration = time.time() - ctrl_start
                    
                    if not controller_success:
                        self.controller_reset_failures += 1
                        self.failed_resets += 1
                        return False, f'Controller reset failed: {controller_message}'
            
            # ========== Step 3: Wait after reset ==========
            if self.start_after_reset and self.delay_after_reset > 0:
                self.get_logger().info('')
                self.get_logger().info(f'⏳ Waiting {self.delay_after_reset}s before starting controller...')
                time.sleep(self.delay_after_reset)
            
            # ========== Step 4: Start Controller ==========
            if self.start_after_reset:
                self.get_logger().info('')
                start_begin = time.time()
                start_success, start_message = self._start_controller()
                controller_start_duration = time.time() - start_begin
                
                if not start_success:
                    self.controller_start_failures += 1
                    self.failed_resets += 1
                    return False, f'Controller start failed: {start_message}'
            
            # ========== Success ==========
            elapsed_time = time.time() - start_time
            self.last_reset_time = time.time()
            
            # Build success message
            success_parts = []
            if self.reset_controller_flag:
                success_parts.append(f'Reset Controller: {controller_reset_duration:.3f}s')
            success_parts.append(f'Reset Gazebo: {gazebo_duration:.3f}s')
            if self.start_after_reset:
                success_parts.append(f'Start Controller: {controller_start_duration:.3f}s')
            
            success_msg = (
                f'Reset #{self.reset_count} completed in {elapsed_time:.3f}s '
                f'[{", ".join(success_parts)}]'
            )
            
            return True, success_msg
            
        except Exception as e:
            self.failed_resets += 1
            error_msg = f'Reset #{self.reset_count} failed: {str(e)}'
            self.get_logger().error(error_msg)
            
            import traceback
            for line in traceback.format_exc().split('\n'):
                if line:
                    self.get_logger().error(f'  {line}')
            
            return False, error_msg
    
    def _reset_controller(self):
        """Reset the edge controller"""
        self.get_logger().info(f'  [1/3] 🤖 Resetting Controller ({self.controller_reset_service_name})...')
        
        if not self.controller_reset_client.wait_for_service(timeout_sec=2.0):
            error_msg = f'Controller reset service not available'
            self.get_logger().error(f'    ❌ {error_msg}')
            return False, error_msg
        
        try:
            request = Trigger.Request()
            self.get_logger().info(f'    📞 Calling reset service...')
            call_start = time.time()
            
            future = self.controller_reset_client.call_async(request)
            
            timeout_end = time.time() + self.controller_reset_timeout
            while not future.done():
                if time.time() > timeout_end:
                    error_msg = f'Timeout after {self.controller_reset_timeout:.3f}s'
                    self.get_logger().error(f'    ❌ {error_msg}')
                    return False, error_msg
                time.sleep(0.01)
            
            call_duration = time.time() - call_start
            
            try:
                response : Trigger.Response = future.result()
            except Exception as e:
                error_msg = f'Failed to get response: {str(e)}'
                self.get_logger().error(f'    ❌ {error_msg}')
                return False, error_msg
            
            if response is None:
                error_msg = 'Service returned None'
                self.get_logger().error(f'    ❌ {error_msg}')
                return False, error_msg
            
            if response.success:
                self.get_logger().info(f'    ✅ Controller reset successful ({call_duration:.3f}s)')
                if response.message:
                    self.get_logger().info(f'       Response: "{response.message}"')
                return True, response.message
            else:
                error_msg = f'Controller returned failure: {response.message}'
                self.get_logger().error(f'    ❌ {error_msg}')
                return False, error_msg
                
        except Exception as e:
            error_msg = f'Exception: {str(e)}'
            self.get_logger().error(f'    ❌ {error_msg}')
            return False, error_msg
    
    def _start_controller(self):
        """Start the edge controller"""
        self.get_logger().info(f'  [3/3] ▶️  Starting Controller ({self.controller_start_service_name})...')
        
        if not self.controller_start_client.wait_for_service(timeout_sec=2.0):
            error_msg = f'Controller start service not available'
            self.get_logger().error(f'    ❌ {error_msg}')
            return False, error_msg
        
        try:
            request = Trigger.Request()
            self.get_logger().info(f'    📞 Calling start service...')
            call_start = time.time()
            
            future = self.controller_start_client.call_async(request)
            
            timeout_end = time.time() + self.controller_start_timeout
            while not future.done():
                if time.time() > timeout_end:
                    error_msg = f'Timeout after {self.controller_start_timeout:.3f}s'
                    self.get_logger().error(f'    ❌ {error_msg}')
                    return False, error_msg
                time.sleep(0.01)
            
            call_duration = time.time() - call_start
            
            try:
                response : Trigger.Response = future.result()
            except Exception as e:
                error_msg = f'Failed to get response: {str(e)}'
                self.get_logger().error(f'    ❌ {error_msg}')
                return False, error_msg
            
            if response is None:
                error_msg = 'Service returned None'
                self.get_logger().error(f'    ❌ {error_msg}')
                return False, error_msg
            
            if response.success:
                self.get_logger().info(f'    ✅ Controller started successfully ({call_duration:.3f}s)')
                if response.message:
                    self.get_logger().info(f'       Response: "{response.message}"')
                return True, response.message
            else:
                error_msg = f'Controller returned failure: {response.message}'
                self.get_logger().error(f'    ❌ {error_msg}')
                return False, error_msg
                
        except Exception as e:
            error_msg = f'Exception: {str(e)}'
            self.get_logger().error(f'    ❌ {error_msg}')
            return False, error_msg
    
    def _reset_gazebo(self):
        """
        Reset Gazebo simulation using the CORRECT sequence:
        1. Pause physics
        2. Clear velocities
        3. Reset world (optional)
        4. Reset pose
        5. Clear velocities again
        6. Resume physics
        """
        self.get_logger().info('  [2/3] 🌍 Resetting Gazebo...')
        
        gazebo_start = time.time()
        step_times = {}
        
        try:
            # ========== Step 1: Pause Physics ==========
            if self.pause_physics:
                self.get_logger().info('    [1/6] ⏸️  Pausing physics...')
                step_start = time.time()
                if not self._pause_physics(True):
                    self.get_logger().warn('      ⚠️  Failed to pause physics (continuing anyway)')
                else:
                    step_times['pause'] = time.time() - step_start
                    self.get_logger().info(f'      ✓ Physics paused ({step_times["pause"]*1000:.0f}ms)')
                time.sleep(self.wait_after_pause)
            
            # ========== Step 2: Clear Velocities (First Time) ==========
            if self.clear_velocities:
                self.get_logger().info('    [2/6] 🧹 Clearing velocities (1st)...')
                step_start = time.time()
                if self._clear_robot_velocities():
                    step_times['clear_vel_1'] = time.time() - step_start
                    self.get_logger().info(f'      ✓ Velocities cleared ({step_times["clear_vel_1"]*1000:.0f}ms)')
                else:
                    self.get_logger().warn('      ⚠️  Failed to clear velocities (continuing anyway)')
                time.sleep(0.1)
            
            # ========== Step 3: Reset World (Optional) ==========
            if self.reset_world_flag:
                self.get_logger().info('    [3/6] 🌍 Resetting world...')
                step_start = time.time()
                if not self._reset_gazebo_world():
                    self.get_logger().warn('      ⚠️  World reset failed (continuing with pose reset)')
                else:
                    step_times['world'] = time.time() - step_start
                    self.get_logger().info(f'      ✓ World reset ({step_times["world"]*1000:.0f}ms)')
                time.sleep(self.wait_after_world_reset)
            
            # ========== Step 4: Reset Robot Pose (with retry) ==========
            if self.reset_robot_pose_flag:
                self.get_logger().info('    [4/6] 🤖 Resetting robot pose...')
                step_start = time.time()
                pose_success = False
                
                for attempt in range(self.reset_attempts):
                    if self._reset_robot_pose():
                        pose_success = True
                        step_times['pose'] = time.time() - step_start
                        self.get_logger().info(f'      ✓ Pose reset ({step_times["pose"]*1000:.0f}ms)')
                        break
                    else:
                        if attempt < self.reset_attempts - 1:
                            self.get_logger().warn(f'      ⚠️  Attempt {attempt+1}/{self.reset_attempts} failed, retrying...')
                            time.sleep(self.wait_between_attempts)
                        else:
                            self.get_logger().error(f'      ❌ All {self.reset_attempts} attempts failed')
                
                if not pose_success:
                    # Terminate by unpausing physics if needed
                    if self.pause_physics:
                        self._pause_physics(False)
                    return False, f'Failed to reset robot pose after {self.reset_attempts} attempts'
                
                time.sleep(self.wait_after_pose_reset)
            
            # ========== Step 5: Clear Velocities (Second Time) ==========
            if self.clear_velocities:
                self.get_logger().info('    [5/6] 🧹 Clearing velocities (2nd)...')
                step_start = time.time()
                if self._clear_robot_velocities():
                    step_times['clear_vel_2'] = time.time() - step_start
                    self.get_logger().info(f'      ✓ Velocities cleared ({step_times["clear_vel_2"]*1000:.0f}ms)')
                else:
                    self.get_logger().warn('      ⚠️  Failed to clear velocities')
                time.sleep(0.1)
            
            # ========== Step 6: Resume Physics ==========
            if self.pause_physics:
                self.get_logger().info('    [6/6] ▶️  Resuming physics...')
                step_start = time.time()
                if not self._pause_physics(False):
                    self.get_logger().error('      ❌ Failed to resume physics')
                    return False, 'Failed to resume physics'
                else:
                    step_times['unpause'] = time.time() - step_start
                    self.get_logger().info(f'      ✓ Physics resumed ({step_times["unpause"]*1000:.0f}ms)')
                time.sleep(self.wait_after_reset)
            
            # ========== Success ==========
            gazebo_duration = time.time() - gazebo_start
            
            timing_parts = []
            for key, value in step_times.items():
                timing_parts.append(f'{key}:{value*1000:.0f}ms')
            timing_str = ', '.join(timing_parts) if timing_parts else 'N/A'
            
            success_msg = f'Gazebo reset completed in {gazebo_duration:.3f}s [{timing_str}]'
            self.get_logger().info(f'    ✅ {success_msg}')
            
            return True, success_msg
            
        except Exception as e:
            # Make sure to unpause physics on exception
            if self.pause_physics:
                try:
                    self._pause_physics(False)
                except:
                    pass
            
            error_msg = f'Gazebo reset exception: {str(e)}'
            self.get_logger().error(f'    ❌ {error_msg}')
            
            import traceback
            self.get_logger().error('    Traceback:')
            for line in traceback.format_exc().split('\n'):
                if line:
                    self.get_logger().error(f'      {line}')
            
            return False, error_msg
    
    def _pause_physics(self, pause=True):
        """Pause or unpause physics simulation"""
        try:
            service_name = f'/world/{self.world_name}/control'
            pause_str = 'true' if pause else 'false'
            
            cmd = [
                'ign', 'service',
                '-s', service_name,
                '--reqtype', 'ignition.msgs.WorldControl',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', str(self.timeout),
                '--req', f'pause: {pause_str}'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout / 1000.0 + 2.0
            )
            
            success = result.returncode == 0
            
            if not success and result.stderr:
                self.get_logger().debug(f'Pause/unpause stderr: {result.stderr[:100]}')
            
            return success
            
        except subprocess.TimeoutExpired:
            self.get_logger().error(f'Timeout while trying to {"pause" if pause else "unpause"} physics')
            return False
        except Exception as e:
            self.get_logger().debug(f'Failed to {"pause" if pause else "unpause"} physics: {e}')
            return False
    
    def _clear_robot_velocities(self):
        """Clear robot linear and angular velocities"""
        try:
            service_name = f'/world/{self.world_name}/set_entity_velocity'
            
            velocity_msg = (
                f"entity: {{name: '{self.robot_name}', type: MODEL}}, "
                f"linear_velocity: {{x: 0.0, y: 0.0, z: 0.0}}, "
                f"angular_velocity: {{x: 0.0, y: 0.0, z: 0.0}}"
            )
            
            cmd = [
                'ign', 'service',
                '-s', service_name,
                '--reqtype', 'ignition.msgs.EntityVelocity',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', str(self.timeout),
                '--req', velocity_msg
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout / 1000.0 + 2.0
            )
            
            success = result.returncode == 0
            
            if not success and result.stderr:
                self.get_logger().debug(f'Clear velocity stderr: {result.stderr[:100]}')
            
            return success
            
        except subprocess.TimeoutExpired:
            self.get_logger().debug('Timeout while clearing velocities')
            return False
        except Exception as e:
            self.get_logger().debug(f'Failed to clear velocities: {e}')
            return False
    
    def _reset_gazebo_world(self):
        """Reset Gazebo world"""
        # Try ROS2 service first
        if self.gz_reset_client.wait_for_service(timeout_sec=1.0):
            try:
                future = self.gz_reset_client.call_async(Empty.Request())
                
                timeout_end = time.time() + 5.0
                while not future.done():
                    if time.time() > timeout_end:
                        break
                    time.sleep(0.01)
                
                if future.done():
                    try:
                        result = future.result()
                        if result is not None:
                            return True
                    except:
                        pass
                        
            except Exception as e:
                self.get_logger().debug(f'ROS2 service error: {e}')
        
        # Fallback to ign command
        try:
            service_name = f'/world/{self.world_name}/control'
            cmd = [
                'ign', 'service',
                '-s', service_name,
                '--reqtype', 'ignition.msgs.WorldControl',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', str(self.timeout),
                '--req', 'reset: {all: true}'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout / 1000.0 + 2.0
            )
            
            return result.returncode == 0
                
        except subprocess.TimeoutExpired:
            self.get_logger().debug('World reset timeout')
            return False
        except Exception as e:
            self.get_logger().debug(f'World reset exception: {e}')
            return False
    
    def _reset_robot_pose(self):
        """Reset robot pose to initial position"""
        try:
            service_name = f'/world/{self.world_name}/set_pose'
            
            pose_msg = (
                f"name: '{self.robot_name}', "
                f"position: {{x: {self.reset_pos_x}, y: {self.reset_pos_y}, z: {self.reset_pos_z}}}, "
                f"orientation: {{x: {self.reset_ori_x}, y: {self.reset_ori_y}, "
                f"z: {self.reset_ori_z}, w: {self.reset_ori_w}}}"
            )
            
            cmd = [
                'ign', 'service',
                '-s', service_name,
                '--reqtype', 'ignition.msgs.Pose',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', str(self.timeout),
                '--req', pose_msg
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout / 1000.0 + 2.0
            )
            
            success = result.returncode == 0
            
            if not success and result.stderr:
                self.get_logger().debug(f'Pose reset stderr: {result.stderr[:100]}')
            
            return success
                
        except subprocess.TimeoutExpired:
            self.get_logger().error('Pose reset timeout')
            return False
        except Exception as e:
            self.get_logger().debug(f'Pose reset exception: {e}')
            return False
    
    def get_reset_stats(self):
        """Get reset statistics"""
        return {
            'reset_count': self.reset_count,
            'failed_resets': self.failed_resets,
            'controller_reset_failures': self.controller_reset_failures,
            'gazebo_failures': self.gazebo_reset_failures,
            'controller_start_failures': self.controller_start_failures,
            'success_rate': (self.reset_count - self.failed_resets) / max(self.reset_count, 1) * 100,
            'last_reset_time': self.last_reset_time
        }


def main(args=None):
    rclpy.init(args=args)
    
    node = GzResetHandlerNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        stats = node.get_reset_stats()
        lines = [
            'Shutting down Gz Reset Handler Node',
            '=' * 70,
            '📊 Final Reset Statistics:',
            f'  Total resets: {stats["reset_count"]}',
            f'  Successful: {stats["reset_count"] - stats["failed_resets"]}',
            f'  Failed: {stats["failed_resets"]}',
            f'    - Controller reset failures: {stats["controller_reset_failures"]}',
            f'    - Gazebo failures: {stats["gazebo_failures"]}',
            f'    - Controller start failures: {stats["controller_start_failures"]}',
            f'  Success rate: {stats["success_rate"]:.1f}%',
            '=' * 70,
        ]

        if rclpy.ok():
            for line in lines:
                node.get_logger().info(line)
        else:
            for line in lines:
                print(f'[gz_reset_handler_node] {line}')
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
