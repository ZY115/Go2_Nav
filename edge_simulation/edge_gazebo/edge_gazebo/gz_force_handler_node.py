#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Gz Force Handler Node - Supports Impulse and Pulse Forces
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import Wrench
import subprocess
import threading
import time

class GzForceHandlerNode(Node):
    def __init__(self):
        super().__init__('gz_force_handler_node')
        
        # Parameters
        self.declare_parameter('world_name', 'obstacle_course')
        self.declare_parameter('model_name', 'go2')
        self.declare_parameter('force_magnitude', 200.0)
        self.declare_parameter('pulse_duration', 0.3)  # Pulse duration
        self.declare_parameter('use_impulse', False)   # Whether to use impulse force
        
        self.world = self.get_parameter('world_name').value
        self.model = self.get_parameter('model_name').value
        self.force_mag = self.get_parameter('force_magnitude').value
        self.pulse_duration = self.get_parameter('pulse_duration').value
        self.use_impulse = self.get_parameter('use_impulse').value
        
        # Topic names
        self.wrench_topic = f"/world/{self.world}/wrench"
        self.persistent_topic = f"/world/{self.world}/wrench/persistent"
        self.clear_topic = f"/world/{self.world}/wrench/clear"
        
        # Create services - Pulse forces (lasting for a duration)
        self.create_service(Trigger, 'push_forward', self.push_forward_cb)
        self.create_service(Trigger, 'push_backward', self.push_backward_cb)
        self.create_service(Trigger, 'push_left', self.push_left_cb)
        self.create_service(Trigger, 'push_right', self.push_right_cb)
        self.create_service(Trigger, 'push_down', self.push_down_cb)
        
        # Create services - Impulse forces (applied once)
        self.create_service(Trigger, 'impulse_forward', self.impulse_forward_cb)
        self.create_service(Trigger, 'impulse_backward', self.impulse_backward_cb)
        self.create_service(Trigger, 'impulse_left', self.impulse_left_cb)
        self.create_service(Trigger, 'impulse_right', self.impulse_right_cb)
        
        # Create services - Rotation and clearing
        self.create_service(Trigger, 'rotate_cw', self.rotate_cw_cb)
        self.create_service(Trigger, 'rotate_ccw', self.rotate_ccw_cb)
        self.create_service(Trigger, 'clear_force', self.clear_force_cb)
        
        # Subscribe to custom force commands
        self.create_subscription(Wrench, 'external_force_cmd', self.wrench_cb, 10)
        
        self.print_info()
    
    def print_info(self):
        """Print node information"""
        self.get_logger().info('='*70)
        self.get_logger().info('🚀 GZ Force Handler Node Started')
        self.get_logger().info('='*70)
        self.get_logger().info(f'🌍 World: {self.world}')
        self.get_logger().info(f'📦 Model: {self.model}')
        self.get_logger().info(f'💪 Force Magnitude: {self.force_mag} N')
        self.get_logger().info(f'⏱️  Pulse Duration: {self.pulse_duration} s')
        self.get_logger().info(f'⚡ Use Impulse: {self.use_impulse}')
        self.get_logger().info('='*70)
        self.get_logger().info('📞 Pulse Force Services (lasting {}s):'.format(self.pulse_duration))
        self.get_logger().info('   ros2 service call /push_forward std_srvs/srv/Trigger')
        self.get_logger().info('   ros2 service call /push_backward std_srvs/srv/Trigger')
        self.get_logger().info('   ros2 service call /push_left std_srvs/srv/Trigger')
        self.get_logger().info('   ros2 service call /push_right std_srvs/srv/Trigger')
        self.get_logger().info('   ros2 service call /push_down std_srvs/srv/Trigger')
        self.get_logger().info('')
        self.get_logger().info('⚡ Impulse Force Services (applied once):')
        self.get_logger().info('   ros2 service call /impulse_forward std_srvs/srv/Trigger')
        self.get_logger().info('   ros2 service call /impulse_backward std_srvs/srv/Trigger')
        self.get_logger().info('   ros2 service call /impulse_left std_srvs/srv/Trigger')
        self.get_logger().info('   ros2 service call /impulse_right std_srvs/srv/Trigger')
        self.get_logger().info('')
        self.get_logger().info('🔄 Rotation Services:')
        self.get_logger().info('   ros2 service call /rotate_cw std_srvs/srv/Trigger')
        self.get_logger().info('   ros2 service call /rotate_ccw std_srvs/srv/Trigger')
        self.get_logger().info('')
        self.get_logger().info('🧹 Clear Service:')
        self.get_logger().info('   ros2 service call /clear_force std_srvs/srv/Trigger')
        self.get_logger().info('='*70)
        self.get_logger().info('📡 Subscribed Topic: /external_force_cmd (geometry_msgs/Wrench)')
        self.get_logger().info('='*70)
    
    def apply_wrench(self, fx, fy, fz, tx=0, ty=0, tz=0, persistent=False):
        """Apply force/torque"""
        topic = self.persistent_topic if persistent else self.wrench_topic
        
        msg = (
            f"entity: {{name: '{self.model}', type: MODEL}}, "
            f"wrench: {{"
            f"force: {{x: {fx}, y: {fy}, z: {fz}}}, "
            f"torque: {{x: {tx}, y: {ty}, z: {tz}}}"
            f"}}"
        )
        
        cmd = ['ign', 'topic', '-t', topic, '-m', 'ignition.msgs.EntityWrench', '-p', msg]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
            return result.returncode == 0
        except Exception as e:
            self.get_logger().error(f'❌ Error executing command: {e}')
            return False
    
    def apply_impulse(self, fx, fy, fz, tx=0, ty=0, tz=0):
        """Apply impulse force (applied once)"""
        force_str = f'F=[{fx:.1f}, {fy:.1f}, {fz:.1f}]'
        torque_str = f'T=[{tx:.1f}, {ty:.1f}, {tz:.1f}]' if any([tx, ty, tz]) else ''
        msg = f'{force_str} {torque_str}'.strip()
        
        self.get_logger().info(f'⚡ Applying impulse: {msg} N/Nm')
        
        # Use impulse force topic
        return self.apply_wrench(fx, fy, fz, tx, ty, tz, persistent=False)
    
    def apply_force_pulse(self, fx, fy, fz, tx=0, ty=0, tz=0):
        """Apply pulse force (lasting for a duration then auto-cleared)"""
        def _apply():
            force_str = f'F=[{fx:.1f}, {fy:.1f}, {fz:.1f}]'
            torque_str = f'T=[{tx:.1f}, {ty:.1f}, {tz:.1f}]' if any([tx, ty, tz]) else ''
            msg = f'{force_str} {torque_str}'.strip()
            
            self.get_logger().info(f'🎯 Applying pulse: {msg} N/Nm, duration {self.pulse_duration}s')
            
            # Apply persistent force
            if not self.apply_wrench(fx, fy, fz, tx, ty, tz, persistent=True):
                self.get_logger().error('❌ Failed to apply force')
                return
            
            # Wait for specified duration
            time.sleep(self.pulse_duration)
            
            # Clear force
            self.clear_force()
            self.get_logger().info('✅ Pulse application completed')
        
        threading.Thread(target=_apply, daemon=True).start()
    
    def clear_force(self):
        """Clear persistent force"""
        msg = f"name: '{self.model}', type: MODEL"
        cmd = ['ign', 'topic', '-t', self.clear_topic, '-m', 'ignition.msgs.Entity', '-p', msg]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
            return result.returncode == 0
        except Exception as e:
            self.get_logger().error(f'❌ Error clearing force: {e}')
            return False
    
    # ========== Pulse Force Services (lasting for a duration) ==========
    
    def push_forward_cb(self, request, response):
        self.get_logger().info('⬆️  Push forward (X+) - Pulse')
        self.apply_force_pulse(self.force_mag, 0, 0)
        response.success = True
        response.message = f'Pushing forward with {self.force_mag}N for {self.pulse_duration}s'
        return response
    
    def push_backward_cb(self, request, response):
        self.get_logger().info('⬇️  Push backward (X-) - Pulse')
        self.apply_force_pulse(-self.force_mag, 0, 0)
        response.success = True
        response.message = f'Pushing backward with {self.force_mag}N for {self.pulse_duration}s'
        return response
    
    def push_left_cb(self, request, response):
        self.get_logger().info('⬅️  Push left (Y+) - Pulse')
        self.apply_force_pulse(0, self.force_mag, 0)
        response.success = True
        response.message = f'Pushing left with {self.force_mag}N for {self.pulse_duration}s'
        return response
    
    def push_right_cb(self, request, response):
        self.get_logger().info('➡️  Push right (Y-) - Pulse')
        self.apply_force_pulse(0, -self.force_mag, 0)
        response.success = True
        response.message = f'Pushing right with {self.force_mag}N for {self.pulse_duration}s'
        return response
    
    def push_down_cb(self, request, response):
        self.get_logger().info('⬇️  Push down (Z-) - Pulse')
        self.apply_force_pulse(0, 0, -self.force_mag)
        response.success = True
        response.message = f'Pushing down with {self.force_mag}N for {self.pulse_duration}s'
        return response
    
    # ========== Impulse Force Services (applied once) ==========
    
    def impulse_forward_cb(self, request, response):
        self.get_logger().info('⚡ Impulse forward (X+)')
        success = self.apply_impulse(self.force_mag * 10, 0, 0)  # Impulse needs larger force
        response.success = success
        response.message = f'Impulse forward with {self.force_mag * 10}N'
        return response
    
    def impulse_backward_cb(self, request, response):
        self.get_logger().info('⚡ Impulse backward (X-)')
        success = self.apply_impulse(-self.force_mag * 10, 0, 0)
        response.success = success
        response.message = f'Impulse backward with {self.force_mag * 10}N'
        return response
    
    def impulse_left_cb(self, request, response):
        self.get_logger().info('⚡ Impulse left (Y+)')
        success = self.apply_impulse(0, self.force_mag * 10, 0)
        response.success = success
        response.message = f'Impulse left with {self.force_mag * 10}N'
        return response
    
    def impulse_right_cb(self, request, response):
        self.get_logger().info('⚡ Impulse right (Y-)')
        success = self.apply_impulse(0, -self.force_mag * 10, 0)
        response.success = success
        response.message = f'Impulse right with {self.force_mag * 10}N'
        return response
    
    # ========== Rotation and Clearing ==========
    
    def rotate_cw_cb(self, request, response):
        self.get_logger().info('🔄 Rotate clockwise (Z-) - Pulse')
        self.apply_force_pulse(0, 0, 0, 0, 0, -50)
        response.success = True
        response.message = 'Rotating clockwise'
        return response
    
    def rotate_ccw_cb(self, request, response):
        self.get_logger().info('🔄 Rotate counter-clockwise (Z+) - Pulse')
        self.apply_force_pulse(0, 0, 0, 0, 0, 50)
        response.success = True
        response.message = 'Rotating counter-clockwise'
        return response
    
    def clear_force_cb(self, request, response):
        self.get_logger().info('🧹 Clearing all persistent forces')
        success = self.clear_force()
        response.success = success
        response.message = 'Force cleared' if success else 'Failed to clear force'
        return response
    
    def wrench_cb(self, msg):
        """Receive custom force command"""
        self.get_logger().info(
            f'📨 Received custom force: F=[{msg.force.x:.1f}, {msg.force.y:.1f}, {msg.force.z:.1f}], '
            f'T=[{msg.torque.x:.1f}, {msg.torque.y:.1f}, {msg.torque.z:.1f}]'
        )
        
        if self.apply_wrench(
            msg.force.x, msg.force.y, msg.force.z,
            msg.torque.x, msg.torque.y, msg.torque.z,
            persistent=True
        ):
            self.get_logger().info('✅ Custom force applied (persistent, needs manual clearing)')
        else:
            self.get_logger().error('❌ Failed to apply custom force')

def main(args=None):
    rclpy.init(args=args)
    node = GzForceHandlerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Clearing all forces...')
        node.clear_force()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
