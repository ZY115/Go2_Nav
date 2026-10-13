#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import Jetson.GPIO as GPIO

import enum
import numpy as np
from edge_msgs.msg import EdgeAction

class LEDIndicatorMode(enum.Enum):
    """Enum for LED indicator modes."""
    SAFETY = 0
    CONTROL = 1


# -------------------- GPIO CONFIG --------------------
SAFETY_PIN_LED = 15
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SAFETY_PIN_LED, GPIO.OUT)

def set_led(red_on: bool):
    GPIO.output(SAFETY_PIN_LED, GPIO.LOW if red_on else GPIO.HIGH)

# -------------------- Safety LED Driver Node --------------------
class SafetyLedDriverNode(Node):
    def __init__(self):
        super().__init__("safety_led_driver")
        
        # Parameters from launch
        self.led_indicator_mode = LEDIndicatorMode[
            self.declare_parameter("led_indicator_mode", "safety").value.upper()
        ]
        
        self.get_logger().info(f"💡 Safety LED Indicator Mode: {self.led_indicator_mode}")
        self.get_logger().info("🟢/🔴 Safety LED Driver Node started.")
        
        ns = self.get_namespace()
        ns_prefix = f"{ns}" if ns != "/" else ""
        
        # Subscriptions based on mode
        if self.led_indicator_mode == LEDIndicatorMode.SAFETY:
            self.get_logger().info(f"⚪ Subscribing to '{ns_prefix}/is_safe' topic for safety status.")
            self.create_subscription(
                msg_type = Bool,
                topic = f"is_safe",
                callback = self.safety_callback,
                qos_profile = 10,
            )
            
        elif self.led_indicator_mode == LEDIndicatorMode.CONTROL:
            self.get_logger().info(f"⚪ Subscribing to '{ns_prefix}/command_action' topic for control status.")
            self.create_subscription(
                msg_type = EdgeAction,
                topic = f"command_action",
                callback = self.control_callback,
                qos_profile = 10,
            )
        else:
            self.get_logger().error(f"❌ Invalid LED indicator mode! Defaulting to SAFETY mode., and subscribing to '{ns_prefix}/is_safe' topic.")
            self.led_indicator_mode = LEDIndicatorMode.SAFETY
            self.create_subscription(
                msg_type = Bool,
                topic = f"is_safe",
                callback = self.safety_callback,
                qos_profile = 10,
            )

        
    def safety_callback(self, msg: Bool):
        """callback when receiving is_safe message"""
        if msg.data:
            set_led(False)  # safe → green light on
            # self.get_logger().info("✅ System is safe.")
        else:
            set_led(True)   # unsafe → red light on
            # self.get_logger().warn("⚠️ System is unsafe!")


    def control_callback(self, msg: EdgeAction):
        """callback when receiving terminal_action message"""
        if msg.action_mode == EdgeAction.STUDENT:
            set_led(False)  # control active → green light on
            # self.get_logger().info("✅ Control action active.")
        elif msg.action_mode == EdgeAction.TEACHER:
            set_led(True)   # control inactive → red light on
            # self.get_logger().warn("⚠️ Control action inactive!")

    def destroy_node(self):
        GPIO.cleanup()
        super().destroy_node()
        self.get_logger().info("💡 GPIO cleaned up and node shut down.")

# -------------------- MAIN --------------------
def main(args=None):
    rclpy.init(args=args)
    node = SafetyLedDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
