#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import Jetson.GPIO as GPIO

# -------------------- GPIO CONFIG --------------------
TASK_PIN_LED = 16
GPIO.setmode(GPIO.BOARD)
GPIO.setup(TASK_PIN_LED, GPIO.OUT)

def set_led(task_done: bool):
    GPIO.output(TASK_PIN_LED, GPIO.HIGH if task_done else GPIO.LOW)

# -------------------- Task LED Driver Node --------------------
class TaskLedDriverNode(Node):
    def __init__(self):
        super().__init__("task_led_driver")
        self.get_logger().info("🔵 Task LED Driver Node started.")
        self.is_task_done = False

        # Subscribe to task status topic
        self.sub_task_ = self.create_subscription(
            msg_type = Bool,
            topic = "is_task_done",
            callback = self.task_callback,
            qos_profile = 10,
        )

        self.reset_led()  # ensure LED is off at startup

    def reset_led(self):
        set_led(False)  # default to off
        self.get_logger().info("💡 Task LED reset to OFF.")

    def task_callback(self, msg: Bool):
        """callback when receiving is_task_done message"""
        if msg.data:
            self.is_task_done = True
        else:
            set_led(False)   # not done → blue light off
            # self.get_logger().warn("⚠️ Task is not done!")
            
        if self.is_task_done:
            set_led(True)  # done → blue light on
            self.get_logger().info("✅ Task is done.")


    def destroy_node(self):
        self.reset_led()
        GPIO.cleanup()
        super().destroy_node()
        self.get_logger().info("💡 GPIO cleaned up and node shut down.")

# -------------------- MAIN --------------------
def main(args=None):
    rclpy.init(args=args)
    node = TaskLedDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
