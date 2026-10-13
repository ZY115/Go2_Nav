#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import time
import rclpy
from rclpy.node import Node
import asyncio
import Jetson.GPIO as GPIO

from go2_webrtc_driver.constants import RTC_TOPIC
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod


# -------------------- GPIO --------------------
PIN_LED = 13
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN_LED, GPIO.OUT)

def set_led(red_on: bool):
    GPIO.output(PIN_LED, GPIO.LOW if red_on else GPIO.HIGH)


# -------------------- Safety Monitor --------------------
class WebRTCSafetyMonitor:
    def __init__(self, logger):
        self.logger = logger
        self.last_acc = [0.0, 0.0, 9.81]
        self.last_gyro = [0.0, 0.0, 0.0]

    def detect(self, msg_json):
        message = msg_json["data"]
        imu = message["imu_state"]
        roll, pitch, yaw = imu["rpy"]

        acc = imu["accelerometer"]
        gyro = imu["gyroscope"]
        body_h = message["body_height"]

        acc_diff = sum(abs(acc[i] - self.last_acc[i]) for i in range(3))
        gyro_diff = sum(abs(gyro[i] - self.last_gyro[i]) for i in range(3))
        self.last_acc, self.last_gyro = acc, gyro

        alert = abs(roll) > 0.17 or abs(pitch) > 0.17 or not (0.22 <= body_h <= 0.5)
        if alert:
            self.logger.warn(f"⚠️ Unsafe: roll={roll:.2f}, pitch={pitch:.2f}, h={body_h:.2f}")
        return alert


# -------------------- Node --------------------
class SafetyLEDDriverNode(Node):
    def __init__(self):
        super().__init__("safety_led_driver")
        self.monitor = WebRTCSafetyMonitor(self.get_logger())
        self.webrtc_ip = self.declare_parameter("webrtc_ip", "192.168.123.161").value
        self.conn = None

    async def connect_webrtc(self):
        """Async WebRTC setup."""
        try:
            self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.webrtc_ip)
            await self.conn.connect()
            self.get_logger().info(f"✅ Connected to Go2 WebRTC @ {self.webrtc_ip}")
            self.conn.datachannel.pub_sub.subscribe(
                RTC_TOPIC["LF_SPORT_MOD_STATE"], self.on_msg
            )
        except Exception as e:
            self.get_logger().error(f"WebRTC connect failed: {e}")

    def on_msg(self, msg):
        """Handle IMU messages."""
        unsafe = self.monitor.detect(msg)
        set_led(unsafe)

    async def spin_async(self):
        """ROS2 spin inside asyncio loop."""
        executor = rclpy.executors.SingleThreadedExecutor()
        while rclpy.ok():
            rclpy.spin_once(self, executor=executor, timeout_sec=0.1)
            await asyncio.sleep(0.01)


# -------------------- MAIN --------------------
async def main_async():
    rclpy.init()
    node = SafetyLEDDriverNode()
    await node.connect_webrtc()
    await node.spin_async()
    node.destroy_node()
    rclpy.shutdown()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
