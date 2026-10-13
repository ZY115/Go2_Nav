#!/usr/bin/bash

ros2 topic pub -1 /edge/synthesis_state edge_msgs/msg/Vec12Msg "{data: [0.0, 0.0, -0., -0., -0., 0.0, 0., -0., 0., 0., -0., -0.1]}"

ros2 topic pub -1 /edge/synthesis_state edge_msgs/msg/Vec12Msg "{data: [0, 0, -0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624]}"

ros2 topic pub -1 /edge/synthesis_state edge_msgs/msg/Vec12Msg "{data: [0, 0, -0.00137003, 0.02392, -0.442369, -0.0794646, 0.074993, -0.134433, 0.086083, -0.194055, -0.12749, -0.530516]}"