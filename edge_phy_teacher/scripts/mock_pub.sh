#!/usr/bin/bash

YAML=$(ros2 pkg prefix edge_phy_teacher)/share/edge_phy_teacher/config/phy_teacher.yaml

ros2 run edge_phy_teacher mock_publisher \
  --ros-args \
  --params-file $YAML