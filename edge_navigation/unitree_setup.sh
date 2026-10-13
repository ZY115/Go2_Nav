#!/bin/bash
echo "Setup unitree ros2 environment"
source /opt/ros/${ROS_DISTRO}/setup.bash
# source $HOME/unitree_ros2/cyclonedds_ws/install/setup.bash

# Keep CycloneDDS C/C++ runtime libraries from the same install tree.
# This avoids ABI-mismatch crashes when libddscxx is in /usr/local/lib.
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Keep ROS 2 middleware separate from Unitree SDK's Cyclone usage.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0" priority="default" multicast="default"/></Interfaces></General></Domain></CycloneDDS>'

