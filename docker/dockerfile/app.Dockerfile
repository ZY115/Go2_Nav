# docker/app.Dockerfile
# ======================================================
# edge-brain-go2 App Image
# ======================================================
# syntax=docker/dockerfile:1.5

# ------------ Build Arguments ------------
ARG PLATFORM
ARG BASE_IMAGE=edge-brain-base

FROM ${BASE_IMAGE}

ARG USERNAME=edgebot
ENV ROS_DISTRO=humble
WORKDIR /home/${USERNAME}

# Switch to root for SSH git clone
USER root


# ------------ Pull private repo ------------
RUN --mount=type=ssh \
    mkdir -p ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    mkdir -p edge_ws/src/ && cd edge_ws/src/ && \
    git clone git@github.com:Charlescai123/edge-brain-go2.git && \
    chown -R ${USERNAME}:${USERNAME} /home/edgebot/edge_ws


# ------------ Ignition Gazebo ------------
RUN apt-get update && apt-get install -y lsb-release gnupg && \
    curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
        tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update && apt-get install -y ignition-fortress

    
# ------------ Extra ROS2 Dependencies ------------
RUN apt update && apt install -y ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-ros-gz-bridge ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-ros-gz-sim ros-${ROS_DISTRO}-ros-gz-interfaces ros-${ROS_DISTRO}-rqt-robot-steering 


# ------------ Python Dependencies ------------
RUN pip3 install --upgrade pip && pip3 install --ignore-installed transforms3d ultralytics cv-bridge opencv-python \
    numpy==1.26.4 segment-anything pdaqp==0.6.7 cvxpy==1.7.3 cvxopt==1.3.2 scipy==1.15.3 setuptools==59.6.0


# ------------ Checkout branch as user ------------
USER ${USERNAME}
RUN cd edge_ws/src/edge-brain-go2 && git checkout Jetson

SHELL ["/bin/bash", "-c"]

# ------------ Install cvxPYGEN and Build Gain Synthesizer --------------
RUN cd edge_ws/src/edge-brain-go2/edge_locomotion/edge_phy_teacher/src/optimization/ecvxcone/third_party/cvxpygen \
    && pip install --user -e . && python build_solver.py


# ------------ Build custom ROS2 packages ------------
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd edge_ws/src/edge-brain-go2 && ./edge_scripts/build_ws.sh || true


# ------------ Environment Variables ------------
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
# ENV PYTHONPATH=/usr/lib/python3/dist-packages:/home/${USERNAME}/edge_ws/install/lib/python3.10/site-packages


# ------------ Entrypoint ------------
COPY --chmod=755 entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["/bin/bash"]
# ENTRYPOINT ["/bin/bash", "-c"]
# CMD ["source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch edge_quadruped_controller bringup.launch.py"]
