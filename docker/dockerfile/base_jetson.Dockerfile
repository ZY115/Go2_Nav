# ======================================================
# Edge-Brain ROS2 Base Image for Nvidia Jetson with CUDA
# ======================================================

# ------------ Base Image ------------
# FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

LABEL maintainer="Yihao Cai <yihao.cai@wayne.edu>" \
      version="v2.0.0" \
      description="Nvidia Jetson: ROS2 Humble + CUDA 12.6 + Robotics Libraries" \
      license="Apache-2.0"


# ------------ System Environment ------------
ARG USER_UID=1000
ARG USERNAME=edgebot    
ARG HOME_PATH=/home/${USERNAME}
ARG THIRD_PARTY_PATH=${HOME_PATH}/third_party

ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8


# ------------ Basic Tools ------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget gnupg2 lsb-release ca-certificates git vim build-essential \
    cmake python3 python3-pip locales iproute2 net-tools bash-completion \
    software-properties-common nmap lsof libglfw3-dev locate liburdfdom-headers-dev \
    liburdfdom-dev liboctomap-dev \
    && locale-gen en_US.UTF-8


# ------------ Install ROS2 ------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep python3-rosinstall python3-vcstool python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-desktop ros-${ROS_DISTRO}-backward-ros ros-${ROS_DISTRO}-common-interfaces \
    ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers ros-${ROS_DISTRO}-octomap \
    ros-${ROS_DISTRO}-octomap-msgs \
    ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-rmw-connextdds

RUN rosdep init || true && rosdep update


# ------------ Create user ------------
RUN useradd -m ${USERNAME} && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN mkdir -p ${THIRD_PARTY_PATH} && chown -R ${USERNAME}:${USERNAME} ${THIRD_PARTY_PATH}
USER ${USERNAME}
WORKDIR ${THIRD_PARTY_PATH}


# ------------ ROS2 Workspace ------------
ARG MAKE_JOBS
ARG ROS2_WS_NAME=edge_ws
ARG ROS2_WS_PATH=/home/${USERNAME}/${ROS2_WS_NAME}
SHELL ["/bin/bash", "-c"]

    
# ------------ Build eigenpy ------------
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libboost-python-dev python3-dev python3-scipy \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-thread-dev \
    && sudo rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/stack-of-tasks/eigenpy.git && \
    cd eigenpy && mkdir build && cd build && \
    cmake .. && \
    make -j${MAKE_JOBS} && \
    sudo make install


# ------------ Build Coal ------------
RUN sudo apt update && \
    sudo apt install -y cmake libeigen3-dev libboost-all-dev python3-dev \
    python3-pip libassimp-dev \
    && sudo rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/coal-library/coal.git && \
    cd coal && git checkout devel && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j${MAKE_JOBS} && \
    sudo make install


# ------------ Build hpp-fcl ------------
RUN git clone --recursive https://github.com/humanoid-path-planner/hpp-fcl.git && \
    cd hpp-fcl && git checkout v2.4.5 && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j${MAKE_JOBS} && \
    sudo make install


# ------------ Build Pinocchio ------------
RUN git clone --recursive https://github.com/stack-of-tasks/pinocchio.git && \
    cd pinocchio && git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_INTERFACE=ON \
    -DBUILD_WITH_COLLISION_SUPPORT=ON \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j${MAKE_JOBS} && \
    sudo make install


# ------------ Build unitree_sdk2 ------------
RUN git clone https://github.com/unitreerobotics/unitree_sdk2.git && \
    cd unitree_sdk2 && git checkout f29ee9f && \
    mkdir build && cd build && \
    cmake .. && \
    make -j${MAKE_JOBS} && \
    sudo make install


# ------------ Build unitree_ros2 ------------
RUN sudo apt update && \
    sudo apt install -y ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-rosidl-generator-dds-idl libyaml-cpp-dev && \
    git clone https://github.com/unitreerobotics/unitree_ros2.git && \
    cd unitree_ros2/cyclonedds_ws/src && \
    git clone https://github.com/ros2/rmw_cyclonedds -b ${ROS_DISTRO} && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x && \
    cd .. && colcon build --packages-select cyclonedds && \
    c && colcon build


# ------------ Build mujoco ------------
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libx11-dev \
    libxext-dev \
    && sudo rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/google-deepmind/mujoco.git && \
    cd mujoco && git checkout 3.3.3 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j${MAKE_JOBS} && \
    sudo make install


# ------------ Build unitree_mujoco ------------
RUN git clone https://github.com/Charlescai123/unitree_mujoco.git && \
    cd unitree_mujoco/simulate && git checkout mujoco-v3.3.3 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j${MAKE_JOBS} 


# ----------- Install PyTorch and toolboxs for JetPack 6.2 -------------
# reference: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# download source: https://download.pytorch.org/whl/torch/
WORKDIR ${HOME_PATH}
RUN mkdir torch_libs && cd torch_libs && \
    wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl && \
    wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl && \
    wget https://nvidia.box.com/shared/static/9agsjfee0my4sxckdpuk9x9gt8agvjje.whl -O torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl && \
    pip3 install torch-2.3.0-cp310-cp310-linux_aarch64.whl torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
    # cd .. && rm -rf torch_libs

# ------------ Set Paths ------------
RUN echo '' >> ${HOME_PATH}/.bashrc
RUN echo '################## Add CUDA Library ##################' >> ${HOME_PATH}/.bashrc
RUN echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ${HOME_PATH}/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ${HOME_PATH}/.bashrc
RUN echo '' >> ${HOME_PATH}/.bashrc
RUN echo '################## Add Pinocchio Library ##################' >> ${HOME_PATH}/.bashrc
RUN echo 'export PATH=/usr/local/bin:$PATH' >> ${HOME_PATH}/.bashrc
RUN echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ${HOME_PATH}/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ${HOME_PATH}/.bashrc
RUN echo 'export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH' >> ${HOME_PATH}/.bashrc
RUN echo 'export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH' >> ${HOME_PATH}/.bashrc
RUN echo '###########################################################' >> ${HOME_PATH}/.bashrc


# ------------ Set Other ENV ------------
ENV ROS_DOMAIN_ID=0
ENV ROS_DISTRO=${ROS_DISTRO}
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
