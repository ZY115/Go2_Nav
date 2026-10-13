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
    curl wget gnupg2 lsb-release ca-certificates git vim build-essential sudo \
    cmake python3 python3-pip locales iproute2 net-tools iputils-ping bash-completion \
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
# [本地修改] 加入 video/render 组：Jetson 的 /dev/nvmap、/dev/nvhost-* 是 root:video 660，
# 不加的话 RViz 起不来（NvRmMemInitNvmap failed with Permission denied，段错误退出）。
# 注意宿主机 gid 未必与容器一致，docker run 时还需 --group-add <宿主机 video/render gid>。
RUN usermod -aG video,render root 2>/dev/null || true
RUN useradd -m -u ${USER_UID} -s /bin/bash -G video,render ${USERNAME} && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
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

# [本地修改] 原来不锁版本，拉 master。锁到本次验证通过的 commit（v3.13.0-36-gfdba511）。
# 注意：**不要**加 git submodule update --init。eigenpy 的 cmake 子模块保持未初始化状态
# 才能装出 pypa-metadata.cmake；初始化后会拉到旧版 jrl-cmakemodules，
# 导致后面 coal 配置时报 "include could not find requested file: pypa-metadata.cmake"。
RUN git clone https://github.com/stack-of-tasks/eigenpy.git && \
    cd eigenpy && git checkout fdba511 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j${MAKE_JOBS} && \
    sudo make install


# ------------ Build Coal ------------
RUN sudo apt update && \
    sudo apt install -y cmake libeigen3-dev libboost-all-dev python3-dev \
    python3-pip libassimp-dev \
    && sudo rm -rf /var/lib/apt/lists/*

# [本地修改] 原来是 devel 分支（滚动）。锁到本次验证通过的 commit（v3.0.4-553-ge11df93d）
RUN git clone --recursive https://github.com/coal-library/coal.git && \
    cd coal && git checkout e11df93d && \
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
# [本地修改] 【关键】原来不锁版本，拉 master。2026-04-13 发布的 v4.0.0 重排了头文件目录，
# 把 pinocchio/spatial/se3.hpp 挪进了 deprecated/，并且关节类型增至 26 种、
# 超过 boost::variant 默认 20 个模板参数的上限。本项目源码按 3.x 的 API 写的，
# 用 4.x 会在 edge_quadruped_controller 上编译失败（旧机器构建于 pinocchio 还是 3.x 时）。
# v3.9.0 是最后一个 3.x。另注：Dockerfile 里 hpp-fcl 锁 v2.4.5 也是 3.x 的搭配。
RUN git clone --recursive https://github.com/stack-of-tasks/pinocchio.git && \
    cd pinocchio && git checkout v3.9.0 && \
    git submodule update --init --recursive && \
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
    source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build

# [本地修改] unitree_ros2 上游自带的 setup.sh 写死了三处旧环境，换机必坏：
#   1) source /opt/ros/foxy  -> 本镜像是 humble，没有 foxy
#   2) $HOME/unitree_ros2    -> 实际在 $HOME/third_party/unitree_ros2
#   3) NetworkInterface enp3s0 -> 那是旧台式机网口名；Jetson 是 eth0
# 第 3 条最致命：CycloneDDS 报 "enp3s0: does not match an available interface"
# 然后 rmw_create_node 直接失败，终端 3 一个话题都收不到（比 DEPLOYMENT.md §10.2
# 描述的“RViz 全黑”更早就挂了）。改成按 GO2_NET_IF 自适应，默认 eth0。
RUN printf '%s\n' \
    '#!/bin/bash' \
    'echo "Setup unitree ros2 environment"' \
    'source /opt/ros/humble/setup.bash' \
    'source $HOME/third_party/unitree_ros2/cyclonedds_ws/install/setup.bash' \
    'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' \
    'export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${GO2_NET_IF:-eth0}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"' \
    > /home/edgebot/third_party/unitree_ros2/setup.sh && \
    chmod +x /home/edgebot/third_party/unitree_ros2/setup.sh

# [本地修改] DISPLAY 自适应。docker run 时 -e DISPLAY=$DISPLAY 会把宿主机当时的
# 显示号固化进容器环境；宿主机重启后 X server 可能起在别的号上（实测 :1 -> :0），
# 容器就去连一个不存在的 socket，Qt 直接 abort()，RViz 以 exit -6 退出 —— launch
# 日志里只报 "process has died"，完全看不出是显示问题。改成按实际存在的 socket 推断。
# 注意 .bashrc 顶部对非交互 shell 有早退，所以这段只在 docker exec -it 下生效。
RUN printf '%s\n' \
    '' \
    '# X display follows whatever socket the host actually has' \
    'for __x in /tmp/.X11-unix/X*; do' \
    '    [ -e "$__x" ] && export DISPLAY=":${__x##*/X}"' \
    '    break' \
    'done' \
    'unset __x' \
    >> /home/edgebot/.bashrc


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
# [本地修改] 原来的三个 nvidia.box.com 直链已全部 404，改用 PyTorch 官方 index。
# 本机 host 是 JetPack 5.1.2（驱动只到 CUDA 11.4），容器里的 CUDA 12.6 用不了，
# 所以装 CPU 版；torchaudio 项目里没有任何地方 import，直接省掉。
# 将来若整机刷成 JetPack 6，把 index-url 换成 NVIDIA 的 jp6/cu126 源即可。
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.3.0 torchvision==0.18.0

# ------------ rosdep install 工作区依赖 ------------
# [本地修改] 原 Dockerfile 只跑了 rosdep init/update，从没跑 rosdep install，
# 导致 colcon 编译时逐个暴露缺失依赖（pcl_ros / xacro / tf_transformations / ...）。
# 这里预装一批已知需要的，减少首次编译踩坑；容器内仍建议再跑一次：
#   rosdep install --from-paths src --ignore-src -r -y
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-tf-transformations \
    ros-${ROS_DISTRO}-joint-state-publisher ros-${ROS_DISTRO}-joint-state-publisher-gui \
    && rm -rf /var/lib/apt/lists/*
USER ${USERNAME}


# ------------ Install Ignition Gazebo Fortress ------------
# [本地修改] 原 Dockerfile 漏了这一步，导致 gz_quadruped_hardware 编译时
# find_package(ignition-gazebo6) 失败，并连带 aborted 掉其余 11 个包。
# 旧机器的 edge_ws_artifacts/install/ 里有 gz_quadruped_hardware，
# 说明当年是在容器里手动补装的，没写进 Dockerfile。
# 注意：ros-humble-ros-gz-sim 官方没有 arm64 包，只能装 ignition 本体。
USER root
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && apt-get install -y --no-install-recommends ignition-fortress && \
    rm -rf /var/lib/apt/lists/*
USER ${USERNAME}


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
RUN echo 'export PYTHONPATH=/usr/local/lib/python3.10/site-packages:$PYTHONPATH' >> ${HOME_PATH}/.bashrc
RUN echo 'export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH' >> ${HOME_PATH}/.bashrc
RUN echo '###########################################################' >> ${HOME_PATH}/.bashrc


# ------------ Set Other ENV ------------
ENV ROS_DOMAIN_ID=0
ENV ROS_DISTRO=${ROS_DISTRO}
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
