#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    trigger_config_path = os.path.join(
        get_package_share_directory('edge_trigger'),
        'config',
        'trigger.yaml'
    )

    update_rate_arg = DeclareLaunchArgument(
        'update_rate',
        default_value='500.0',
        description='Trigger update rate (Hz)'
    )

    print_processing_arg = DeclareLaunchArgument(
        'print_processing',
        default_value='false',
        description='Print processing logs'
    )
    
    start_with_teacher_arg = DeclareLaunchArgument(
        'start_with_teacher',
        default_value='true',
        description='Start in teacher mode if true, else start in student mode. \
            Starting mode is important for safety.'
    )
    
    action_source_arg = DeclareLaunchArgument(
        'action_source',
        # [本地修改] 原为 '-1' (auto)。注意这个默认值会覆盖 trigger.yaml 里的同名项
        # （见下面 parameters=[...]：YAML 先加载，这里的 LaunchConfiguration 后覆盖），
        # 所以只改 YAML 是没有效果的。
        # auto 模式下 selectAction() 在系统 SAFE 时一律返回 STUDENT，而 drl_student
        # 在 bringup.real.end2end.launch.py:301 是注释掉的、从不启动，于是
        # is_stu_action_initialized_ 恒为 false，trigger 直接 return，把 phy_teacher
        # 算好的 6 维机身加速度全部丢弃 -> FSMTrotting::calcTau 里 accel 恒为零向量,
        # 机身姿态/速度误差完全没有反馈。接上 student 之前必须锁定为 teacher。
        # 【2026-09-20 实测回退】改成 '1' 让教师闭环后，机身摇晃恶化了一个数量级：
        # roll 峰峰 1.4 deg -> 14.3 deg, wx RMS 0.044 -> 0.447 rad/s。
        # 原因是教师输出本身没整定好：accel[3] 均值 -58.3 rad/s^2、accel[5] 均值 -39.2,
        # 而 FSMTrotting::calcTau 的饱和限是 +-40 / +-20 —— 等于常年顶死饱和，是恒定
        # 大力矩而非纠偏反馈。在教师增益整定好之前不要闭这个环。
        default_value='-1',
        description='Action source: -1 -- none, 0 -- student, 1 -- teacher'
    )

    trigger_node = Node(
        package='edge_trigger',
        executable='trigger',
        namespace='edge',
        name='trigger',
        output='screen',
        parameters=[
            trigger_config_path,        # Load default config from YAML
            {
                # Override parameters with launch arguments
                'update_rate': LaunchConfiguration('update_rate'),
                'print_processing': LaunchConfiguration('print_processing'),
                'start_with_teacher': LaunchConfiguration('start_with_teacher'),
                'action_source': LaunchConfiguration('action_source'),
            }
        ]
    )

    return LaunchDescription([
        update_rate_arg,
        print_processing_arg,
        start_with_teacher_arg,
        action_source_arg,
        
        trigger_node,
    ])
