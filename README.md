# Edge-Brain-Go2

ROS2 Humble control / navigation / learning framework for the Unitree Go2.
This repository is the working version after migrating to a Jetson AGX Orin.

Environment: Jetson AGX Orin, JetPack 5.1.2 (L4T R35.4.1), ROS2 Humble inside a Docker container.
The robot is connected over Ethernet, `192.168.123.99` ↔ `192.168.123.161`.

---

## 1. Startup

Prerequisite: use the remote to make the robot lie down, then **disable motion control** in the app.

Three terminals:

```bash
go1co     # keyboard input
go2vi     # locomotion
go3nav    # navigation
```

In terminal 1, press `2` to stand, `4` to start the gait, then click a goal point in RViz.

| Key | Action |
|---|---|
| `1` | Passive, motors released |
| `2` | Stand; pause while walking |
| `4` | Start gait |
| `W/A/S/D` | Translate |
| `J/L` | Turn |

Emergency stop: **L2+B** on the remote. All joints switch to damping immediately.
It does not go through WiFi or any Jetson-side software; `go2vi` must be restarted to clear it.

---

## 2. Architecture

```
LiDAR + IMU ──> localization ──> terrain ──> global plan ──> local avoidance ──> velocity ──> joint torques
 (firmware)      point_lio      analysis    far_planner     localPlanner       pathFollower    controller
```

### go1co

Starts [`keyboard_input`](edge_interfaces/input_interfaces/keyboard_input/src/KeyboardInput.cpp),
which turns keystrokes into `/control_input` (`command` holds the number key, `lx/ly/rx/ry` the stick values).
It performs no control computation.

### go2vi

Starts [`bringup.real.end2end.launch.py`](edge_bringup/launch/bringup.real.end2end.launch.py):

| Component | Role |
|---|---|
| `ros2_control_node` | 400 Hz control loop, drives the motors through `unitree_sdk2` |
| `edge_quadruped_controller` | Controller plugin |
| `phy_teacher` + `gain_synthesizer` | Model-based teacher, synthesizes gains online via LMI |
| `trigger` | Teacher / student arbitration |
| `rl_runtime` | Episode scheduling, goal-reached detection |
| `global_rviz` | Visualization |

`drl_student_launch` is commented out in the launch file, so the learned policy never starts.

Inside the control loop
([`FSMTrotting.cpp`](edge_control/platform/quadruped/edge_quadruped_controller/src/FSM/FSMTrotting.cpp)):

```
getUserCmd()   select command source, clamp, low-pass
calcCmd()      integrate pcd_ (body position target) and yaw_cmd_ (yaw target)
GaitGenerator  compute footholds and swing trajectories from the gait phase
calcTau()      dd_pcd = Kpp·position error + Kdp·velocity error
               d_wbd  = kp_w·attitude error + Kd_w·angular velocity error
BalanceCtrl    QP solve, distribute body acceleration into foot forces under friction cone constraints
```

Joint commands are finally emitted by `write()` in
[`go2_hardware_interface.cpp`](edge_hardware/unitree_go2_hardware/src/go2_hardware_interface.cpp),
which is also where the remote emergency stop intercepts.

### go3nav

Starts [`system_real_robot_with_route_planner.launch.py`](edge_navigation/bringup/edge_nav_bringup/launch/system_real_robot_with_route_planner.launch.py):

```
transform_everything   /utlidar/cloud + /utlidar/imu  ->  body frame
point_lio              tightly-coupled LiDAR-inertial odometry -> /odom (446Hz), /registered_scan (14.5Hz)
terrainAnalysis        -> /terrain_map       near field
terrainAnalysisExt     -> /terrain_map_ext   far field
far_planner            OpenCV contour extraction -> visibility graph -> global path -> /way_point
localPlanner           collision-check a fan of candidate paths -> /path
pathFollower           track the path -> /cmd_vel (50Hz)
```

- [`transform_everything.py`](edge_navigation/utilities/transform_sensors/transform_sensors/transform_everything.py)
- [`far_planner.cpp`](edge_navigation/planner/route_planner/far_planner/src/far_planner.cpp) /
  [`contour_detector.cpp`](edge_navigation/planner/route_planner/far_planner/src/contour_detector.cpp)
- [`localPlanner.cpp`](edge_navigation/planner/local_planner/src/localPlanner.cpp)
- [`pathFollower.cpp`](edge_navigation/planner/local_planner/src/pathFollower.cpp)

### Setting a goal point

```
RViz GoalpointTool -> /goal_point
far_planner        -> /way_point
localPlanner       -> /path
pathFollower       -> /cmd_vel
edge_quadruped_controller -> FSMTrotting -> foot forces -> joint torques
```

### Two coordinate frames

| | Source | Used by |
|---|---|---|
| `/odom` | point_lio (LiDAR + IMU), map frame | All navigation and planning |
| `estimator_` | Leg kinematics + IMU Kalman filter, leg_odom frame | Inside the control loop |

They have different origins; mixing them produces position errors.
`/state_estimation` from the upstream CMU code is remapped to `/odom` in every launch file.

---

## 3. The Unitree stock controller

I tried to use the same PD control the remote uses, to get a stable gait.
Unitree only exposes a set of API calls — the implementation is a black box,
so the actual code cannot be modified or debugged.

Actions exposed by the API
([`ros2_sport_client.h`](edge_hardware/unitree_go2_hardware/include/unitree_go2_hardware/common/ros2_sport_client.h)):

```
Move(vx, vy, vyaw)    StopMove()            BalanceStand()
FootRaiseHeight(h)    BodyHeight(h)         SpeedLevel(n)
SwitchGait(d)         ContinuousGait(bool)  Euler(r, p, y)
TrajectoryFollow(path)
StandUp / StandDown / Sit / RiseSit / Damp / RecoveryStand / Stretch
```

Parameter-level terrain adaptation is possible — foot raise height, body height, gait switching.

### Running navigation on the stock controller

`pathFollower` has a switch that forwards `/cmd_vel` to the Unitree sport API:

```bash
# Enable motion control in the app, use the remote to make the robot stand
go3nav test_unitree_controller:=true
# Click a goal point in RViz and it starts walking
```

The gait is stable in testing, but the firmware's built-in obstacle avoidance is unreliable
and the robot can still hit walls. Still being investigated.

Note that [`pathFollower.cpp`](edge_navigation/planner/local_planner/src/pathFollower.cpp)
publishes `/cmd_vel` **unconditionally**; the switch only controls whether the sport request
is sent in addition.

Navigation, localization, terrain analysis and path planning all sit outside the black box
and are identical either way.

---

## 4. Missing files

The code came from an archive. Comparing against the original git history, 202 files were not
in the working tree when it was packaged, including five controller implementations:

```
edge_control/platform/quadruped/
    leg_pd_controller            leg PD
    ocs2_quadruped_controller    OCS2 optimal control
    rl_quadruped_controller      reinforcement learning
    safety_patch_controller      safety patch
    unitree_guide_controller     Unitree reference implementation

LICENSE                          the original project is Apache 2.0
dependencies.repo                dependency manifest
```

Only `edge_quadruped_controller` remains, which is the one currently in use.
