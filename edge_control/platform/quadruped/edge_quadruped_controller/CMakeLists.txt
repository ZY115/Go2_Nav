cmake_minimum_required(VERSION 3.8)
project(edge_quadruped_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find the dependencies
set(CONTROLLER_DEPENDENCIES
  std_srvs
  rclcpp
  pluginlib
  rcpputils
  controller_interface
  realtime_tools
  std_msgs
  control_input_msgs
  controller_common
  pinocchio
  nav_msgs
  tf2_ros
  geometry_msgs
  edge_msgs
  edge_ros_interfaces
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
foreach (Dependency IN ITEMS ${CONTROLLER_DEPENDENCIES})
    find_package(${Dependency} REQUIRED)
endforeach ()

# Add the libraries
add_library(${PROJECT_NAME} SHARED
  src/edge_quadruped_controller.cpp
  src/FSM/FSMFixedStand.cpp

  src/FSM/FSMFreeStand.cpp
  src/FSM/FSMBalanceTest.cpp
  src/FSM/FSMTrotting.cpp
  src/robot_model/QuadrupedModel.cpp

  src/estimator/Estimator.cpp
  src/control/LowPassFilter.cpp
  src/control/BalanceCtrl.cpp

  src/utils/quadProgpp/Array.cc
  src/utils/quadProgpp/QuadProg++.cc

  src/gait/PhaseScheduler.cpp
  src/gait/FootstepPlanner.cpp
  src/gait/GaitGenerator.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
  "$<INSTALL_INTERFACE:include/unitree_guide_controller/include>"
  PRIVATE
  src
)

ament_target_dependencies(
  ${PROJECT_NAME} PUBLIC
  ${CONTROLLER_DEPENDENCIES}
)

pluginlib_export_plugin_description_file(controller_interface edge_quadruped_controller.xml)

install(
        DIRECTORY include/
        DESTINATION include/${PROJECT_NAME}
)

install(
        TARGETS ${PROJECT_NAME}
        EXPORT export_${PROJECT_NAME}
        ARCHIVE DESTINATION lib/${PROJECT_NAME}
        LIBRARY DESTINATION lib/${PROJECT_NAME}
        RUNTIME DESTINATION bin
)

install(
        DIRECTORY launch rviz config
        DESTINATION share/${PROJECT_NAME}/
)

ament_export_dependencies(${CONTROLLER_INCLUDE_DEPENDS})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
