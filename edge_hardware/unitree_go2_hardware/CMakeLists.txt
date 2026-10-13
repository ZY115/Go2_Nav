cmake_minimum_required(VERSION 3.8)
project(unitree_go2_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Generate symbols for IDE indexer
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------- Dependencies ----------------
set(PKG_DEPENDENCIES
  edge_msgs
  edge_ros_interfaces
  geometry_msgs
  go2_sport_api
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  unitree_api
)

find_package(ament_cmake REQUIRED)
find_package(backward_ros REQUIRED)
find_package(unitree_sdk2 REQUIRED)

foreach(Dependency IN ITEMS ${PKG_DEPENDENCIES})
  find_package(${Dependency} REQUIRED)
endforeach()

include(GNUInstallDirs)

# ---------------- Library ----------------
add_library(${PROJECT_NAME}
  SHARED
  src/go2_hardware_interface.cpp

  # providers
  src/api/providers/go2_motion_provider.cpp
  src/api/providers/go2_state_provider.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

target_link_libraries(${PROJECT_NAME}
  unitree_sdk2
)

ament_target_dependencies(${PROJECT_NAME}
  ${PKG_DEPENDENCIES}
)

# ros2_control plugin
pluginlib_export_plugin_description_file(
  hardware_interface
  go2_hardware_interface.xml
)

# ---------------- Tests ----------------

# CRC test
add_executable(test_crc32
  test/test_crc32.cpp
)

target_include_directories(test_crc32 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_crc32
  unitree_sdk2
)

# Go2 velocity action executor node
add_executable(go2_vel_action_executor
  src/go2_vel_action_executor.cpp
)

target_link_libraries(go2_vel_action_executor
  ${PROJECT_NAME}
)

ament_target_dependencies(go2_vel_action_executor
  edge_msgs
  rclcpp
)

# Go2 state to edge state bridge node
add_executable(go2_edge_state_bridge
  src/go2_edge_state_bridge.cpp
)

target_link_libraries(go2_edge_state_bridge
  ${PROJECT_NAME}
)

ament_target_dependencies(go2_edge_state_bridge
  edge_msgs
  rclcpp
)

# Go2 real robot executor node
add_executable(go2_real_robot_executor src/go2_real_robot_executor.cpp)
target_include_directories(go2_real_robot_executor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(go2_real_robot_executor
  rclcpp
  edge_msgs
  unitree_api
  go2_sport_api
)

# Go2 cmd_vel executor node
add_executable(go2_cmd_vel_executor src/go2_cmd_vel_executor.cpp)
target_include_directories(go2_cmd_vel_executor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(go2_cmd_vel_executor
  rclcpp
  geometry_msgs
  unitree_api
  go2_sport_api
)

# Go2 state provider comprehensive test
add_executable(test_go2_state_provider
  test/test_go2_state_provider.cpp
)

target_link_libraries(test_go2_state_provider
  ${PROJECT_NAME}
)

ament_target_dependencies(test_go2_state_provider
  edge_msgs
  rclcpp
)

# Range obstacle test
add_executable(test_range_obstacle
  test/test_range_obstacle.cpp
)

target_link_libraries(test_range_obstacle
  ${PROJECT_NAME}
)

# Comprehensive sport mode state test
add_executable(test_sport_mode_state
  test/test_sport_mode_state.cpp
)

target_link_libraries(test_sport_mode_state
  ${PROJECT_NAME}
)


# ---------------- Install ----------------

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS
    test_crc32
    go2_vel_action_executor
    go2_edge_state_bridge
    go2_real_robot_executor
    go2_cmd_vel_executor
    test_go2_state_provider
    test_range_obstacle
    test_sport_mode_state
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# ---------------- Lint ----------------

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

# ---------------- Export ----------------

ament_export_dependencies(${PKG_DEPENDENCIES})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

ament_package()