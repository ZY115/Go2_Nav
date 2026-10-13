cmake_minimum_required(VERSION 3.8)
project(edge_trigger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(edge_msgs REQUIRED)
find_package(edge_ros_interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(visualization_msgs REQUIRED)

ament_export_include_directories(include)

add_library(edge_safety_monitor SHARED
  src/safety_monitor.cpp
)

target_include_directories(edge_safety_monitor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(edge_safety_monitor
  ament_index_cpp
)

target_link_libraries(edge_safety_monitor
  yaml-cpp
)

add_executable(trigger
  src/trigger.cpp
)

target_include_directories(trigger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(trigger
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  edge_msgs
  edge_ros_interfaces
)

target_link_libraries(trigger
  edge_safety_monitor
)

add_executable(virtual_status_indicator_node
  src/virtual_status_indicator_node.cpp
)

ament_target_dependencies(virtual_status_indicator_node
  rclcpp
  std_msgs
  visualization_msgs
  edge_msgs
)

add_executable(test_trigger
  test/trigger_test.cpp
)

target_include_directories(test_trigger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(test_trigger
  rclcpp
  std_msgs
  edge_msgs
  edge_ros_interfaces
)

install(TARGETS
  edge_safety_monitor
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(TARGETS
  trigger
  virtual_status_indicator_node
  test_trigger
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(ament_index_cpp yaml-cpp)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
