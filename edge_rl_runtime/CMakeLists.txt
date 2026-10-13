cmake_minimum_required(VERSION 3.8)
project(edge_rl_runtime)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(edge_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(edge_ros_interfaces REQUIRED)
find_package(control_input_msgs REQUIRED)

# Reward composer library
add_library(reward_composer SHARED
  src/reward_composer.cpp
)

target_include_directories(reward_composer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(reward_composer
  yaml-cpp
)

ament_target_dependencies(reward_composer
  rclcpp
  edge_msgs
  edge_ros_interfaces
)


# RL runtime node
add_executable(rl_runtime_node
  src/rl_runtime_node.cpp
  src/deadzone_discriminator.cpp
)

target_include_directories(rl_runtime_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(rl_runtime_node
  reward_composer
)

# ensure modern C++ features
target_compile_features(rl_runtime_node PRIVATE cxx_std_17)

ament_target_dependencies(rl_runtime_node
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  edge_msgs
  edge_ros_interfaces
  control_input_msgs
)

install(TARGETS
  rl_runtime_node
  DESTINATION lib/${PROJECT_NAME}
)

# install(
#   DIRECTORY include/
#   DESTINATION include
# )

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

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
