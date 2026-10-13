cmake_minimum_required(VERSION 3.8)
project(edge_ros_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(edge_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(test_conversion src/test_conversion.cpp)
ament_target_dependencies(test_conversion
  rclcpp
  edge_msgs
)

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS test_conversion DESTINATION lib/${PROJECT_NAME})
ament_export_include_directories(include ${EIGEN3_INCLUDE_DIRS})
ament_export_dependencies(Eigen3 edge_msgs rclcpp)


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
