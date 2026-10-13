cmake_minimum_required(VERSION 3.8)
project(edge_phy_teacher LANGUAGES C CXX)

# ------------------ Compiler flags ------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# ------------------ Dependencies ------------------
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(edge_msgs REQUIRED)
find_package(edge_ros_interfaces REQUIRED)
find_package(edge_trigger REQUIRED)

# ------------------ ecvxcone C library ------------------
set(ecvxcone_headers
  include/ecvxcone/cvxopt.h
  include/ecvxcone/lapack.h
  include/ecvxcone/misc.h
  include/ecvxcone/blas.h
  include/ecvxcone/base.h
  include/ecvxcone/solver.h
  include/ecvxcone/cpg_workspace.h
  include/ecvxcone/cpg_solve.h
)

set(ecvxcone_sources
  src/optimization/ecvxcone/base.c
  src/optimization/ecvxcone/blas.c
  src/optimization/ecvxcone/debug.c
  src/optimization/ecvxcone/dense.c
  src/optimization/ecvxcone/lapack.c
  src/optimization/ecvxcone/misc.c
  src/optimization/ecvxcone/solver.c
  src/optimization/ecvxcone/sparse.c
  src/optimization/ecvxcone/solver_utils.c
  src/optimization/ecvxcone/cpg_solve.c
  src/optimization/ecvxcone/cpg_workspace.c
)

add_library(ecvxcone SHARED ${ecvxcone_sources})
target_include_directories(ecvxcone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/ecvxcone)
set_target_properties(ecvxcone PROPERTIES
  LINKER_LANGUAGE C
  POSITION_INDEPENDENT_CODE ON
)

# ------------------ C++ library ------------------
add_library(lmi_solver_lib
  src/optimization/lmi_gain_solver.cpp
)
target_include_directories(lmi_solver_lib PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(lmi_solver_lib
  ecvxcone
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
)
ament_target_dependencies(lmi_solver_lib
  rclcpp
  std_msgs
)

# ------------------ Executable ------------------

# Physical teacher node
add_executable(phy_teacher src/phy_teacher.cpp)
target_include_directories(phy_teacher PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(phy_teacher
  lmi_solver_lib
)

ament_target_dependencies(phy_teacher rclcpp edge_msgs edge_ros_interfaces edge_trigger)

# Ensure RPATH includes relative lib/
set_target_properties(phy_teacher PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
)

# Gain synthesizer
add_executable(gain_synthesizer src/gain_synthesizer.cpp)
target_include_directories(gain_synthesizer PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(gain_synthesizer lmi_solver_lib)
ament_target_dependencies(gain_synthesizer rclcpp edge_msgs)


# ------------------ Testing ------------------
add_executable(mock_publisher test/mock_publisher.cpp)
ament_target_dependencies( 
  mock_publisher 
  rclcpp 
  std_msgs
  edge_msgs
)

add_executable(control_gain_listener test/control_gain_listener.cpp)
target_link_libraries(control_gain_listener lmi_solver_lib)
ament_target_dependencies(control_gain_listener rclcpp edge_msgs)
target_include_directories(control_gain_listener PUBLIC include)


# ------------------ Install ------------------
install(TARGETS
  phy_teacher
  lmi_solver_lib
  ecvxcone
  mock_publisher
  control_gain_listener
  gain_synthesizer
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY scripts
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ------------------ Testing (Optional) ------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()