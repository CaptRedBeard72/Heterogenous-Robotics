cmake_minimum_required(VERSION 3.5)
project(td3)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    launch
    models
    scripts
    urdf
    worlds
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
    scripts/replay_buffer.py
    scripts/test_velodyne_node.py
    scripts/combined_test_velodyne_node.py
    scripts/train_velodyne_node.py
    scripts/point_cloud2.py
    scripts/vint_train_node.py
    scripts/vint_test_node.py 
    scripts/combined_vint_test_node.py 
    scripts/combined_test_node.py
    launch/multi_robot_scenario.launch.py
    launch/robot_state_publisher.launch.py
    launch/robot_state_publisher_combined.launch.py
    launch/train_simulation.launch.py
    launch/test_simulation.launch.py
    launch/pioneer3dx.rviz
    launch/pioneer3dx_combined.rviz
    launch/train_vint_simulation.launch.py
    launch/test_vint_simulation.launch.py
    launch/test_combined_simulation.launch.py 
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

