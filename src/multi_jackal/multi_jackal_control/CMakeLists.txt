cmake_minimum_required(VERSION 2.8.3)
project(multi_jackal_control)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  controller_manager
  interactive_marker_twist_server
  gazebo_ros_control
  hector_gazebo_plugins
  joint_state_controller
  diff_drive_controller
  roslaunch
)

catkin_package()

roslaunch_add_file_check(launch/jackal_control.launch)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY
  config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
