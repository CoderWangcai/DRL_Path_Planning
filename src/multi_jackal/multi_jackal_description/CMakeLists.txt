cmake_minimum_required(VERSION 2.8.3)
project(multi_jackal_description)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  lms1xx
  nav_msgs
  pointgrey_camera_driver
  robot_state_publisher
  roslaunch
  gazebo_ros
  xacro
  urdf
)

catkin_package()

roslaunch_add_file_check(launch/jackal_description.launch)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
install(PROGRAMS
   scripts/env_run
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY meshes launch urdf
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
