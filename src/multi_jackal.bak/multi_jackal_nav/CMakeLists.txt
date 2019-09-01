cmake_minimum_required(VERSION 2.8.3)
project(multi_jackal_nav)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  robot_localization
  roslaunch
  tf
  move_base
)

catkin_package()

roslaunch_add_file_check(launch/jackal_nav.launch)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY
  launch params
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
