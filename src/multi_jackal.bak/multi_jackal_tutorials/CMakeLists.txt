cmake_minimum_required(VERSION 2.8.3)
project(multi_jackal_tutorials)

find_package(catkin REQUIRED COMPONENTS roslaunch)

catkin_package()

roslaunch_add_file_check(launch/one_jackal.launch)
roslaunch_add_file_check(launch/two_jackal.launch)
roslaunch_add_file_check(launch/three_jackal.launch)
roslaunch_add_file_check(launch/jackal_with_gps.launch)

install(
	DIRECTORY launch rviz worlds
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
