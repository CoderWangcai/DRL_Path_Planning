# Multi-Jackal Simulator using Gazebo ROS

The ROS documentation can be found [here](http://wiki.ros.org/multi_jackal_tutorials).
# Overview
These packages make use of the robotic simulator Gazebo, along with the Jackal 
robot description. Multiple Jackals are spawned and are able to be moved 
independently. The difference between this package and the [Jackal package](https://github.com/jackal/jackal), 
is that multiple Jackals are able to be spawned without causing overlap. 
Significant amounts of code are from Clearpaths Jackal package, this just has 
minor changes.

If you only want to simulate one, then follow the 
[guide](https://www.clearpathrobotics.com/assets/guides/jackal/simulation.html). 
The problem is that it isn't scalable. They use the same transformation tree and 
some message names. You can see the problem yourself if you spawn two and have a 
look at the topics and TF tree.

# Files
## multi_jackal_tutorials
The starting point for simulating the robots. Contains launch and config files.
Starts up a Gazebo session and launches robots using `multi_jackal_base`.
Example: `roslaunch multi_jackal_tutorials one_jackal.launch`.

## multi_jackal_base
Contains a single launch file that calls all other jackal components.

## multi_jackal_control
Launches the velocity controller plugin and robot controls.

## multi_jackal_description
Creates a plugin for publishing robot states and transformations. Loads a 
parameter that describes the robot for use in Gazebo.

## multi_jackal_nav
Creates the localisation and move_base nodes.

# Running
Make sure the file `multi_jackal_description/scripts/env_run` is executable.

Example launch files can be found in `multi_jackal_tutorials/launch`. Gazebo and RVIZ 
can be viewed with `gzclient` and `roslaunch multi_jackal_tutorials rviz.launch`.
