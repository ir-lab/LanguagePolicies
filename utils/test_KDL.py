# Check if KDL is able to import
# If not, the installed solver version doesn't match the pyton bindings version
import PyKDL as kdl

# Let's see if KDL can create a frame
# If not, might be a dynamic linking issue of the PyKDL.so file
frame = kdl.Frame()

# Let's load the robot
from collect_data import Robot
# This should print a forward kinematics result for the robot's base joints
robot = Robot(test=True)
# Get some joints to play with
joints = robot.base_joints
# Well, now, can we call the same function as a class method
frame = robot.getTcpFromAngles(joints)