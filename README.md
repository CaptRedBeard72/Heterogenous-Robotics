# Goal-Driven Deep RL Policy for Robot Navigation
Deep Reinforcement Learning for mobile robot navigation in ROS2 Gazebo simulator. Using Twin Delayed Deep Deterministic Policy Gradient (TD3) neural network, a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Trained in ROS2 Humble & Gazebo simulator with PyTorch.


# How To Run

Install Python 3.10, ROS2 Humble, Gazebo 11 on Ubuntu 22.04

For Training - 
```
ros2 launch td3 train_simulation.launch.py
```

For Testing - 
```
ros2 launch td3 test_simulation.launch.py
```

# Academic Integrity
If you are currently enrolled in this course, please refer to IIIT-Delhi's Policy on Academic Integrity before referring to any of the repository contents. This repository contains the work we did as undergrads at IIIT-Delhi in CSE-564 Reinforcement Learning course. We do not encourage plagiarism of any kind.

Main files currently being used and updated: hcr.world, train_velodyn_node.py and train_simulation.launch.py
