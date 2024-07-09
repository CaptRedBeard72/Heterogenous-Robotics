# Heterogenous Collaborative Goal-Driven Deep RL Policy for Robot Navigation
Deep Reinforcement Learning for mobile robot navigation in ROS2 Gazebo simulator. Using Twin Delayed Deep Deterministic Policy Gradient (TD3) neural network and Visual Navigation Transformer (ViNT), two robots (one with a Lidar sensor and the other with a cameral sensor) learn to detect a target and take that target to a designated goal point in a simulated environment while avoiding obstacles. Trained in ROS2 Humble & Gazebo simulator with PyTorch. 

# Future Goals
Put both robots together in a single simulated world and perform training/testing while the opposite robots sensor data is being read by the other (i.e. Lidar using Camera data and vice-versa).


# How To Run

Install Python 3.10, ROS2 Humble, Gazebo 11 on Ubuntu 22.04

For Training - 
```
ros2 launch td3 train_simulation.launch.py
ros2 launch td3 train_vint_simulation.launch.py
```

For Testing - 
```
ros2 launch td3 test_simulation.launch.py
ros2 launch td3 test_vint_simulation.launch.py
ros2 launch td3 test_combined_simulation.launch.py
```

Original unmodified codes referenced here:
```
https://github.com/vishweshvhavle/deep-rl-navigation

```
Additional resources:
```
https://github.com/LantaoYu/MARL-Papers
https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file
