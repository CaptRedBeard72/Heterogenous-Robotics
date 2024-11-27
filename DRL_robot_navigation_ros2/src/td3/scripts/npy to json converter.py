#!/usr/bin/env python3

import os
import numpy as np
import json

def npy_to_json(npy_file, json_file):
    # Load the .npy file
    data = np.load(npy_file, allow_pickle=True).item()  # Use `.item()` if it's a dictionary-like object

    # Save to a .json file
    with open(json_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Converted {npy_file} to {json_file}")

# File paths and output locations
file_directories = {
    "baseline": "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/velodyne_experiments/baseline",
    "different_activation": "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/velodyne_experiments/different_activation",
    "tuned_learning_rate": "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/velodyne_experiments/tuned_learning_rate"
}

npy_files = {
    "baseline": "baseline.npy",
    "different_activation": "different_activation.npy",
    "tuned_learning_rate": "tuned_learning_rate.npy"
}

# Iterate over files and convert
for key, dir_path in file_directories.items():
    dir_path = os.path.expanduser(dir_path)  # Expand `~` to full path
    npy_file = os.path.join(dir_path, npy_files[key])
    json_file = os.path.splitext(npy_file)[0] + ".json"

    npy_to_json(npy_file, json_file)
