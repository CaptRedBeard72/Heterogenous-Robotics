import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Paths to the TensorBoard log directories
log_dirs = {
    "camera": "/home/tyler/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/runs/camera",
    "velodyne": "/home/tyler/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/runs/velodyne"
}

# Function to list all available tags in the log directory
def list_available_tags(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc.Tags()['scalars']

# Function to extract data from TensorBoard logs
def extract_tb_data(log_dir, tag):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Check if the tag is in the log file
    if tag not in event_acc.Tags()['scalars']:
        raise ValueError(f"Tag {tag} not found in {log_dir}")

    # Extract scalar values
    scalars = event_acc.Scalars(tag)
    steps = [x.step for x in scalars]
    values = [x.value for x in scalars]

    return steps, values

# Tags used in the training script for camera and velodyne
tags = {
    "camera": ["camera_loss", "camera_AvQ", "camera_MaxQ"],
    "velodyne": ["lidar_loss", "lidar_AvQ", "lidar_MaxQ"]
}

data = {}
steps_data = {}

for sensor, log_dir in log_dirs.items():
    print(f"\nAvailable tags in the {sensor} log directory:")
    available_tags = list_available_tags(log_dir)
    for tag in available_tags:
        print(tag)

    data[sensor] = {}
    steps_data[sensor] = None
    for tag in tags[sensor]:
        steps, values = extract_tb_data(log_dir, tag)
        data[sensor][tag] = values
        if steps_data[sensor] is None:
            steps_data[sensor] = steps

# Save data to CSV
for sensor, sensor_data in data.items():
    df = pd.DataFrame(sensor_data, index=steps_data[sensor])
    csv_file = os.path.join(log_dirs[sensor], f"{sensor}_training_metrics.csv")
    df.to_csv(csv_file)
    print(f"{sensor.capitalize()} training metrics saved to {csv_file}")

# Plot the data
plt.figure(figsize=(12, 16))

for i, sensor in enumerate(tags.keys()):
    plt.subplot(3, 2, i*3 + 1)
    plt.plot(steps_data[sensor], data[sensor][f"{tags[sensor][0]}"], label=f'{sensor.capitalize()} Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 2, i*3 + 2)
    plt.plot(steps_data[sensor], data[sensor][f"{tags[sensor][1]}"], label=f'{sensor.capitalize()} Average Q')
    plt.xlabel('Iteration')
    plt.ylabel('Average Q')
    plt.legend()

    plt.subplot(3, 2, i*3 + 3)
    plt.plot(steps_data[sensor], data[sensor][f"{tags[sensor][2]}"], label=f'{sensor.capitalize()} Max Q')
    plt.xlabel('Iteration')
    plt.ylabel('Max Q')
    plt.legend()

plt.tight_layout()
plt.show()
