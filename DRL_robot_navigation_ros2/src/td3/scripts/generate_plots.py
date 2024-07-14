import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to the TensorBoard log directory
log_dir = "/home/tyler/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/runs"

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

# Tags used in the training script for camera and lidar
tags = {
    "camera": ["camera_loss", "camera_AvQ", "camera_MaxQ"],
    "lidar": ["lidar_loss", "lidar_AvQ", "lidar_MaxQ"]
}

data = {}
for sensor, sensor_tags in tags.items():
    data[sensor] = {}
    for tag in sensor_tags:
        steps, values = extract_tb_data(log_dir, tag)
        data[sensor][tag] = values

# Save data to CSV
for sensor, sensor_data in data.items():
    df = pd.DataFrame(sensor_data, index=steps)
    csv_file = os.path.join(log_dir, f"{sensor}_training_metrics.csv")
    df.to_csv(csv_file)
    print(f"{sensor.capitalize()} training metrics saved to {csv_file}")

# Plot the data
plt.figure(figsize=(12, 16))

for i, sensor in enumerate(tags.keys(), 1):
    plt.subplot(3, 2, i)
    plt.plot(steps, data[sensor][f"{sensor}_loss"], label=f'{sensor.capitalize()} Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 2, i + 2)
    plt.plot(steps, data[sensor][f"{sensor}_AvQ"], label=f'{sensor.capitalize()} Average Q')
    plt.xlabel('Iteration')
    plt.ylabel('Average Q')
    plt.legend()

    plt.subplot(3, 2, i + 4)
    plt.plot(steps, data[sensor][f"{sensor}_MaxQ"], label=f'{sensor.capitalize()} Max Q')
    plt.xlabel('Iteration')
    plt.ylabel('Max Q')
    plt.legend()

plt.tight_layout()
plt.show()
