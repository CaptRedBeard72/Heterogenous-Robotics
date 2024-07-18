import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

# Normalize data
normalized_data = {}
for sensor, sensor_data in data.items():
    normalized_data[sensor] = {}
    for tag, values in sensor_data.items():
        max_value = max(values)
        normalized_data[sensor][tag] = [(value / max_value) * 100 for value in values]

# Interpolate velodyne data to match camera steps
interp_velodyne_data = {}
camera_steps = steps_data['camera']
for tag in tags['velodyne']:
    velodyne_interp = interp1d(steps_data['velodyne'], normalized_data['velodyne'][tag], kind='linear', fill_value='extrapolate')
    interp_velodyne_data[tag] = velodyne_interp(camera_steps)

# Save interpolated velodyne data to CSV
interp_velodyne_df = pd.DataFrame(interp_velodyne_data, index=camera_steps)
velodyne_csv_file = os.path.join(log_dirs['velodyne'], f"velodyne_training_metrics_interpolated.csv")
interp_velodyne_df.to_csv(velodyne_csv_file)
print(f"Interpolated velodyne training metrics saved to {velodyne_csv_file}")

# Save normalized camera data to CSV
camera_df = pd.DataFrame(normalized_data['camera'], index=camera_steps)
camera_csv_file = os.path.join(log_dirs['camera'], f"camera_training_metrics_normalized.csv")
camera_df.to_csv(camera_csv_file)
print(f"Camera normalized training metrics saved to {camera_csv_file}")

# Save original normalized velodyne data to CSV
original_velodyne_df = pd.DataFrame(normalized_data['velodyne'], index=steps_data['velodyne'])
original_velodyne_csv_file = os.path.join(log_dirs['velodyne'], f"velodyne_training_metrics_normalized_original.csv")
original_velodyne_df.to_csv(original_velodyne_csv_file)
print(f"Original Velodyne normalized training metrics saved to {original_velodyne_csv_file}")

# Plot the normalized data with interpolated velodyne data
plt.figure(figsize=(12, 12))

# Plot camera data
plt.subplot(3, 1, 1)
plt.plot(camera_steps, normalized_data["camera"]["camera_loss"], label='Camera Loss')
plt.plot(camera_steps, normalized_data["camera"]["camera_AvQ"], label='Camera Average Q')
plt.plot(camera_steps, normalized_data["camera"]["camera_MaxQ"], label='Camera Max Q')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Camera Data')
plt.legend()
plt.grid(True)

# Plot interpolated velodyne data
plt.subplot(3, 1, 2)
plt.plot(camera_steps, interp_velodyne_data["lidar_loss"], label='Velodyne Loss (Interpolated)')
plt.plot(camera_steps, interp_velodyne_data["lidar_AvQ"], label='Velodyne Average Q (Interpolated)')
plt.plot(camera_steps, interp_velodyne_data["lidar_MaxQ"], label='Velodyne Max Q (Interpolated)')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Velodyne Data (Interpolated)')
plt.legend()
plt.grid(True)

# Plot original velodyne data
plt.subplot(3, 1, 3)
plt.plot(steps_data['velodyne'], normalized_data["velodyne"]["lidar_loss"], label='Velodyne Loss')
plt.plot(steps_data['velodyne'], normalized_data["velodyne"]["lidar_AvQ"], label='Velodyne Average Q')
plt.plot(steps_data['velodyne'], normalized_data["velodyne"]["lidar_MaxQ"], label='Velodyne Max Q')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Original Velodyne Data')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
