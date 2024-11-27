import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Paths to .json files
file_paths = {
    "Baseline": "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/velodyne_experiments/baseline/baseline.json",
    "Different Activation": "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/velodyne_experiments/different_activation/different_activation.json",
    "Tuned Learning Rate": "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/velodyne_experiments/tuned_learning_rate/tuned_learning_rate.json"
}

# Load data from files
data = {}
for label, path in file_paths.items():
    with open(os.path.expanduser(path), "r") as file:
        data[label] = json.load(file)

# Metrics to analyze
success_criteria_metrics = ["avg_reward", "successful_pushes", "collisions", "goals_reached", "box_detected"]
training_metrics = ["train_loss", "train_avg_Q", "train_max_Q"]

# Compute averages
def compute_averages(dataset, metrics):
    return {metric: sum(dataset[metric]) / len(dataset[metric]) for metric in metrics}

# Prepare comparison tables
success_criteria_averages = {label: compute_averages(dataset, success_criteria_metrics) for label, dataset in data.items()}
training_averages = {label: compute_averages(dataset, training_metrics) for label, dataset in data.items()}

# Convert to DataFrames
success_criteria_table = pd.DataFrame(success_criteria_averages).T  # Transpose for readability
training_table = pd.DataFrame(training_averages).T  # Transpose for readability

# Function to save a DataFrame as a PNG table
def save_table_as_png(dataframe, title, filename):
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust size as needed
    ax.axis("off")  # Hide axes
    ax.axis("tight")  # Adjust layout tightly around the table
    table = ax.table(
        cellText=dataframe.round(2).values,
        colLabels=dataframe.columns,
        rowLabels=dataframe.index,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(dataframe.columns))))  # Adjust column widths
    plt.title(title, fontsize=14, pad=20)
    plt.savefig(filename, bbox_inches="tight", dpi=300)  # Save as PNG with high resolution
    plt.close(fig)

# Save both tables as PNG
save_table_as_png(success_criteria_table, "Success Criteria Averages", "success_criteria_averages.png")
save_table_as_png(training_table, "Training Metrics Averages", "training_metrics_averages.png")

print("Tables saved as PNG files: success_criteria_averages.png and training_metrics_averages.png")
