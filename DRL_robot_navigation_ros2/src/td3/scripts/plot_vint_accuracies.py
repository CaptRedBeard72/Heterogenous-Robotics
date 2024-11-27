import os
import matplotlib.pyplot as plt
import json

# Paths to .json files
file_paths = {
    "Baseline": "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/vint_experiments/baseline/evaluations_baseline.json",
    "Different Activation": "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/vint_experiments/different_activation/evaluations_different_activation.json",
    "Tuned Learning Rate": "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/vint_experiments/tuned_learning_rate/evaluations_tuned_learning_rate.json"
}

# Load data from files
data = {}
for label, path in file_paths.items():
    with open(os.path.expanduser(path), "r") as file:
        data[label] = json.load(file)

# Define styles
markers = {"Baseline": "o", "Different Activation": "s", "Tuned Learning Rate": "d"}  # Marker per experiment
colors_and_styles = {
    "avg_reward": ("blue", "-"),
    "successful_pushes": ("green", "--"),
    "collisions": ("red", "-."),
    "goals_reached": ("purple", ":"),
    "box_detected": ("orange", "-"),
}
scaling_factors = {
    "avg_reward": 100,
    "successful_pushes": 10,
    "collisions": 10,
    "goals_reached": 1,  # No scaling for goals
    "box_detected": 10,
}

# Plot Success Criteria Data
plt.figure(figsize=(16, 12))
for label, dataset in data.items():
    epochs = list(range(len(dataset["avg_reward"])))
    for metric, (color, line_style) in colors_and_styles.items():
        scaling_factor = scaling_factors[metric]
        scaled_data = [value / scaling_factor for value in dataset[metric]]
        plt.plot(
            epochs,
            scaled_data,
            label=f"{label} - {metric.replace('_', ' ').title()} (Scaled)" if scaling_factor != 1 else f"{label} - {metric.replace('_', ' ').title()}",
            color=color,
            marker=markers[label],
            linestyle=line_style,
        )

plt.xlabel("Epoch")
plt.ylabel("Success Criteria (Scaled)")
plt.title("Success Criteria Comparison Across Experiments")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Define styles for training metrics
colors_and_styles_training = {
    "train_loss": ("blue", "-"),
    "train_avg_Q": ("green", "--"),
    "train_max_Q": ("red", "-."),
}

# Plot Loss and Q-Values
plt.figure(figsize=(16, 12))
for label, dataset in data.items():
    epochs = list(range(len(dataset["train_loss"])))
    for metric, (color, line_style) in colors_and_styles_training.items():
        plt.plot(
            epochs,
            dataset[metric],
            label=f"{label} - {metric.replace('_', ' ').title()}",
            color=color,
            marker=markers[label],
            linestyle=line_style,
        )

plt.xlabel("Episode")
plt.ylabel("Training Metrics")
plt.title("Loss and Q-Value Comparison Across Experiments")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
