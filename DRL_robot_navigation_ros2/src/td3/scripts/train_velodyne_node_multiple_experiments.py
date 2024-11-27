#!/usr/bin/env python3

import os
import time
import json

import numpy as np
from numpy import ndarray

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from replay_buffer import ReplayBuffer

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from rclpy.client import Client
from rclpy.executors import MultiThreadedExecutor

import threading
import math

from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from point_cloud2 import detect_objects
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
import point_cloud2 as pc2

import time
# Parameters
GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.2
TIME_DELTA = 0.1
MAX_DISTANCE_FROM_BOX = 4.0  # Maximum allowed distance from the box
MAX_BOX_GOAL_DISTANCE = 6.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_odom = None
environment_dim = 75
velodyne_data = np.ones(environment_dim) * 10

last_odom: Odometry = None

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation_function):
        super(Actor, self).__init__()
        self.activation = getattr(F, activation_function)
        self.layers = nn.ModuleList()

        # Create dynamic hidden layers
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.output_layer = nn.Linear(input_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s: Tensor) -> Tensor:
        for layer in self.layers:
            s = self.activation(layer(s))
        a = self.tanh(self.output_layer(s))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation_function):
        super(Critic, self).__init__()

        self.activation = getattr(F, activation_function)
        self.q1_layers = nn.ModuleList()
        self.q2_layers = nn.ModuleList()

        # Create dynamic hidden layers for Q1
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_layers:
            self.q1_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.q1_output = nn.Linear(input_dim, 1)

        # Create dynamic hidden layers for Q2
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_layers:
            self.q2_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.q2_output = nn.Linear(input_dim, 1)

    def forward(self, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        sa = torch.cat([s, a], dim=-1)  # Concatenate state and action

        # Compute Q1
        q1 = sa
        for layer in self.q1_layers:
            q1 = self.activation(layer(q1))
        q1 = self.q1_output(q1)

        # Compute Q2
        q2 = sa
        for layer in self.q2_layers:
            q2 = self.activation(layer(q2))
        q2 = self.q2_output(q2)

        return q1, q2

class TD3:
    def __init__(self, state_dim, action_dim, max_action, logger, actor_lr, critic_lr, actor_config, critic_config):
        # Actor and Actor Target
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=actor_config["hidden_layers"],
            activation_function=actor_config["activation_function"]
        ).to(device)
        self.actor_target = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=actor_config["hidden_layers"],
            activation_function=actor_config["activation_function"]
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # Initialize target network
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic and Critic Target
        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=critic_config["hidden_layers"],
            activation_function=critic_config["activation_function"]
        ).to(device)
        self.critic_target = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=critic_config["hidden_layers"],
            activation_function=critic_config["activation_function"]
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # Initialize target network
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.iter_count = 0
        self.logger = logger

    def get_action(self, state: ndarray) -> ndarray:
        state_tensor: torch.Tensor = torch.Tensor(state.reshape(1, -1)).to(device)
        action_tensor: torch.Tensor = self.actor(state_tensor).cpu()
        action_ndarray: ndarray = action_tensor.detach().numpy()
        return action_ndarray.flatten()
    
    def train(
        self,
        replay_buffer: ReplayBuffer,
        iterations: int,
        batch_size: int = 64,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.1,
        noise_clip: float = 0.3,
        policy_freq: int = 2,
    ):
        av_Q = 0
        max_Q = float('-inf')
        av_loss = 0

        for it in range(iterations):
            # Sample a batch of transitions from the replay buffer
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states, batch_indices = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Compute target Q-value
            next_action: torch.Tensor = self.actor_target(next_state)
            noise: torch.Tensor = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-values
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q: torch.Tensor = torch.min(target_Q1, target_Q2)
            discounted_future: torch.Tensor = (1 - done) * discount * target_Q
            target_Q: torch.Tensor = reward + discounted_future.detach()

            # Update metrics   
            av_Q += torch.mean(target_Q).item()
            max_Q = max(max_Q, torch.max(target_Q).item())

            # Get current Q-values
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute TD errors
            td_errors = (target_Q - current_Q1).abs().detach().cpu().numpy()

            # Compute critic loss
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            av_loss += loss.item()

            # Optimize critic
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Gradient clipping
            self.critic_optimizer.step()

            # Update replay buffer priorities
            replay_buffer.update_priorities(batch_indices, td_errors)

            # Update actor and target networks
            if it % policy_freq == 0:
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Gradient clipping
                self.actor_optimizer.step()

                # Update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Average metrics over iterations
        av_Q /= iterations
        av_loss /= iterations

        # Log metrics
        self.iter_count += 1
        self.logger.info(f"Training Iteration: {self.iter_count}, Avg Loss: {av_loss}, Avg Q: {av_Q}, Max Q: {max_Q}")

        return av_loss, av_Q, max_Q

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename)))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename)))

class GazeboEnv(Node):
    def __init__(self, environment_dim):
        super().__init__('env')
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0
        self.goal_x = 3.5
        self.goal_y = 3.5
        self.collision = False
        self.box_detected_flag = False
        self.previous_box_position = None
        self.timeout_start_time = time.time()
        self.timeout_duration = 5 * 60
        self.previous_box_to_goal_distance = None 


        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.box_state = ModelState()
        self.box_state.model_name = "target_box"
        self.target_position = [self.goal_x, self.goal_y]

        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request()

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)

        self.model_states_subscriber = self.create_subscription(ModelStates, "/gazebo/model_states", self.model_states_callback, 10)

        self.start_time = None
        self.last_box_detection_time = 0
        self.box_detection_cooldown = 5
        self.timeout_occurred = False

        self.create_timer(1.0, self.publish_goal_marker)

        self.get_logger().info(f"Environment dimension: {self.environment_dim}")
        self.get_logger().info(f"State dimension: {self.environment_dim + 7}")

    def model_states_callback(self, msg: ModelStates):
        try:
            if "target_box" in msg.name:
                index = msg.name.index("target_box")
                self.box_state.pose = msg.pose[index]
                # self.get_logger().info(f"Updated box state: {self.box_state.pose}")
            else:
                self.get_logger().warning("Box model not found in the model states.")
        except Exception as e:
            self.get_logger().error(f"Error in model_states_callback: {e}")

    def is_box_detected(self, velodyne_data, detection_range=2.0):
        # Identify points within detection range
        indices = np.where(velodyne_data < detection_range)[0]
        if len(indices) < 5:  # Minimum points to consider as a box
            return False

        # Compute cluster center as the average of detected points
        angles = np.linspace(-np.pi, np.pi, len(velodyne_data))
        x_coords = velodyne_data[indices] * np.cos(angles[indices])
        y_coords = velodyne_data[indices] * np.sin(angles[indices])

        # Use the centroid as the box center approximation
        box_center_x = np.mean(x_coords)
        box_center_y = np.mean(y_coords)

        # self.get_logger().info(f"Box Center Detected: ({box_center_x}, {box_center_y})")
        self.box_detected_flag = True
        return box_center_x, box_center_y

    def has_box_moved(self):
        current_box_position = [self.box_state.pose.position.x, self.box_state.pose.position.y]
        
        if self.previous_box_position is None:
            self.previous_box_position = current_box_position
            return False  # No movement detected on the first check
        
        # Calculate movement magnitude
        box_movement = np.linalg.norm(np.array(current_box_position) - np.array(self.previous_box_position))
        box_moved = box_movement > 0.01  # Increase threshold to 1 cm for floating-point precision

        # Log debug information
        # self.get_logger().info(f"Previous Box Position: {self.previous_box_position}")
        # self.get_logger().info(f"Current Box Position: {current_box_position}")
        # self.get_logger().info(f"Movement Magnitude: {box_movement}")
        # self.get_logger().info(f"Box Moved Detected: {box_moved}")

        # Update previous position only if movement is detected
        if box_moved:
            self.previous_box_position = current_box_position

        return box_moved

    def observe_collision(self, laser_data):
        min_laser = min(laser_data)
        # self.get_logger().info(f"Min laser distance: {min_laser}")
        if min_laser < COLLISION_DIST:
            # self.get_logger().info("Box touched!")
            return True, min_laser
        return False, min_laser
    
    def _handle_collision(self, angle):
        box_x = self.box_state.pose.position.x
        box_y = self.box_state.pose.position.y
        direction_to_goal = np.arctan2(self.goal_y - box_y, self.goal_x - box_x)

        angle_diff = direction_to_goal - angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.5  # Move forward with constant speed
        vel_cmd.angular.z = 1 * -angle_diff  # Adjust orientation towards the goal
        self.vel_pub.publish(vel_cmd)

    def _approach_and_align(self, angle):
        self.box_detected_flag = True
        box_x = self.box_state.pose.position.x
        box_y = self.box_state.pose.position.y
        direction_to_box = np.arctan2(box_y - self.odom_y, box_x - self.odom_x)

        angle_diff = direction_to_box - angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.5  # Move forward with constant speed
        vel_cmd.angular.z = 1 * -angle_diff  # Adjust orientation towards the box
        self.vel_pub.publish(vel_cmd)

    def _explore(self, action, clusters=None):
        vel_cmd = Twist()
        if clusters:
            # Choose the closest cluster as the exploration target
            closest_cluster = min(clusters, key=lambda c: np.linalg.norm(c["center"][:2]))
            target_angle = np.arctan2(closest_cluster["center"][1], closest_cluster["center"][0])
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = target_angle * 0.5
        else:
            vel_cmd.linear.x = float(action[0])
            vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)

    def step(self, action):
        """
        Executes one step in the environment based on the provided action.
        """
        global last_odom, velodyne_data
        done = False

        # Ensure odometry data is available
        if last_odom is None:
            start_time = time.time()
            timeout = 10
            while last_odom is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
            if last_odom is None:
                return np.zeros(self.environment_dim + 7), 0, True, False

        # Update robot and box positions
        self.odom_x = last_odom.pose.pose.position.x
        self.odom_y = last_odom.pose.pose.position.y

        quaternion = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        box_x = self.box_state.pose.position.x
        box_y = self.box_state.pose.position.y

        current_box_to_goal_distance = np.linalg.norm(
            [box_x - self.goal_x, box_y - self.goal_y]
        )
        distance_to_box = np.linalg.norm([self.odom_x - box_x, self.odom_y - box_y])

        if self.previous_box_to_goal_distance is None:
            self.previous_box_to_goal_distance = current_box_to_goal_distance

        collision, _ = self.observe_collision(velodyne_data)
        self.collision = collision
        box_moved = self.has_box_moved()
        box_detected = self.is_box_detected(velodyne_data)
        self.box_detected_flag = box_detected

        # Control logic
        if collision:
            self._handle_collision(angle)
        elif box_detected:
            self._approach_and_align(angle)
        else:
            self._explore(action)

        # Simulate environment update
        self.call_service(self.unpause)
        time.sleep(TIME_DELTA)
        self.call_service(self.pause)

        # Compute reward and check termination conditions
        timeout = (time.time() - self.timeout_start_time) > self.timeout_duration
        reached_goal = current_box_to_goal_distance < GOAL_REACHED_DIST
        box_too_far = current_box_to_goal_distance > MAX_BOX_GOAL_DISTANCE
        robot_too_far = distance_to_box > MAX_DISTANCE_FROM_BOX

        reward = self.get_reward(
            reached_goal=reached_goal,
            timeout=timeout,
            box_moved=box_moved,
            current_box_to_goal_distance=current_box_to_goal_distance,
        )

        done = timeout or reached_goal or box_too_far or robot_too_far

        # Construct state
        laser_state = velodyne_data[:self.environment_dim]
        robot_state = [
            current_box_to_goal_distance,
            angle,
            distance_to_box,
            box_x - self.odom_x,
            box_y - self.odom_y,
            action[0],
            action[1],
        ]

        state = np.concatenate((laser_state, robot_state))

        # Log termination conditions
        if timeout:
            self.get_logger().info("Episode terminated due to timeout.")
        if reached_goal:
            self.get_logger().info("Episode terminated as goal was reached.")
        if box_too_far:
            self.get_logger().info("Episode terminated as box moved too far from the goal.")
        if robot_too_far:
            self.get_logger().info("Episode terminated as robot moved too far from the box.")

        # current_box_position = np.array([box_x, box_y])
        # box_movement = np.linalg.norm(current_box_position - self.previous_box_position)
        # self.get_logger().info(f"Box Movement: {box_movement}")

        # progress = self.previous_box_to_goal_distance - current_box_to_goal_distance
        # self.get_logger().info(f"Box Progress: {progress}")

        # self.get_logger().info(f"Action received in step: {action}")

        return state, reward, done, current_box_to_goal_distance < GOAL_REACHED_DIST

    def get_reward(self, reached_goal, timeout, box_moved, current_box_to_goal_distance):
        reward = 0.0
        progress = 0.0

        if reached_goal:
            reward = 500.0
        elif timeout:
            reward = -20.0
        elif box_moved:
            progress = self.previous_box_to_goal_distance - current_box_to_goal_distance
            reward += max(2.0, 10.0 * progress)  # Minimum reward for any movement

        elif self.box_detected_flag:
            reward += 5.0  # Small reward for detecting the box
        else:
            reward = -0.05  # Penalize lack of meaningful action

        # self.get_logger().info(f"Reward: {reward}, Progress: {progress}")
        return reward

    def reset(self):
        global last_odom, velodyne_data
        last_odom = None
        velodyne_data = np.ones(self.environment_dim) * 10  # Initialize with correct size
        self.timeout_start_time = time.time()  # Reset the timeout timer at reset
        self.collision = False  # Reset the collision flag
        self.box_detected_flag = False

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/reset_world service not available, waiting again...')
        try:
            self.reset_proxy.call_async(self.req)
        except Exception as exc:
            self.get_logger().error(f'Service call failed: {exc}')

        self.box_state.pose.position.x = 3.0
        self.box_state.pose.position.y = 0.0
        self.box_state.pose.position.z = 0.25

        self.set_state.publish(self.box_state)

        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.01
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.set_state.publish(self.set_self_state)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(self.req)
        except Exception as exc:
            self.get_logger().error(f'Service call failed: {exc}')

        time.sleep(TIME_DELTA)
        start_time = time.time()
        timeout = 30
        while last_odom is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if last_odom is None:
            self.get_logger().error("Timeout: Odometry data not received after reset.")
            return np.zeros(self.environment_dim + 7), 0, True, False

        laser_state = velodyne_data[:self.environment_dim]

        box_x = self.box_state.pose.position.x
        box_y = self.box_state.pose.position.y

        distance_to_goal = np.linalg.norm([box_x - self.goal_x, box_y - self.goal_y])
        angle = 0  # Reset the robot's orientation to 0

        robot_state = [
            distance_to_goal,
            angle,
            0.0,  # Reset distance to box
            0.0,  # Reset x difference to box
            0.0,  # Reset y difference to box
            0.0,  # Reset linear action
            0.0   # Reset angular action
        ]

        self.get_logger().info(f"Box state after reset: {self.box_state.pose}")
        self.get_logger().info(f"Robot state after reset: {self.set_self_state.pose}")

        state = np.concatenate((laser_state, robot_state))

        return state

    def call_service(self, service: Client) -> None:
        while not service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            # self.get_logger().info(f"Calling service: {service.srv_name}")
            future: Future = service.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future)
            # self.get_logger().info(f"Service {service.srv_name} call completed successfully.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def publish_goal_marker(self):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class VelodyneSubscriber(Node):
    def __init__(self, env: GazeboEnv, ground_threshold=-0.2):
        super().__init__('velodyne_subscriber')
        self.env = env  # Reference to GazeboEnv
        self.environment_dim = env.environment_dim
        self.ground_threshold = ground_threshold
        self.subscription = self.create_subscription(PointCloud2, "/velodyne_points", self.velodyne_callback, 10)

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim])
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        velodyne_data = np.ones(self.environment_dim) * 10  # Reset data

        # Process Velodyne point cloud to calculate laser data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        for i in range(len(data)):
            if data[i][2] > -0.2:  # Filter ground points
                dot = data[i][0]
                mag1 = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2)
                value = dot / mag1
                value = np.clip(value, -1.0, 1.0)
                beta = math.acos(value) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

        # # Log raw points
        points = np.array(list(pc2.read_points(v, skip_nans=True, field_names=("x", "y", "z"))))
        # self.get_logger().info(f"Raw points detected: {len(points)}")

        # # Existing logic for ground filtering
        filtered_points = points[(points[:, 2] > self.ground_threshold) & 
                                (np.linalg.norm(points[:, :2], axis=1) < 4.0)]
        # self.get_logger().info(f"Filtered points: {len(filtered_points)}")

        if len(filtered_points) == 0:
        #     self.get_logger().info("No valid points after filtering.")
            return

        # # Detect clusters
        clusters = detect_objects(v, detection_range=4.0, eps=0.2, min_samples=10)
        # self.get_logger().info(f"Detected {len(clusters)} potential objects.")
        # for idx, cluster in enumerate(clusters):
        #     self.get_logger().info(f"Cluster {idx}: Center={cluster['center']}, Size={cluster['size']}")

        # Update exploration if no box is detected
        if not self.env.box_detected_flag and clusters:
            self.env._explore(action=None, clusters=clusters)


def evaluate(env: GazeboEnv, network: TD3, epoch: int, eval_episodes: int = 10):
    """
    Evaluates the agent's performance over several episodes.
    Metrics:
    - Average Reward: Mean cumulative reward per episode.
    - Successful Pushes: Number of times the box was successfully pushed.
    - Collisions: Count of collisions during evaluation.
    - Goals Reached: Number of episodes where the goal was reached.
    - Box Detected: Number of times the box was detected across episodes.
    """
    avg_reward = 0.0
    collisions = 0
    goals_reached = 0
    successful_pushes = 0
    box_detected_count = 0

    for episode in range(eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, reached_goal = env.step(a_in)

            episode_reward += reward
            if env.collision:
                collisions += 1
            if reached_goal:
                goals_reached += 1
            if env.has_box_moved():
                successful_pushes += 1
            if env.box_detected_flag:
                box_detected_count += 1

        avg_reward += episode_reward
        env.get_logger().info(
            f"Episode {episode + 1}/{eval_episodes}: Reward={episode_reward}, "
            f"Collisions={collisions}, Goals Reached={goals_reached}, "
            f"Successful Pushes={successful_pushes}, Box Detected={box_detected_count}"
        )

    avg_reward /= eval_episodes
    avg_collisions = collisions / eval_episodes
    avg_successful_pushes = successful_pushes / eval_episodes
    avg_box_detected = box_detected_count / eval_episodes

    env.get_logger().info(f"Evaluation Results Epoch {epoch}:")
    env.get_logger().info(
        f"Average Reward: {avg_reward}, Collisions: {avg_collisions}, "
        f"Successful Pushes: {avg_successful_pushes}, Goals Reached: {goals_reached}, "
        f"Box Detected: {avg_box_detected}"
    )

    return avg_reward, avg_successful_pushes, avg_collisions, goals_reached, avg_box_detected

def load_experiments(config_file):
    """
    Loads all experiments from a single JSON configuration file.
    """
    with open(config_file, "r") as f:
        data = json.load(f)
    return data["experiments"]  # Assuming the experiments are stored under a key called "experiments"

def run_experiments(config_file):
    """
    Loads all experiments from a JSON file and runs each experiment sequentially.
    """
    experiments = load_experiments(config_file)  # Load experiments from the JSON file

    for config in experiments:
        print(f"Running experiment: {config.get('experiment_name', 'Unnamed Experiment')}")

        # Define the directory for the experiment's results
        experiment_dir = os.path.expanduser(
            f"~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/velodyne_experiments/{config.get('experiment_name', 'default')}"
        )
        os.makedirs(experiment_dir, exist_ok=True)

        # Pass the configuration and experiment directory to `main`
        main(config, experiment_dir)
        print(f"Experiment completed. Results saved in {experiment_dir}.")

def main(config, experiment_dir, args=None):
    rclpy.init(args=args)

    seed = 0
    eval_freq = 5000
    max_ep = 500
    eval_ep = 10
    max_timesteps = 5000000
    expl_noise = 0.25 # Starting exploration noise
    expl_decay_steps = 500000
    expl_min = 0.1
    batch_size = 100
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    buffer_size = 1000000
    file_name = config.get("experiment_name", "td3_velodyne")
    save_model = True
    load_model = False
    start_time = time.time()
    # random_near_obstacle = True

    hyperparams = config["hyperparameters"]
    
    model_dir = os.path.expanduser("~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models/{config.get('experiment_name', 'default')}")
    experiment_dir = os.path.expanduser("~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/velodyne_experiments")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    environment_dim = 75
    robot_dim = 7

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    env = GazeboEnv(environment_dim)

    network = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        logger=env.get_logger(),
        actor_lr=hyperparams["learning_rate_actor"],
        critic_lr=hyperparams["learning_rate_critic"],
        actor_config=config["actor_network"],
        critic_config=config["critic_network"]
    )

    replay_buffer = ReplayBuffer(buffer_size, seed)

    if load_model:
        try:
            network.load(file_name, model_dir)
        except Exception as e:
            print(f"Could not load the stored model parameters, initializing training with random parameters: {e}")

    evaluations = {
        "avg_reward": [],
        "successful_pushes": [],
        "collisions": [],
        "goals_reached": [],
        "box_detected": [],
        "train_loss": [],
        "train_avg_Q": [],
        "train_max_Q": []
    }

    timestep = 0
    timesteps_since_eval = 0
    done = True
    epoch = 1

    # count_rand_actions = 0
    # random_action = []

    odom_subscriber = OdomSubscriber()

    velodyne_subscriber = VelodyneSubscriber(env)

    executor: MultiThreadedExecutor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)
    executor.add_node(env)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = env.create_rate(10)

    try:
        while rclpy.ok() and timestep < max_timesteps:
            
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if elapsed_time >= 4 * 3600:  
                print("Training stopped: Exceeded 4-hour time limit.")
                break

            if done:
                if timestep != 0:
                    av_loss, av_Q, max_Q = network.train(
                        replay_buffer,
                        episode_timesteps,
                        batch_size,
                        discount,
                        tau,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                    )

                    evaluations["train_loss"].append(av_loss)
                    evaluations["train_avg_Q"].append(av_Q)
                    evaluations["train_max_Q"].append(max_Q) 

                    if timesteps_since_eval >= eval_freq:
                        timesteps_since_eval %= eval_freq
                        avg_reward, successful_pushes, col, goal_reached, avg_box_detected = evaluate(env, network, epoch, eval_episodes=eval_ep)
                        evaluations["avg_reward"].append(avg_reward)
                        evaluations["successful_pushes"].append(successful_pushes)
                        evaluations["collisions"].append(col)
                        evaluations["goals_reached"].append(goal_reached)
                        evaluations["box_detected"].append(avg_box_detected)


                    if save_model:
                        experiment_name = config.get("experiment_name", "td3_velodyne")
                        network.save(f"{experiment_name}_td3", model_dir)
                    epoch += 1

                try:
                    state = env.reset()
                    done = False
                    episode_reward = 0
                    episode_timesteps = 0
                    env.box_detected_flag = False
                except Exception as e:
                    env.get_logger().error(f"Error during environment reset: {e}")
                    continue

            if expl_noise > expl_min:
                expl_noise -= (expl_noise - expl_min) / (2 * expl_decay_steps)

            if timestep % 1000 == 0:
                print(f"Exploration noise: {expl_noise}")

            action = network.get_action(np.array(state))
            action = np.clip(action + np.random.normal(0, expl_noise, size=action_dim), -max_action, max_action)

            # Remove the condition for random actions near obstacles
            a_in = [(action[0] + 1) / 2, action[1]]
            try:
                next_state, reward, done, reached_goal = env.step(a_in)
                episode_reward += reward

                if reached_goal:
                    print("Goal reached!")

                done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
                replay_buffer.add(state, action, reward, done_bool, next_state)

                state = next_state
                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1

            except Exception as e:
                env.get_logger().error(f"Error during environment step: {e}")
                done = True

            rate.sleep()

    except KeyboardInterrupt:
        pass

    finally:

        # Save evaluations to a file
        
        experiment_name = config.get("experiment_name", "default")
        evaluation_file_name = f"evaluations_{experiment_name}.json"
        evaluation_npy_file_name = f"evaluations_{experiment_name}.npy"
        np.save(os.path.join(experiment_dir, evaluation_npy_file_name), evaluations)
        with open(os.path.join(experiment_dir, evaluation_file_name), "w") as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=4)
        
        env.destroy_node()
        odom_subscriber.destroy_node()
        velodyne_subscriber.destroy_node()
        rclpy.shutdown()
        executor_thread.join()

if __name__ == "__main__":
    config_file = os.path.expanduser(
        "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/json_configs/experiments.json"
    )
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")
    run_experiments(config_file)