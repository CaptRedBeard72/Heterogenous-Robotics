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
from threading import Lock

from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

import time
from torchviz import make_dot

from torchvision import transforms
from PIL import Image as PILImage
from sensor_msgs.msg import Image


# Parameters
GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.2
TIME_DELTA = 0.1
MAX_DISTANCE_FROM_BOX = 4.0  # Maximum allowed distance from the box
MAX_BOX_GOAL_DISTANCE = 6.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

last_odom = None

camera_data = np.ones((3, 224, 224))  # Placeholder for image data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

last_odom: Odometry = None

# Define ViNT model
class ViNT(nn.Module):
    def __init__(self):
        super(ViNT, self).__init__()
        # Define the ViT structure
        self.embedding_dim = 512
        self.num_classes = 1000  # Number of output classes in ViT

        # Patch Embedding
        self.patch_size = 16
        self.num_patches = (128 // self.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Class and Distillation tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 2, self.embedding_dim))  # +2 for class and distillation tokens

        # Transformer Encoder Layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=6
        )

        # MLP Head for Classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.num_classes)
        )

        # Distillation Head for knowledge distillation
        self.dist_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.num_classes)
        )


    def forward(self, x):
        # Patch Embedding
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)

        # Add Class and Distillation tokens
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        dist_tokens = self.dist_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.pos_embedding

        # Transformer Encoder
        x = self.transformer(x)

        # Separate outputs for classification and distillation heads
        cls_output = self.mlp_head(x[:, 0])
        dist_output = self.dist_head(x[:, 1])

        return cls_output, dist_output

vint_model = ViNT().to(device)

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
        s = s.float()
        a = a.float()

        # Ensure tensors are 2D
        assert s.dim() == 2, f"State tensor must be 2D, got shape {s.shape}"
        assert a.dim() == 2, f"Action tensor must be 2D, got shape {a.shape}"

        # print(f"[Critic Forward] State shape: {s.shape}")
        # print(f"[Critic Forward] Action shape: {a.shape}")

        # Concatenate state and action
        sa = torch.cat([s, a], dim=-1)
        # print(f"[Critic Forward] Concatenated State-Action shape: {sa.shape}")

        # Forward pass
        q1 = sa
        for layer in self.q1_layers:
            q1 = self.activation(layer(q1))
        q1 = self.q1_output(q1)

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
        # Ensure state is converted to tensor on the correct device
        state_tensor: torch.Tensor = torch.tensor(state, device=device).float().reshape(1, -1)
        
        action_tensor: torch.Tensor = self.actor(state_tensor).cpu()
        action_ndarray: ndarray = action_tensor.detach().numpy()
        return action_ndarray.flatten()
    
    def train(
        self,
        replay_buffer: ReplayBuffer,
        iterations: int,
        batch_size: int = 16,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.1,
        noise_clip: float = 0.3,
        policy_freq: int = 2,
    ):
        av_Q = 0
        max_Q = float('-inf')
        av_loss = 0

        # Handle edge cases
        if iterations <= 0:
            self.logger.warning("No iterations provided for training. Skipping training step.")
            return av_loss, av_Q, max_Q

        if replay_buffer.size() < batch_size:
            self.logger.warning(f"Replay buffer too small: {replay_buffer.size()} (required: {batch_size}). Skipping training.")
            return av_loss, av_Q, max_Q

        for it in range(iterations):
            # Sample a batch of transitions from the replay buffer
            batch = replay_buffer.sample_batch(batch_size)

            # Ensure the sampled batch is converted to float32 arrays
            batch_states = np.vstack([np.array(s, dtype=np.float32) for s in batch[0]])
            batch_next_states = np.vstack([np.array(s, dtype=np.float32) for s in batch[4]])
            batch_actions = np.vstack([np.array(a, dtype=np.float32) for a in batch[1]])
            batch_rewards = np.array(batch[2], dtype=np.float32)
            batch_dones = np.array(batch[3], dtype=np.float32)

            # Convert to PyTorch tensors
            state = torch.tensor(batch_states, device=device)
            next_state = torch.tensor(batch_next_states, device=device)
            action = torch.tensor(batch_actions, device=device)
            reward = torch.tensor(batch_rewards, device=device).unsqueeze(1)
            done = torch.tensor(batch_dones, device=device).unsqueeze(1)

            # Compute target Q-value
            next_action = self.actor_target(next_state)
            noise = torch.randn_like(next_action, device=device) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-values
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * discount * target_Q.detach()

            # Update metrics
            av_Q += target_Q.mean().item()
            max_Q = max(max_Q, target_Q.max().item())

            # Get current Q-values
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            av_loss += loss.item()

            # Optimize critic
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # Update actor and target networks
            if it % policy_freq == 0:
                actor_loss = -self.critic(state, self.actor(state))[0].mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                # Update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Average metrics over iterations
        av_Q /= iterations
        av_loss /= iterations

        return av_loss, av_Q, max_Q

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename)))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename)))

class GazeboEnv(Node):
    def __init__(self):
        super().__init__('env')
        self.reset_lock = Lock()
        self.camera_processing_active = True 
        self.environment_dim = None
        self.robot_dim = None
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
        self.model = vint_model.to(device)
        self.camera_data_thread = threading.Thread(target=self.spin_camera_data, daemon=True)
        self.camera_data_thread.start()
        
        # Add projection layer
        self.robot_state_projection = nn.Linear(7, 512).to(device)

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

        self.camera_data = torch.zeros((1, vint_model.embedding_dim), device=device)  # Initialize as tensor

        self.camera_subscriber = self.create_subscription(
            Image, "/camera1/image_raw", self.camera_callback, 10
        )
        self.get_logger().info("Subscribed to /camera/image_raw")


    def camera_callback(self, msg):
        # self.get_logger().info("Camera callback triggered")
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            img = PILImage.fromarray(img)
            img_tensor = transform(img).to(device).float().unsqueeze(0)  # Add batch dimension

            # Extract features using ViNT
            cls_output, _ = self.model(img_tensor)  # Use only the classification output
            self.camera_data = cls_output
            # self.get_logger().info(f"Camera data updated: {cls_output.shape}")
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {e}")

    def spin_camera_data(self):
        self.get_logger().info(f"spin_camera_data Thread ID: {threading.get_ident()}")
        self.get_logger().info("Camera data thread running.")
        single_executor = rclpy.executors.SingleThreadedExecutor()

        while rclpy.ok() and self.camera_processing_active:
            try:
                single_executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                self.get_logger().error(f"Error in spin_once: {e}")
            time.sleep(0.1)

    def model_states_callback(self, msg: ModelStates):
        try:
            if "target_box" in msg.name:
                index = msg.name.index("target_box")
                self.box_state.pose = msg.pose[index]

            else:
                self.get_logger().warning("Box model not found in the model states.")
        except Exception as e:
            self.get_logger().error(f"Error in model_states_callback: {e}")

    def is_box_detected(self, image_features, detection_threshold=2.0):
        detected = torch.any(image_features > detection_threshold).item()
        return detected

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

    def observe_collision(self, robot_position, box_position):
        collision_distance = 0.47  # Based on half of the box size (0.5 / 2)
        distance = np.linalg.norm(np.array(robot_position) - np.array(box_position))
        if distance < collision_distance:
            # self.get_logger().info("Box touched!")
            return True
        return False
    
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
        global last_odom
        done = False

        # self.get_logger().info(f"step Thread ID: {threading.get_ident()}")

        # Ensure odometry data is available
        if last_odom is None:
            start_time = time.time()
            timeout = 10
            while last_odom is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
            if last_odom is None:
                self.get_logger().warning("Odometry data not available within timeout. Terminating step.")
                return np.zeros(self.environment_dim + self.robot_dim), 0, True, False

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
        current_box_to_goal_distance = np.linalg.norm([box_x - self.goal_x, box_y - self.goal_y])
        distance_to_box = np.linalg.norm([self.odom_x - box_x, self.odom_y - box_y])

        if self.previous_box_to_goal_distance is None:
            self.previous_box_to_goal_distance = current_box_to_goal_distance

        robot_position = [self.odom_x, self.odom_y]

        # Use ViNT features from the camera data
        if isinstance(self.camera_data, np.ndarray):
            self.get_logger().warning("Invalid camera data detected. Using default features.")
            image_features = torch.zeros((1, vint_model.embedding_dim), device=device)
        else:
            image_features = self.camera_data.view(1, -1)

        collision = self.observe_collision(robot_position, [box_x, box_y])
        self.collision = collision
        box_moved = self.has_box_moved()
        box_detected = self.is_box_detected(image_features)
        self.box_detected_flag = box_detected

        # Control logic
        if collision:
            # self.get_logger().info("Collision detected, handling collision.")
            self._handle_collision(angle)
        elif box_detected:
            # self.get_logger().info("Box detected, aligning and approaching.")
            self._approach_and_align(angle)
        else:
            # self.get_logger().info("No collision or box detection. Exploring environment.")
            self._explore(action)

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

        # Log termination conditions
        if timeout:
            self.get_logger().info("Episode terminated due to timeout.")
            done = True
        if reached_goal:
            self.get_logger().info("Episode terminated as goal was reached.")
            done = True
        if box_too_far:
            self.get_logger().info("Episode terminated as box moved too far from the goal.")
            done = True
        if robot_too_far:
            self.get_logger().info("Episode terminated as robot moved too far from the box.")
            done = True

        # Construct state
        robot_state = [
            current_box_to_goal_distance,
            angle,
            distance_to_box,
            box_x - self.odom_x,
            box_y - self.odom_y,
            action[0],
            action[1],
        ]
        robot_state_tensor = torch.tensor(robot_state).to(device).float().unsqueeze(0)
        projected_robot_state = self.robot_state_projection(robot_state_tensor)

        state = torch.cat((image_features, projected_robot_state), dim=1).view(1, -1)
        state = state.detach()

        return state, reward, done, reached_goal

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
        global last_odom, camera_data

        self.get_logger().info(f"reset Thread ID: {threading.get_ident()}")
        with self.reset_lock:
            self.get_logger().info("Resetting environment...")
        
        self.camera_processing_active = False

        # Call the /reset_world service
        self.get_logger().info("Calling /reset_world service...")
        reset_request = Empty.Request()
        future = self.reset_proxy.call_async(reset_request)
        
        # Wait for the reset service to complete
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)  # Add timeout
            if future.result() is not None:
                self.get_logger().info("/reset_world service called successfully.")
            else:
                self.get_logger().error("Failed to call /reset_world service.")
                return None
        except rclpy.executors.TimeoutException:
            self.get_logger().error("Timeout while waiting for /reset_world service.")
            return None
        
        time.sleep(2.0)

        last_odom = None
        camera_data = np.zeros((3, 224, 224))
        self.timeout_start_time = time.time()
        self.collision = False
        self.box_detected_flag = False
        self.previous_box_position = None
        self.previous_box_to_goal_distance = None
        
        self.get_logger().info("Camera processing paused for reset.")

        if not self.call_service(self.pause):
            self.get_logger().error("Failed to pause physics. Reset incomplete.")
            return None

        time.sleep(TIME_DELTA + 1.0)

        # Reset positions
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

        if not self.call_service(self.unpause):
            self.get_logger().error("Failed to unpause physics. Reset incomplete.")
            return None

        time.sleep(TIME_DELTA + 1.0)

        # Ensure odometry data is received
        start_time = time.time()
        timeout = 30
        while last_odom is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if last_odom is None:
            self.get_logger().warning("Odometry data not received after reset. Using default state.")
        else:
            self.get_logger().info("Odometry data received after reset")

        # Validate positions
        box_distance_to_goal = np.linalg.norm([
            self.box_state.pose.position.x - self.goal_x,
            self.box_state.pose.position.y - self.goal_y
        ])
        if box_distance_to_goal > MAX_BOX_GOAL_DISTANCE:
            self.get_logger().warning("Box reset incorrectly. Re-initializing...")
            self.box_state.pose.position.x = 3.0
            self.box_state.pose.position.y = 0.0
            self.set_state.publish(self.box_state)
            time.sleep(1.0)

        self.camera_processing_active = True
        self.get_logger().info("Camera processing resumed after reset.")

        # Construct initial state
        distance_to_goal = np.linalg.norm([self.box_state.pose.position.x - self.goal_x, self.box_state.pose.position.y - self.goal_y])
        
        angle = 0

        robot_state = [
            distance_to_goal,
            angle,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        
        if isinstance(self.camera_data, torch.Tensor):
            image_features = self.camera_data.view(1, -1)
        else:
            self.get_logger().warning("Camera data invalid after reset. Using default features.")
            image_features = torch.zeros((1, vint_model.embedding_dim), device=device)

        robot_state_tensor = torch.tensor(robot_state).to(device).float().unsqueeze(0)
        projected_robot_state = self.robot_state_projection(robot_state_tensor)

        if self.environment_dim is None or self.robot_dim is None:
            self.environment_dim = image_features.numel()  # Total elements in image_features
            self.robot_dim = projected_robot_state.numel()  # Total elements in projected_robot_state
            self.get_logger().info(f"[DEBUG] Calculated Environment Dim: {self.environment_dim}")
            self.get_logger().info(f"[DEBUG] Calculated Robot Dim: {self.robot_dim}")

        state = torch.cat((image_features, projected_robot_state), dim=1)
        self.get_logger().info(f"[DEBUG] Calculated State Dim: {self.environment_dim + self.robot_dim}")

        self.get_logger().info("Environment reset complete.")

        return state

    def call_service(self, service: Client) -> bool:
        for attempt in range(5):  # Retry up to 5 times
            if not service.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Service not available, retrying...')
                continue
            try:
                future: Future = service.call_async(Empty.Request())
                rclpy.spin_until_future_complete(self, future)
                return True  # Success
            except Exception as e:
                self.get_logger().warning(f"Service call failed: {e}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
        self.get_logger().error("Service call failed after multiple attempts.")
        return False

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
        # self.get_logger().info("Odometry data received.")


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
            action = network.get_action(state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state)
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
            f"~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/vint_experiments/{config.get('experiment_name', 'default')}"
        )
        os.makedirs(experiment_dir, exist_ok=True)

        # Pass the configuration and experiment directory to `main`
        main(config, experiment_dir)
        print(f"Experiment completed. Results saved in {experiment_dir}.")

# Update visualize_network to accept state_dim
def visualize_network(vint_model: ViNT):
    """
    Visualizes the ViNT network architecture and saves the structure as a PNG file.
    """
    result_dir = os.path.expanduser(
        "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results"
    )
    os.makedirs(result_dir, exist_ok=True)

    # ViNT model visualization
    sample_input = torch.randn(4, 3, 128, 128).to(device)  # Example input for ViNT model
    output = vint_model(sample_input)

    # Save the visualization
    vint_file_path = os.path.join(result_dir, "vint_network_architecture")
    make_dot(output[0], params=dict(vint_model.named_parameters())).render(
        vint_file_path, format="png"
    )
    print(f"ViNT network architecture visualization saved as: {vint_file_path}.png")

def main(config, experiment_dir, args=None):
    rclpy.init(args=args)

    seed = 0
    eval_freq = 5000
    max_ep = 500
    eval_ep = 10
    max_timesteps = 5000000
    expl_noise = 0.25
    expl_decay_steps = 500000
    expl_min = 0.1
    batch_size = 100
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    buffer_size = 1000000
    file_name = config.get("experiment_name", "td3_camera")
    save_model = True
    load_model = False
    start_time = time.time()

    hyperparams = config["hyperparameters"]
    
    result_dir = os.path.expanduser("~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results")
    model_dir = os.path.expanduser("~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models/{config.get('experiment_name', 'default')}")
    experiment_dir = os.path.expanduser("~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results/vint_experiments")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    action_dim = 2
    max_action = 1

    # Debug dimensions
    # print(f"[Main] State dimension: {state_dim}")
    # print(f"[Main] Action dimension: {action_dim}")

    env = GazeboEnv()  

    env.reset()
    print(f"[DEBUG] Environment Dim: {env.environment_dim}, Robot Dim: {env.robot_dim}", flush=True)
    
    state_dim = env.environment_dim + env.robot_dim 

    print(f"[DEBUG] State Dim: {state_dim}", flush=True)
    
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

    visualize_network(vint_model)

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

    executor: MultiThreadedExecutor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(odom_subscriber)
    executor.add_node(env)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    print(f"Executor Thread ID: {executor_thread.ident}")
    print("Starting executor thread.")
    executor_thread.start()
    print(f"Executor Thread Started with ID: {threading.get_ident()}")
    print("Executor thread started.")

    rate = env.create_rate(10)

    try:
        while rclpy.ok() and timestep < max_timesteps:
            
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if elapsed_time >= 4 * 3600:  
                print("Training stopped: Exceeded 4-hour time limit.")
                break

            if done:
                env.get_logger().info(f"[DEBUG] Done flag set. Timestep: {timestep}")

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
                        env.get_logger().info("Starting evaluation.")
                        avg_reward, successful_pushes, col, goal_reached, avg_box_detected = evaluate(env, network, epoch, eval_episodes=eval_ep)
                        env.get_logger().info(
                            f"[DEBUG] Evaluation Results -> Avg Reward: {avg_reward}, Success: {successful_pushes}, Collisions: {col}, Goals: {goal_reached}, Box Detected: {avg_box_detected}"
                        )
                        evaluations["avg_reward"].append(avg_reward)
                        evaluations["successful_pushes"].append(successful_pushes)
                        evaluations["collisions"].append(col)
                        evaluations["goals_reached"].append(goal_reached)
                        evaluations["box_detected"].append(avg_box_detected)

                    if save_model:
                        experiment_name = config.get("experiment_name", "td3_camera")
                        network.save(f"{experiment_name}_td3", model_dir)
                    epoch += 1
                    env.get_logger().info(f"[DEBUG] Epoch incremented to: {epoch}")

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

            action = network.get_action(state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state)
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
                # print(f"[DEBUG] Incremented timestep: {timestep}", flush=True)
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
        rclpy.shutdown()
        executor_thread.join()

if __name__ == "__main__":
    config_file = os.path.expanduser(
        "~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/json_configs/vint_experiments.json"
    )
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")
    run_experiments(config_file)