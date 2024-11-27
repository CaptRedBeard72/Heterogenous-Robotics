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

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from rclpy.client import Client
from rclpy.executors import MultiThreadedExecutor

import threading

from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

import time

from torchvision import transforms
from PIL import Image as PILImage

# Parameters
GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.2
TIME_DELTA = 0.1
MAX_DISTANCE_FROM_BOX = 4.0  # Maximum allowed distance from the box
MAX_BOX_GOAL_DISTANCE = 6.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_odom = None
environment_dim = 999
robot_dim = 8
state_dim = environment_dim + robot_dim
camera_data = np.ones((3, 224, 224))  # Placeholder for image data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

last_odom: Odometry = None

# Define ViNT model
class ViNT(nn.Module):
    def __init__(self):
        super(ViNT, self).__init__()
        # Define the ViT structure
        self.embedding_dim = 768
        self.num_classes = 1000  # Number of output classes in ViT

        # Patch Embedding
        self.patch_size = 16
        self.num_patches = (224 // self.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Class and Distillation tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 2, self.embedding_dim))  # +2 for class and distillation tokens

        # Transformer Encoder Layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=12, dim_feedforward=3072),
            num_layers=12
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
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 1024)  # Increased network capacity
        self.layer_2 = nn.Linear(1024, 512)
        self.layer_3 = nn.Linear(512, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s: Tensor) -> Tensor:
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a
    
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.max_action = max_action

    def get_action(self, state: ndarray) -> ndarray:
        state_tensor: torch.Tensor = torch.Tensor(state.reshape(1, -1)).to(device)
        action_tensor: torch.Tensor = self.actor(state_tensor).cpu()
        action_ndarray: ndarray = action_tensor.detach().numpy()
        return action_ndarray.flatten()

    def load(self, filename, directory):
        actor_path = os.path.join(directory, f"{filename}_actor.pth")
        if not os.path.isfile(actor_path):
            raise FileNotFoundError(f"Model file not found: {actor_path}")
        print(f"Loading actor model from: {actor_path}")
        self.actor.load_state_dict(torch.load(actor_path))

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
        self.timeout_duration = 10 * 60
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

        self.camera_data = np.zeros((3, 224, 224))  # Placeholder for image data

        self.box_marker = Marker()
        self.box_marker.header.frame_id = "world"
        self.box_marker.type = Marker.CUBE
        self.box_marker.action = Marker.ADD
        self.box_marker.scale.x = 0.5  # Box dimensions
        self.box_marker.scale.y = 0.5
        self.box_marker.scale.z = 0.5
        self.box_marker.color.a = 1.0  # Alpha
        self.box_marker.color.r = 0.0
        self.box_marker.color.g = 1.0
        self.box_marker.color.b = 0.0

        self.get_logger().info(f"Environment dimension: {self.environment_dim}")
        self.get_logger().info(f"State dimension: {state_dim}")

    def camera_callback(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            img = PILImage.fromarray(img)
            self.camera_data = transform(img).to(device).float()
            self.get_logger().info("Camera data updated.")
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {e}")

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
        global last_odom
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

        robot_position = [self.odom_x, self.odom_y]

        if isinstance(self.camera_data, np.ndarray):
            self.camera_data = torch.tensor(self.camera_data).to(device).float()

        image_features = vint_model(self.camera_data.unsqueeze(0)).squeeze(0)

        box_x = self.box_state.pose.position.x
        box_y = self.box_state.pose.position.y
        box_position = [box_x, box_y]

        collision = self.observe_collision(robot_position, box_position)
        self.collision = collision
        box_moved = self.has_box_moved()
        box_detected = self.is_box_detected(image_features)
        self.box_detected_flag = box_detected

        # Control logic
        if collision:
            self._handle_collision(angle)
        elif box_detected:
            self._approach_and_align(angle)
        else:
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
        
        state = torch.cat((image_features, torch.tensor(robot_state).to(device).float()))

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
        global last_odom, camera_data
        last_odom = None
        camera_data = np.ones((3, 224, 224))  # Initialize with correct size
        self.timeout_start_time = time.time()
        self.collision = False
        self.box_detected_flag = False
        self.previous_box_position = None

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/reset_world service not available, waiting again...')
        try:
            self.reset_proxy.call_async(self.req)
        except rclpy.ServiceException as exc:
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

        time.sleep(TIME_DELTA + 1.0)  # Increased sleep time to allow the simulation to stabilize
        start_time = time.time()
        timeout = 30
        while last_odom is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if last_odom is None:
            self.get_logger().error("Timeout: Odometry data not received after reset.")
            return np.zeros(self.environment_dim + 7), 0, True, False

        box_x = self.box_state.pose.position.x
        box_y = self.box_state.pose.position.y

        distance_to_goal = np.linalg.norm([box_x - self.goal_x, box_y - self.goal_y])
        angle = 0

        robot_state = [
            distance_to_goal,
            angle,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]

        if isinstance(self.camera_data, np.ndarray):
            self.camera_data = torch.tensor(self.camera_data).to(device).float()

        image_features = vint_model(self.camera_data.unsqueeze(0)).squeeze(0)
        self.get_logger().info(f"Image features shape: {image_features.shape}")

        state = torch.cat((image_features, torch.tensor(robot_state).to(device).float()))

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

if __name__ == '__main__':
    rclpy.init(args=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    max_ep = 500
    file_name = "td3_camera"
    result_dir = os.path.expanduser("~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results")
    model_dir = "/home/tyler/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models"  # Update this path to your model directory
    environment_dim = 999
    robot_dim = 8

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    network = TD3(state_dim, action_dim, max_action)
    try:
        network.load(file_name, model_dir)
    except Exception as e:
        raise ValueError(f"Could not load the stored model parameters: {e}")

    done = True
    episode_timesteps = 0

    env = GazeboEnv(environment_dim)
    odom_subscriber = OdomSubscriber()

    executor: MultiThreadedExecutor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(odom_subscriber)
    executor.add_node(env)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = env.create_rate(10)
    
    while rclpy.ok():
        if done:
            state = env.reset()
            done = False
            episode_timesteps = 0
        else:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = env.step(a_in)
            done = 1 if episode_timesteps + 1 == max_ep else int(done)

            state = next_state
            episode_timesteps += 1

    rclpy.shutdown()
    executor_thread.join()
