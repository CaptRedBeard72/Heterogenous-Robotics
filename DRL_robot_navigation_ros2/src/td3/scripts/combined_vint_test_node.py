#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer

import rclpy
from rclpy.node import Node
import threading
import math
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from torchvision import transforms
from PIL import Image as PILImage

# Parameters
GOAL_REACHED_DIST = 0.5
TIME_DELTA = 0.2
MAX_DISTANCE_FROM_BOX = 4.0  # Maximum allowed distance from the box

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

# Define ViNT model
class ViNT(nn.Module):
    def __init__(self, pretrained=True):
        super(ViNT, self).__init__()
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=pretrained)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

vint_model = ViNT().to(device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a
    
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.max_action = max_action

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        actor_path = os.path.join(directory, f"{filename}_actor.pth")
        if not os.path.isfile(actor_path):
            raise FileNotFoundError(f"Model file not found: {actor_path}")
        print(f"Loading actor model from: {actor_path}")
        self.actor.load_state_dict(torch.load(actor_path))

class GazeboEnv(Node):
    def __init__(self, environment_dim, namespace):
        super().__init__('env')
        self.environment_dim = environment_dim
        self.namespace = namespace
        self.odom_x = 0
        self.odom_y = 0
        self.goal_x = 3.5
        self.goal_y = 3.5
        self.collision = False
        self.box_detected_flag = False
        self.previous_box_position = None
        self.timeout_start_time = time.time()
        self.timeout_duration = 120

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = -1.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.box_state = ModelState()
        self.box_state.model_name = "target_box"
        self.target_position = [self.goal_x, self.goal_y]

        self.vel_pub = self.create_publisher(Twist, f"/{self.namespace}/cmd_vel", 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request()

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.box_marker_pub = self.create_publisher(Marker, 'box_marker', 10)

        self.model_states_subscriber = self.create_subscription(ModelStates, "/gazebo/model_states", self.model_states_callback, 10)
        self.camera_subscriber = self.create_subscription(Image, "/camera/image_raw", self.camera_callback, 10)

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

    def model_states_callback(self, msg):
        try:
            if "target_box" in msg.name:
                index = msg.name.index("target_box")
                self.box_state.pose = msg.pose[index]

                # Update the marker's position
                self.box_marker.header.stamp = self.get_clock().now().to_msg()
                self.box_marker.pose = self.box_state.pose
                self.box_marker_pub.publish(self.box_marker)

                # self.get_logger().info(f"Box state updated: ({self.box_state.pose.position.x}, {self.box_state.pose.position.y})")
            else:
                self.get_logger().warning("Box model not found in the model states.")
        except Exception as e:
            self.get_logger().error(f"Error in model_states_callback: {e}")

    def is_box_detected(self, image_features, detection_threshold=3.0):
        # Check if any feature exceeds the threshold
        detected = torch.any(image_features > detection_threshold).item()
        # self.get_logger().info(f"Box detection status: {detected}, Threshold: {detection_threshold}")
        return detected

    def has_box_moved(self):
        current_box_position = [self.box_state.pose.position.x, self.box_state.pose.position.y]
        if self.previous_box_position is None:
            self.previous_box_position = current_box_position
            return False
        box_moved = np.linalg.norm(np.array(current_box_position) - np.array(self.previous_box_position)) > 0.01
        self.previous_box_position = current_box_position
        return box_moved

    def observe_collision(self, robot_position, box_position):
        collision_distance = 0.47  # Based on half of the box size (0.5 / 2)
        distance = np.linalg.norm(np.array(robot_position) - np.array(box_position))
        # self.get_logger().info(f"Distance to box: {distance}, Collision threshold: {collision_distance}")
        if distance < collision_distance:
            self.get_logger().info("Box touched!")
            return True
        return False

    def step(self, action):
        global last_odom
        done = False
        box_detected = False

        if last_odom is None:
            start_time = time.time()
            timeout = 10
            while last_odom is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
            if last_odom is None:
                return np.zeros(state_dim), 0, True, False

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

        robot_position = [self.odom_x, self.odom_y]

        if isinstance(self.camera_data, np.ndarray):
            self.camera_data = torch.tensor(self.camera_data).to(device).float()

        image_features = vint_model(self.camera_data.unsqueeze(0)).squeeze(0)
        # self.get_logger().info(f"Image features: {image_features[:10]}")  # Log first 10 features for brevity

        box_detected = self.is_box_detected(image_features)
        # self.get_logger().info(f"Box detected: {box_detected}")

        box_x = self.box_state.pose.position.x
        box_y = self.box_state.pose.position.y
        box_position = [box_x, box_y]

        # self.get_logger().info(f"Calling observe_collision with robot_position: {robot_position} and box_position: {box_position}")
        collision = self.observe_collision(robot_position, box_position)
        # self.get_logger().info(f"observe_collision returned: {collision}")

        if collision:
            box_x = self.box_state.pose.position.x
            box_y = self.box_state.pose.position.y
            direction_to_goal = np.arctan2(self.goal_y - box_y, self.goal_x - box_x)

            # Normalize the angle difference to the range [-pi, pi]
            angle_diff = direction_to_goal - angle
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            vel_cmd = Twist()
            vel_cmd.linear.x = 0.5  # Move forward with constant speed
            vel_cmd.angular.z = 1 * -angle_diff  # Adjust orientation towards the goal
            self.vel_pub.publish(vel_cmd)
        elif box_detected:
            self.box_detected_flag = True
            box_x = self.box_state.pose.position.x
            box_y = self.box_state.pose.position.y
            direction_to_box = np.arctan2(box_y - self.odom_y, box_x - self.odom_x)

            # Normalize the angle difference to the range [-pi, pi]
            angle_diff = direction_to_box - angle
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            vel_cmd = Twist()
            vel_cmd.linear.x = 0.5  # Move forward with constant speed
            vel_cmd.angular.z = 1 * -angle_diff  # Adjust orientation towards the box
            self.vel_pub.publish(vel_cmd)
        else:
            self.box_detected_flag = False
            vel_cmd = Twist()
            vel_cmd.linear.x = float(action[0])
            vel_cmd.angular.z = float(action[1])
            self.vel_pub.publish(vel_cmd)
            
        self.call_service(self.unpause)
        time.sleep(TIME_DELTA)
        self.call_service(self.pause)

        distance_to_goal = np.linalg.norm([box_x - self.goal_x, box_y - self.goal_y])
        distance_to_box = np.linalg.norm([self.odom_x - box_x, self.odom_y - box_y])

        robot_state = [
            distance_to_goal,
            angle,
            distance_to_box,
            box_x - self.odom_x,
            box_y - self.odom_y,
            action[0],
            action[1]
        ]

        state = torch.cat((image_features, torch.tensor(robot_state).to(device).float()))

        box_detected = self.is_box_detected(image_features)

        reached_goal = distance_to_goal < GOAL_REACHED_DIST
        box_moved = self.has_box_moved()
        current_time = time.time()

        if current_time - self.timeout_start_time > self.timeout_duration:
            self.get_logger().info("Timeout: The robot did not achieve the goal in the given time.")
            done = True
            reward = self.get_reward(
                reached_goal, 
                collision=collision, 
                timeout=True, 
                box_detected=box_detected, 
                current_time=current_time, 
                box_moved=box_moved, 
                distance_to_box=distance_to_box
            )
        else:
            reward = self.get_reward(
                reached_goal, 
                collision=collision, 
                timeout=False, 
                box_detected=box_detected, 
                current_time=current_time, 
                box_moved=box_moved, 
                distance_to_box=distance_to_box
            )

        if distance_to_box > MAX_DISTANCE_FROM_BOX:
            self.get_logger().info("Penalty: Robot too far from box -300 points.")
            done = True
            reward -= 300.0

        if reached_goal:
            done = True

        return state.detach().cpu().numpy(), reward, done, reached_goal

    def get_reward(self, target, collision, timeout, box_detected, current_time, box_moved, distance_to_box):
        reward = 0.0

        if target:
            self.get_logger().info("Reward: Target reached +1000 points!")
            reward = 1000.0
        elif timeout:
            self.get_logger().info("Penalty: Timeout occurred -500 points")
            reward = -500.0
        elif collision and box_moved:
            self.get_logger().info("Reward: Box contact +200 points")
            reward = 200.0
        elif box_detected and (current_time - self.last_box_detection_time) > self.box_detection_cooldown:
            self.get_logger().info("Reward: Box detected +20 points")
            self.last_box_detection_time = current_time
            reward = 20.0
        else:
            reward += max(0, (2.0 - distance_to_box) * 5)

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
        self.set_self_state.pose.position.y = -1.0
        self.set_self_state.pose.position.z = 0.01
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
        self.set_state.publish(self.set_self_state)

        self.get_logger().info(f"Reset: Robot initial position: ({self.set_self_state.pose.position.x}, {self.set_self_state.pose.position.y})")
        self.get_logger().info(f"Reset: Box initial position: ({self.box_state.pose.position.x}, {self.box_state.pose.position.y})")

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(self.req)
        except rclpy.ServiceException as exc:
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

        return state.detach().cpu().numpy()

    def call_service(self, service):
        while not service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            future = service.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future)
        except rclpy.ServiceException as e:
            self.get_logger().error(f"Service call failed: {e}")

    def publish_goal_marker(self):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
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
    def __init__(self, namespace):
        super().__init__('odom_subscriber')
        self.namespace = namespace
        self.subscription = self.create_subscription(Odometry, f'/{self.namespace}/odom', self.odom_callback, 10)

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

if __name__ == '__main__':
    rclpy.init(args=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    max_ep = 500
    file_name = "td3_camera"  # Use the LiDAR model actor data
    model_dir = "/home/tyler/ros2_ws/src/deep-rl-navigation/DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models"  # Update this path to your model directory
    environment_dim = 999
    robot_dim = 8
    namespace = "robot_camera"

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

    env = GazeboEnv(environment_dim, namespace)
    odom_subscriber = OdomSubscriber(namespace)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(env)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(2)

    while rclpy.ok():
        if done:
            state = env.reset()
            done = False
            episode_timesteps = 0
        else:
            action = network.get_action(np.array(state))
            print(f"Action: {action}")  # Debug the action taken
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = env.step(a_in)
            done = 1 if episode_timesteps + 1 == max_ep else int(done)

            state = next_state
            episode_timesteps += 1

    rclpy.shutdown()
    executor_thread.join()
