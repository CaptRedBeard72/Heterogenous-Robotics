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
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
import point_cloud2 as pc2
from torchvision import transforms
from PIL import Image as PILImage

# Parameters
GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.47
TIME_DELTA = 0.1
MAX_DISTANCE_FROM_BOX = 4.0  # Maximum allowed distance from the box

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_odom = None
camera_data = np.ones((3, 224, 224))
environment_dim = 75
velodyne_data = np.ones(environment_dim) * 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ViNT(nn.Module):
    def __init__(self, pretrained=True):
        super(ViNT, self).__init__()
        torch.hub._hub_dir = '/tmp/torch/hub'
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

def is_box_detected(velodyne_data, detection_range=3.0):
    detection_threshold = sum(velodyne_data < detection_range)
    detected = detection_threshold > (0.05 * len(velodyne_data))  # Adjust the threshold as needed
    return detected

class GazeboEnv(Node):
    def __init__(self, params):
        robot_name = params['model_name']
        odom_topic = params['odometry_topic']
        vel_topic = params['vel_topic']
        pointcloud_topic = params['pointcloud_topic']
        camera_topic = params['camera_topic']
        environment_dim = params['environment_dim']
        robot_dim = params['robot_dim']

        valid_node_name = robot_name.replace('/', '_').replace(':', '_')
        super().__init__(valid_node_name + '_env', namespace=robot_name)
        self.environment_dim = environment_dim
        self.robot_dim = robot_dim
        self.odom_x = 0
        self.odom_y = 0
        self.goal_x = params['goal_x']
        self.goal_y = params['goal_y']
        self.collision = False
        self.box_detected_flag = False
        self.previous_box_position = None
        self.timeout_start_time = time.time()
        self.timeout_duration = 120

        self.get_logger().info(f"Initializing {robot_name} with odom_topic {odom_topic} and vel_topic {vel_topic}")

        self.set_self_state = ModelState()
        self.set_self_state.model_name = robot_name
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

        self.vel_pub = self.create_publisher(Twist, vel_topic, 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request()

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)

        self.model_states_subscriber = self.create_subscription(ModelStates, "/gazebo/model_states", self.model_states_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.lidar_initialized = False
        self.camera_initialized = False

        if pointcloud_topic:
            self.lidar_initialized = True
            self.pointcloud_subscriber = self.create_subscription(PointCloud2, pointcloud_topic, self.velodyne_callback, 10)
            self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
            for m in range(environment_dim - 1):
                self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim])
            self.gaps[-1][-1] += 0.03

        if camera_topic:
            self.camera_initialized = True
            self.camera_subscriber = self.create_subscription(Image, camera_topic, self.camera_callback, 10)

        self.start_time = None
        self.last_box_detection_time = 0
        self.box_detection_cooldown = 5
        self.timeout_occurred = False

        self.create_timer(1.0, self.publish_goal_marker)

        self.get_logger().info(f"Environment dimension: {self.environment_dim}")
        state_dim = self.environment_dim + self.robot_dim
        self.get_logger().info(f"State dimension: {state_dim}")

    def camera_callback(self, msg):
        global camera_data
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            img = PILImage.fromarray(img)
            camera_data = transform(img).to(device).float()
            self.get_logger().info("Camera data updated.")
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {e}")

    def model_states_callback(self, msg):
        try:
            if "target_box" in msg.name:
                index = msg.name.index("target_box")
                self.box_state.pose = msg.pose[index]
            else:
                self.get_logger().warning("Box model not found in the model states.")
        except Exception as e:
            self.get_logger().error(f"Error in model_states_callback: {e}")

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

    def velodyne_callback(self, v):
        global velodyne_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                value = dot / mag1
                value = np.clip(value, -1.0, 1.0)
                beta = math.acos(value) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

    def has_box_moved(self):
        current_box_position = [self.box_state.pose.position.x, self.box_state.pose.position.y]
        if self.previous_box_position is None:
            self.previous_box_position = current_box_position
            return False
        box_moved = np.linalg.norm(np.array(current_box_position) - np.array(self.previous_box_position)) > 0.01  # Threshold to consider as movement
        self.previous_box_position = current_box_position
        return box_moved

    def observe_collision(self, laser_data):
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            self.get_logger().info("Box touched!")
            return True, min_laser
        return False, min_laser

    def step(self, action):
        global last_odom, camera_data, velodyne_data
        done = False

        if last_odom is None:
            start_time = time.time()
            timeout = 10
            while last_odom is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
            if last_odom is None:
                return np.zeros(self.environment_dim + self.robot_dim), 0, True, False

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
        box_detected = is_box_detected(velodyne_data)

        collision, min_laser = self.observe_collision(velodyne_data)

        if collision:
            box_x = self.box_state.pose.position.x
            box_y = self.box_state.pose.position.y
            direction_to_goal = np.arctan2(self.goal_y - box_y, self.goal_x - box_x)

            angle_diff = direction_to_goal - angle
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            vel_cmd = Twist()
            vel_cmd.linear.x = 0.7
            vel_cmd.angular.z = 1.0 * -angle_diff
            self.vel_pub.publish(vel_cmd)
        elif box_detected:
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
            vel_cmd.linear.x = 0.7
            vel_cmd.angular.z = 2 * -angle_diff
            self.vel_pub.publish(vel_cmd)
        else:
            self.box_detected_flag = False
            vel_cmd = Twist()
            vel_cmd.linear.x = float(action[0])
            vel_cmd.angular.z = float(action[1]) * 1.5
            self.vel_pub.publish(vel_cmd)

        self.call_service(self.unpause)
        time.sleep(TIME_DELTA)
        self.call_service(self.pause)

        laser_state = velodyne_data[:self.environment_dim]

        box_x = self.box_state.pose.position.x
        box_y = self.box_state.pose.position.y

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

        self.get_logger().info(f"Laser data shape: {laser_state.shape}")
        self.get_logger().info(f"Robot state shape: {np.array(robot_state).shape}")

        if self.camera_initialized:
            image_features = vint_model(camera_data.unsqueeze(0)).squeeze(0)
            self.get_logger().info(f"Image features shape: {image_features.shape}")
            state = torch.cat((torch.tensor(laser_state), image_features, torch.tensor(robot_state).to(device).float())).detach().cpu().numpy()
        else:
            state = np.concatenate((laser_state, robot_state))

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

        # Concatenating arrays
        if self.camera_initialized:
            state = torch.cat((torch.tensor(laser_state), image_features, torch.tensor(robot_state).to(device).float())).detach().cpu().numpy()
        else:
            state = np.concatenate((laser_state, robot_state))

        return state, reward, done, reached_goal

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
        global last_odom, velodyne_data, camera_data
        last_odom = None
        velodyne_data = np.ones(self.environment_dim) * 10
        camera_data = np.ones((3, 224, 224))  # Initialize with correct size
        self.timeout_start_time = time.time()
        self.collision = False
        self.box_detected_flag = False
        self.previous_box_position = None

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/reset_world service not available, waiting again...')
        try:
            self.reset_proxy.call_async(self.req)
            rclpy.spin_once(self)  # Ensure the call is processed
        except rclpy.ServiceException as exc:
            self.get_logger().error(f'Service call failed: {exc}')

        if not self.wait_for_odometry():
            return np.zeros(self.environment_dim + self.robot_dim), 0, True, False

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

        time.sleep(TIME_DELTA + 1.0)  # Increased sleep time to allow the simulation to stabilize

        if last_odom is None:
            self.get_logger().error("Timeout: Odometry data not received after reset.")
            return np.zeros(self.environment_dim + self.robot_dim), 0, True, False

        laser_state = velodyne_data[:self.environment_dim]
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

        self.get_logger().info(f"Laser data shape: {laser_state.shape}")
        self.get_logger().info(f"Robot state shape: {np.array(robot_state).shape}")

        if self.camera_initialized:
            image_features = vint_model(camera_data.unsqueeze(0)).squeeze(0)
            self.get_logger().info(f"Image features shape: {image_features.shape}")
            state = torch.cat((torch.tensor(laser_state), image_features, torch.tensor(robot_state).to(device).float())).detach().cpu().numpy()
        else:
            state = np.concatenate((laser_state, robot_state))

        self.get_logger().info(f"State in reset method: {state}, State shape: {state.shape}")

        return state

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

    def wait_for_odometry(self, timeout=30):
        start_time = time.time()
        while last_odom is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        if last_odom is None:
            self.get_logger().error("Timeout: Odometry data not received after reset.")
            return False
        return True

class OdomSubscriber(Node):
    def __init__(self, topic):
        super().__init__('odom_subscriber_' + topic.replace('/', '_'))
        self.subscription = self.create_subscription(Odometry, topic, self.odom_callback, 10)
        self.get_logger().info(f"Subscribed to {topic}")

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data
        self.get_logger().info(f"Received odometry data on {self.subscription.topic_name}")
        self.get_logger().debug(f"Odometry data: {od_data}")

class VelodyneSubscriber(Node):
    def __init__(self, environment_dim, topic):
        super().__init__('velodyne_subscriber')
        self.environment_dim = environment_dim
        self.subscription = self.create_subscription(PointCloud2, topic, self.velodyne_callback, 10)

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim])
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                value = dot / mag1
                value = np.clip(value, -1.0, 1.0)
                beta = math.acos(value) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

if __name__ == '__main__':
    rclpy.init(args=None)

    # Initialize device and configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    max_ep = 500
    torch.manual_seed(seed)
    np.random.seed(seed)

    # File paths and dimensions
    file_name_lidar = "td3_velodyne"
    file_name_camera = "td3_camera"
    model_dir = "/home/tyler/ros2_ws/src/deep-rl-navigation/DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models"
    environment_dim_lidar = 75
    environment_dim_camera = 999
    robot_dim_lidar = 7
    robot_dim_camera = 8

    state_dim_lidar = environment_dim_lidar + robot_dim_lidar
    state_dim_camera = environment_dim_camera + robot_dim_camera
    action_dim = 2
    max_action = 1

    # Load networks
    network_lidar = TD3(state_dim_lidar, action_dim, max_action)
    network_camera = TD3(state_dim_camera, action_dim, max_action)
    network_lidar.load(file_name_lidar, model_dir)
    network_camera.load(file_name_camera, model_dir)

    # Environment setup
    envs = [
        GazeboEnv({
            'model_name': "robot_lidar",
            'odometry_topic': "/robot_lidar/odom",
            'vel_topic': "/cmd_vel_lidar",
            'pointcloud_topic': "/velodyne_points",
            'camera_topic': None,
            'environment_dim': environment_dim_lidar,
            'robot_dim': robot_dim_lidar,
            'goal_x': 3.5,  # Set goal x coordinate to 3.5
            'goal_y': 3.5   # Set goal y coordinate to 3.5
        }),
        GazeboEnv({
            'model_name': "robot_camera",
            'odometry_topic': "/robot_camera/odom",
            'vel_topic': "/cmd_vel_camera",
            'pointcloud_topic': None,
            'camera_topic': "/camera/image_raw",
            'environment_dim': environment_dim_camera,
            'robot_dim': robot_dim_camera,
            'goal_x': 3.5,  # Set goal x coordinate to 3.5
            'goal_y': 3.5   # Set goal y coordinate to 3.5
        })
    ]

    odom_subscribers = [
        OdomSubscriber("/robot_lidar/odom"),
        OdomSubscriber("/robot_camera/odom")
    ]

    velodyne_subscribers = [
        VelodyneSubscriber(environment_dim_lidar, "/velodyne_points")
    ]

    # Executor setup
    executor = rclpy.executors.MultiThreadedExecutor()
    for env in envs:
        executor.add_node(env)
    for subscriber in odom_subscribers + velodyne_subscribers:
        executor.add_node(subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        # Main loop
        done = [True, True]
        episode_timesteps = [0, 0]

        while rclpy.ok():
            for i, env in enumerate(envs):
                if done[i]:
                    state = env.reset()
                    done[i] = False
                    episode_timesteps[i] = 0
                else:
                    network = network_lidar if i == 0 else network_camera
                    action = network.get_action(np.array(state))
                    # Ensure action is in the correct format
                    action = [(action[0] + 1) / 2, action[1]]
                    next_state, reward, done[i], target = env.step(action)
                    done[i] = 1 if episode_timesteps[i] + 1 == max_ep else int(done[i])
                    state = next_state
                    episode_timesteps[i] += 1

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        rclpy.shutdown()
        executor_thread.join()
