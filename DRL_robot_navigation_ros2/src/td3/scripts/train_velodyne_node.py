#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer

import rclpy
from rclpy.node import Node
import threading
import math
import random
from gazebo_msgs.msg import ModelState, ContactsState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
import point_cloud2 as pc2
from sklearn.cluster import DBSCAN

# Parameters
GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_odom = None
environment_dim = 75
velodyne_data = np.ones(environment_dim) * 10

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

        # Ensure symmetric initialization
        nn.init.uniform_(self.layer_1.weight, -0.003, 0.003)
        nn.init.uniform_(self.layer_2.weight, -0.003, 0.003)
        nn.init.uniform_(self.layer_3.weight, -0.003, 0.003)

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)
        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        s11 = F.relu(self.layer_2_s(s1))
        s12 = F.relu(self.layer_2_a(a))
        s1 = F.relu(s11 + s12)
        q1 = self.layer_3(s1)
        s2 = F.relu(self.layer_4(s))
        s21 = F.relu(self.layer_5_s(s2))
        s22 = F.relu(self.layer_5_a(a))
        s2 = F.relu(s21 + s22)
        q2 = self.layer_6(s2)
        return q1, q2

# TD3 Algorithm
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, expl_noise=0.1):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter(log_dir="./DRL_robot_navigation_ros2/src/td3/scripts/runs")
        self.iter_count = 0
        self.expl_noise = expl_noise  # Initialize exploration noise

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action = action + np.random.normal(0, self.expl_noise, size=action.shape)  # Symmetric action noise
        action = np.clip(action, -self.max_action, self.max_action)
        return action
    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            next_action = self.actor_target(next_state)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            current_Q1, current_Q2 = self.critic(state, action)
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                actor_loss = -self.critic(state, self.actor(state))[0].mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.iter_count += 1
            self.writer.add_scalar("loss", loss.item(), self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename)))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename)))

def is_box_detected(velodyne_data):
    # Adjust these thresholds as needed
    box_distance_min = 0.1
    box_distance_max = 3.0

    try:
        box_points = [
            distance for distance in velodyne_data
            if box_distance_min < distance < box_distance_max
        ]
    except Exception as e:
        print(f"Error in box detection loop: {e}")
        print(f"velodyne_data: {velodyne_data}")
        return False

    return len(box_points) > 3  # Arbitrary threshold to determine if the box is detected

def adjust_action_towards_box(robot_x, robot_y, robot_theta, box_position):
    # Calculate the direction towards the box and adjust the action
    box_x, box_y = box_position
    delta_x = box_x - robot_x
    delta_y = box_y - robot_y
    angle_to_box = np.arctan2(delta_y, delta_x)

    # Calculate the angle difference
    angle_diff = angle_to_box - robot_theta
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

    # Define linear and angular velocity adjustments
    linear_velocity = 0.5  # Move forward
    angular_velocity = angle_diff  # Turn towards the box

    return [linear_velocity, angular_velocity]  # Return as a list

# GazeboNode 
class GazeboEnv(Node):
    def __init__(self, environment_dim):
        super().__init__('env')
        self.environment_dim = environment_dim  # Use environment_dim parameter
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 3.5  # Instance variable
        self.goal_y = 3.5  # Instance variable

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
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

        self.box_contact = False  # Initialize box contact status
        self.box_contact_subscriber = self.create_subscription(
            ContactsState,
            "/gazebo/default/physics/contacts",
            self.box_contact_callback,
            10
        )

        self.start_time = None  # To track the start time of each episode
        self.last_box_detection_time = 0  # Track the last time the box was detected
        self.box_detection_cooldown = 5  # Cooldown period in seconds
        self.timeout_occurred = False  # Track if timeout occurred in the current episode

    def box_contact_callback(self, msg):
        self.box_contact = any(
            "target_box::link::target_box_collision" in state.collision1_name or
            "target_box::link::target_box_collision" in state.collision2_name
            for state in msg.states
        )
        if self.box_contact:
            self.get_logger().info("Box contact detected")

    def step(self, action):
        global last_odom, velodyne_data
        done = False  # Initialize done variable at the beginning

        if last_odom is None:
            self.get_logger().warning("Odometry data not received yet, waiting...")
            start_time = time.time()
            timeout = 10
            while last_odom is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
        if last_odom is None:
            self.get_logger().error("Timeout: Odometry data not received.")
            return np.zeros(self.environment_dim + 4), 0, True, False

        # Check if the box is detected
        try:
            box_detected = is_box_detected(velodyne_data)
        except Exception as e:
            self.get_logger().error(f"Error during box detection: {e}")
            box_detected = False

        # Update odometry data
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

        # Check if the box is detected and adjust the action if needed
        if box_detected:
            box_position = [self.box_state.pose.position.x, self.box_state.pose.position.y]
            action = adjust_action_towards_box(self.odom_x, self.odom_y, angle, box_position)

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])  # Ensure symmetric scaling
        vel_cmd.angular.z = float(action[1])  # Ensure symmetric scaling
        self.vel_pub.publish(vel_cmd)

        self.call_service(self.unpause)
        time.sleep(TIME_DELTA)
        self.call_service(self.pause)

        laser_state = [velodyne_data[:]]

        # Calculate distance and angle to the goal
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        beta = math.acos(dot / mag1)
        if skew_y < 0:
            beta = -beta if skew_x >= 0 else 0 - beta
        theta = beta - angle
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Check if the box is at the target position
        box_position = [self.box_state.pose.position.x, self.box_state.pose.position.y]
        box_target_distance = np.linalg.norm(np.array(box_position) - np.array(self.target_position))
        reached_goal = box_target_distance < GOAL_REACHED_DIST

        if reached_goal:
            self.get_logger().info("Box reached the goal!")
            done = True
            target = True
        else:
            target = False

        # Timeout handling
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        elapsed_time = current_time - self.start_time
        if elapsed_time > 360 and not self.timeout_occurred:
            self.get_logger().info("Timeout: Robot did not find the box within 5 minutes.")
            done = True
            timeout = True
            self.timeout_occurred = True  # Mark the timeout as occurred
        else:
            timeout = False

        # Construct the state
        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)

        min_laser = min(velodyne_data)  # Just calculate min_laser for potential use in rewards
        reward = self.get_reward(target, action, min_laser, self.box_contact, timeout, box_detected, current_time)

        # Reset box contact status
        self.box_contact = False

        return state, reward, done, target
    
    def get_reward(self, target, action, min_laser, box_contact, timeout, box_detected, current_time):
        reward = 0.0

        if target:
            self.get_logger().info("Target reached +1000 points!")
            reward = 1000.0
        elif timeout:
            self.get_logger().info("Timeout occurred -500 points")
            reward = -500.0  # Negative reward for timeout
        elif box_contact:
            self.get_logger().info("Box contact +500 points")
            reward = 500.0  # Additional reward for touching the box
        elif box_detected and (current_time - self.last_box_detection_time) > self.box_detection_cooldown:
            self.get_logger().info("Box detected +10 points")
            self.last_box_detection_time = current_time  # Update the last detection time
            reward = 10.0  # Reward for detecting the box
        else:
            # Reward for moving towards the box
            box_distance = np.linalg.norm([self.box_state.pose.position.x - self.odom_x, self.box_state.pose.position.y - self.odom_y])
            reward += 1.0 - box_distance

            # Reward for moving towards the goal
            distance_to_goal = np.linalg.norm([self.goal_x - self.box_state.pose.position.x, self.goal_y - self.box_state.pose.position.y])
            reward += 1.0 - distance_to_goal

            # Reward for moving forward and slight penalty for turning
            forward_reward = 1.0 - abs(action[0])  # Reward for moving forward
            turn_penalty = 0.5 * abs(action[1])  # Penalize excessive turning

            # Proximity penalty for being too close to obstacles
            proximity_penalty = (1 - min_laser if min_laser < 1 else 0.0) / 2

            # Combine rewards and penalties
            reward += forward_reward - turn_penalty - proximity_penalty

        return reward

    def reset(self):
        global last_odom, velodyne_data
        last_odom = None
        self.get_logger().info("Resetting environment...")

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/reset_world service not available, waiting again...')
        try:
            self.reset_proxy.call_async(self.req)
        except rclpy.ServiceException as exc:
            self.get_logger().error(f'Service call failed: {exc}')

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        self.set_self_state.pose.position.x = np.random.uniform(-4.5, 4.5)
        self.set_self_state.pose.position.y = np.random.uniform(-4.5, 4.5)
        self.set_self_state.pose.orientation.z = quaternion.z
        self.set_self_state.pose.orientation.w = quaternion.w
        self.set_state.publish(self.set_self_state)

        self.goal_x = 3.5
        self.goal_y = 3.5

        self.box_state.pose.position.x = np.random.uniform(-4.5, 4.5)
        self.box_state.pose.position.y = np.random.uniform(-4.5, 4.5)
        self.set_state.publish(self.box_state)

        self.publish_markers([0.0, 0.0])

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(self.req)
        except rclpy.ServiceException as exc:
            self.get_logger().error(f'Service call failed: {exc}')

        time.sleep(TIME_DELTA)
        while last_odom is None:
            self.get_logger().info("Waiting for odometry data...")
            rclpy.spin_once(self, timeout_sec=0.1)

        laser_state = [velodyne_data[:]]
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        beta = math.acos(dot / mag1)
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)

        # Reset the start time
        self.start_time = None
        self.last_box_detection_time = 0  # Reset the box detection time
        self.timeout_occurred = False  # Reset the timeout occurrence

        return state

    def call_service(self, service):
        while not service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            future = service.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future)
        except rclpy.ServiceException as e:
            self.get_logger().error(f"Service call failed: {e}")

    def publish_markers(self, action):
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

    # @staticmethod
    # def observe_collision(laser_data, box_detected, box_contact):
    #     min_laser = min(laser_data)
    #     collision = min_laser < COLLISION_DIST and not box_detected and not box_contact
    #     return collision, min_laser

class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class VelodyneSubscriber(Node):
    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.subscription = self.create_subscription(PointCloud2, "/velodyne_points", self.velodyne_callback, 10)
        self.subscription

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim])
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                beta = math.acos(dot / mag1) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break
        # Debugging: print out the structure of velodyne_data
        # print(f"Processed velodyne_data: {velodyne_data}")

def evaluate(env, network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    goal_reached = 0
    for _ in range(eval_episodes):
        env.get_logger().info(f"Evaluating episode {_}")
        state = env.reset()
        done = False
        while not done:
            action = network.get_action(np.array(state))
            # env.get_logger().info(f"Action: {action}")
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, reached_goal = env.step(a_in)
            avg_reward += reward
            if reward < -90:
                col += 1
            if reached_goal:
                goal_reached += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    env.get_logger().info("..............................................")
    env.get_logger().info(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward %f, avg_col %f, goal_reached %i"
        % (eval_episodes, epoch, avg_reward, avg_col, goal_reached)
    )
    env.get_logger().info("..............................................")
    return avg_reward

def main(args=None):
    rclpy.init(args=args)

    seed = 0
    eval_freq = 5e3
    max_ep = 500
    eval_ep = 10
    max_timesteps = 5e6
    expl_noise = 1  # Initial exploration noise
    expl_decay_steps = 500000
    expl_min = 0.1
    batch_size = 40
    discount = 0.99999
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    buffer_size = 1e6
    file_name = "td3_velodyne"
    save_model = True
    load_model = False
    random_near_obstacle = True

    result_dir = os.path.expanduser("~/ros2_ws/src/deep-rl-navigation/DRL_robot_navigation_ros2/src/td3/scripts/results")
    model_dir = os.path.expanduser("~/ros2_ws/src/deep-rl-navigation/DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models")
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    environment_dim = 25
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim * 3 + robot_dim  # Correctly calculate state_dim
    action_dim = 2
    max_action = 1

    network = TD3(state_dim, action_dim, max_action, expl_noise)
    replay_buffer = ReplayBuffer(buffer_size, seed)
    if load_model:
        try:
            network.load(file_name, model_dir)
        except Exception as e:
            print(f"Could not load the stored model parameters, initializing training with random parameters: {e}")

    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1

    count_rand_actions = 0
    random_action = []

    env = GazeboEnv(environment_dim)
    odom_subscriber = OdomSubscriber()
    velodyne_subscriber = VelodyneSubscriber()
    
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = env.create_rate(10)

    try:
        while rclpy.ok() and timestep < max_timesteps:
            if done:
                env.get_logger().info(f"Done. timestep : {timestep}")
                if timestep != 0:
                    env.get_logger().info("Training")
                    network.train(
                        replay_buffer,
                        episode_timesteps,
                        batch_size,
                        discount,
                        tau,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                    )

                if timesteps_since_eval >= eval_freq:
                    env.get_logger().info("Validating")
                    timesteps_since_eval %= eval_freq
                    evaluations.append(
                        evaluate(env, network=network, epoch=epoch, eval_episodes=eval_ep)
                    )

                    if save_model:
                        network.save(file_name, model_dir)
                        np.save(os.path.join(result_dir, file_name), evaluations)
                    epoch += 1

                try:
                    state = env.reset()
                    done = False
                    episode_reward = 0
                    episode_timesteps = 0
                except Exception as e:
                    env.get_logger().error(f"Error during environment reset: {e}")
                    continue

            if expl_noise > expl_min:
                expl_noise -= ((1 - expl_min) / expl_decay_steps) * 0.1  # Adjust the decrement factor as needed
                network.expl_noise = expl_noise  # Update exploration noise

            action = network.get_action(np.array(state))
            action = np.clip(action + np.random.normal(0, expl_noise, size=action_dim), -max_action, max_action)

            if random_near_obstacle:
                if (
                    np.random.uniform(0, 1) > 0.85
                    and min(state[4:-8]) < 0.6
                    and count_rand_actions < 1
                ):
                    count_rand_actions = np.random.randint(8, 15)
                    random_action = np.random.uniform(-1, 1, 2)

                if count_rand_actions > 0:
                    count_rand_actions -= 1
                    action = random_action
                    action[0] = -1

            a_in = [(action[0] + 1) / 2, action[1]]
            try:
                next_state, reward, done, target = env.step(a_in)
                episode_reward += reward

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
        rclpy.shutdown()
        executor_thread.join()

if __name__ == '__main__':
    main()
