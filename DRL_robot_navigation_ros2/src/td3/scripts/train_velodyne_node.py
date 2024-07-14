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
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
import point_cloud2 as pc2

# Parameters
GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.47
TIME_DELTA = 0.1
MAX_DISTANCE_FROM_BOX = 4.0  # Maximum allowed distance from the box

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_odom = None
environment_dim = 75
velodyne_data = np.ones(environment_dim) * 10

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
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

class TD3(object):  # Corrected class name to match instantiation
    def __init__(self, state_dim, action_dim, max_action, logger, log_dir):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter(log_dir=log_dir)
        self.iter_count = 0
        self.logger = logger  # Add logger to the class

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=0.99,  # Corrected discount to 0.99
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = float('-inf')  # Corrected inf to float('-inf')
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q).item())  # Added .item() to get scalar value
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss.item()  # Added .item() to get scalar value
        self.iter_count += 1
        # Write new values for tensorboard
        self.logger.info(f"Reward/Penalty: loss={av_loss / iterations}, Av.Q={av_Q / iterations}, Max.Q={max_Q}, Iterations={self.iter_count}")  # Updated logger
        self.writer.add_scalar("lidar_loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("lidar_AvQ", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("lidar_MaxQ", max_Q, self.iter_count)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )

def is_box_detected(velodyne_data, detection_range=3.0):
    detection_threshold = sum(velodyne_data < detection_range)
    detected = detection_threshold > (0.05 * len(velodyne_data))  # Adjust the threshold as needed
    return detected

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
        self.timeout_duration = 120

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

        # Add logger for environment dimension
        self.get_logger().info(f"Environment dimension: {self.environment_dim}")

        # Add logger for state dimension
        state_dim = self.environment_dim + 7  # 7 is the robot_dim
        self.get_logger().info(f"State dimension: {state_dim}")

    def model_states_callback(self, msg):
        try:
            if "target_box" in msg.name:
                index = msg.name.index("target_box")
                self.box_state.pose = msg.pose[index]
            else:
                self.get_logger().warning("Box model not found in the model states.")
        except Exception as e:
            self.get_logger().error(f"Error in model_states_callback: {e}")

    def has_box_moved(self):
        current_box_position = [self.box_state.pose.position.x, self.box_state.pose.position.y]
        if self.previous_box_position is None:
            self.previous_box_position = current_box_position
            return False
        box_moved = np.linalg.norm(np.array(current_box_position) - np.array(self.previous_box_position)) > 0.01  # Threshold to consider as movement
        self.previous_box_position = current_box_position
        return box_moved

    def observe_collision(self, laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            self.get_logger().info("Box touched!")
            return True, min_laser
        return False, min_laser


    def step(self, action):
        global last_odom, velodyne_data
        done = False

        if last_odom is None:
            start_time = time.time()
            timeout = 10
            while last_odom is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
            if last_odom is None:
                return np.zeros(self.environment_dim + 7), 0, True, False  # robot_dim is 7

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
            vel_cmd.linear.x = 0.7  # Move forward with constant speed
            vel_cmd.angular.z = 1.0 * -angle_diff  # Tighter turn towards the goal
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
            vel_cmd.linear.x = 0.7  # Move forward with constant speed
            vel_cmd.angular.z = 2 * -angle_diff  # Tighter turn towards the box
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
        global last_odom, velodyne_data
        last_odom = None
        velodyne_data = np.ones(self.environment_dim) * 10  # Initialize with correct size
        self.timeout_start_time = time.time()  # Reset the timeout timer at reset
        self.collision = False  # Reset the collision flag

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/reset_world service not available, waiting again...')
        try:
            self.reset_proxy.call_async(self.req)
        except rclpy.ServiceException as exc:
            self.get_logger().error(f'Service call failed: {exc}')

        # Ensure the initial positions and orientations are set according to the world file
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
        except rclpy.ServiceException as exc:
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

        state = np.concatenate((laser_state, robot_state))

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
        # Publish visual data in Rviz
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
    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class VelodyneSubscriber(Node):
    def __init__(self, environment_dim):
        super().__init__('velodyne_subscriber')
        self.environment_dim = environment_dim
        self.subscription = self.create_subscription(PointCloud2, "/velodyne_points", self.velodyne_callback, 10)

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

def evaluate(env, network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    goal_reached = 0
    successful_pushes = 0

    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, reached_goal = env.step(a_in)
            episode_reward += reward
            avg_reward += reward
            if reward < -90:
                col += 1
            if reached_goal:
                goal_reached += 1
            if env.has_box_moved():  # Call the method instead of accessing a non-existing attribute
                successful_pushes += 1

    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_successful_pushes = successful_pushes / eval_episodes

    env.get_logger().info("..............................................")
    env.get_logger().info(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward %f, avg_col %f, avg_successful_pushes %f, goal_reached %i"
        % (eval_episodes, epoch, avg_reward, avg_col, avg_successful_pushes, goal_reached)
    )
    env.get_logger().info("..............................................")
    return avg_reward

def main(args=None):
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
    file_name = "td3_velodyne"
    save_model = True
    load_model = False
    random_near_obstacle = True

    result_dir = os.path.expanduser("~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/results")
    model_dir = os.path.expanduser("~/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    environment_dim = 75
    robot_dim = 7

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    env = GazeboEnv(environment_dim)
    log_dir = "/home/tyler/ros2_ws/src/Heterogenous-Robotics/DRL_robot_navigation_ros2/src/td3/scripts/runs/velodyne"
    network = TD3(state_dim, action_dim, max_action, env.get_logger(), log_dir)  # Pass the logger and log_dir
    replay_buffer = ReplayBuffer(buffer_size, seed)
    if load_model:
        try:
            network.load(file_name, model_dir)
        except Exception as e:
            print(f"Could not load the stored model parameters, initializing training with random parameters: {e}")

    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    done = True
    epoch = 1

    count_rand_actions = 0
    random_action = []

    odom_subscriber = OdomSubscriber()
    velodyne_subscriber = VelodyneSubscriber(environment_dim)

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)
    executor.add_node(env)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = env.create_rate(10)

    try:
        while rclpy.ok() and timestep < max_timesteps:
            if done:
                if timestep != 0:
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
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate(env, network=network, epoch=epoch, eval_episodes=eval_ep))

                    if save_model:
                        network.save(file_name, model_dir)
                        np.save(os.path.join(result_dir, file_name), evaluations)
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
                expl_noise -= ((expl_noise - expl_min) / expl_decay_steps)

            action = network.get_action(np.array(state))
            action = np.clip(action + np.random.normal(0, expl_noise, size=action_dim), -max_action, max_action)

            # Remove the condition for random actions near obstacles
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
