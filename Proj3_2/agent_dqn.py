#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import amp  # Updated import for AMP

from agent import Agent
from dqn_model import DQN
from plot import Plot

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

# class Replay_Buffer():
#     def __init__(self, capacity, position, state_shape, device):

#         self.capacity = capacity
#         self.position = position
#         self.state_space = state_shape

#         self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
#         self.actions = np.zeros(capacity, dtype=np.int64)
#         self.rewards = np.zeros(capacity, dtype=np.float32)
#         self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
#         self.dones = np.zeros(capacity, dtype=np.bool_)
#         self.priorities = np.ones(capacity, dtype=np.float32)

#         self.device = device

    
#     def push(self, state, action, reward, next_state, done):
#         self.states[self.position] = state
#         self.actions[self.position] = state
#         self.rewards[self.position] = state
#         self.next_states[self.position] = state
#         self.dones[self.position] = state


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args

        self.dqn = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.loss = torch.nn.HuberLoss()
        self.optimizer = torch.optim.AdamW(self.dqn.parameters(), lr=args.learning_rate)

        self.buffer = deque(maxlen=self.args.buffer_len)
        self.steps = 0
        
        # args
        self.num_episodes = args.num_episodes
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.gamma = args.gamma
        self.training_start = args.training_start
        self.update_freq = args.update_freq
        self.decay_rate = args.decay_rate
        self.batch_size = args.batch_size

        self.rewards = []
        self.losses = []

        if args.test_dqn:
            logger.info('Loading trained model')
            if args.filename:
                self.dqn.load_model(args.filename)
            else:
                self.dqn.load_model()

        # Enable CUDNN Benchmarking for optimized convolution operations
        torch.backends.cudnn.benchmark = True
            
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        pass
     
    def make_action(self, observation, num_actions=4, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        # print("Obs tensor: ", obs_tensor)
        
        with amp.autocast(device_type=self.device.type):
            q_vals = self.dqn(obs_tensor)
            # print("q_vals: ", q_vals)
        
        # print("epsilon: ", self.epsilon)
        if not test and random.random() < self.epsilon:
            # print("Action space: ", self.env.action_space.n)
            action = self.env.action_space.sample()
            # print("Random action from make_action: ", action)
        else:
            action = torch.argmax(q_vals, dim=1).item()
            # print("Max action from make_action: ", action)

        return action
    
    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        self.buffer.append((state, action, reward, next_state, done)) 
           
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        batch = random.sample(self.buffer, k=self.batch_size)
        return batch 

    def update(self):
        if len(self.buffer) < self.training_start: # start training after filling buffer
            return 
        experiences = self.replay_buffer()
        states, actions, rewards, next_states, dones = zip(*experiences)
        # print("Next states: ", next_states)
        # print("Next states shape: ", np.array(next_states).shape)

        states = torch.tensor(np.array(states), dtype=torch.float32).permute(0,3,1,2).to(self.device, non_blocking=True)
        # print("States: ", states)
        # print("States shape: ", states.shape)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device, non_blocking=True)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device, non_blocking=True)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).permute(0,3,1,2).to(self.device, non_blocking=True)
        dones = torch.tensor(np.array(dones), dtype=torch.int64).to(self.device, non_blocking=True)

        qs = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_qs = self.target(next_states).max(dim=1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_qs

        loss = self.loss(qs, targets.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def train(self):
        """
        Implement your training algorithm here
        """
        for episode in tqdm(range(self.args.num_episodes)):
            state = self.env.reset()
            # print("initial state shape: ", state.shape)
            done = False

            total_reward = 0
            episode_loss = 0
            steps = 0

            while not done:
                action = self.make_action(state, test=False)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.push(state, action, reward, next_state, done) # push to replay buffer

                total_reward += reward 
                self.steps += 1 # global steps
                steps += 1 # episode steps

                loss = self.update()
                if loss is not None:
                    episode_loss += loss.item()

                if self.steps > 10000 and self.epsilon > self.epsilon_min:
                    self.epsilon -= self.decay_rate
                    # if self.steps % 10 == 0:
                    #     print("Epsilon: ", self.epsilon)
                
                if self.steps % 5000 == 0:
                    self.target.load_state_dict(self.dqn.state_dict())

                state = next_state
            
            self.rewards.append(total_reward)
            self.losses.append(episode_loss)
            
            # plotting, logging, saving
            if episode and episode % self.args.write_freq == 0:
                # rewards plot
                plot = Plot(self.rewards)
                plot.plot("Rewards", "Rewards", "training_rewards", self.args.filename)

                # loss plot
                plot = Plot(self.losses[10:])
                plot.plot("Loss", "Loss", "training_loss", self.args.filename)

                logger.info(f"Episode {episode+1}: Loss = {episode_loss}")
                avg_reward = sum(self.rewards[-100:]) / 100 if len(self.rewards) >= 100 else sum(self.rewards) / len(self.rewards)
                logger.info(f"Episode {episode+1}: Avg Reward Last 100 = {avg_reward}")
                logger.info(f"Episode {episode+1}: Epsilon = {self.epsilon}")
                logger.info(f"Episode {episode+1}: Steps this episode = {steps}")

                self.save_model(self.args.filename)
    
    def save_model(self, weights_filename):
        try:
            torch.save(self.dqn.state_dict(), f"models/{weights_filename}.pt")
            print(f"Model saved at models/{weights_filename}.pt")
        except:
            print(f"Model {weights_filename} could not be saved")

    def load_model(self, weights_filename):
        try:
            self.dqn.load_state_dict(torch.load(f"models/{weights_filename}.pt"))
            print(f"Successfully loaded weights file models/{weights_filename}.pt.")
        except:
            print(f"No weights file available at models/{weights_filename}.pt")

