#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from tqdm import tqdm
# from accelerate import Accelerator
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


from agent import Agent
from dqn_model import DQN
from environment import Environment
from argument import add_arguments
from plot import Plot

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


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

        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
        
        # hyperparameters
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.025
        self.epsilon_decay = 0.99
        self.update_freq = 5000
        self.D = deque(maxlen=10000)
        self.frame_stack = deque(maxlen=4) # is this needed
        self.frame_stack.clear()

        self.dqn = DQN()
        self.dqn.to(self.device)  # Move the model to the device

        self.target = DQN() # initializing Q values for main and target network
        self.target.to(self.device)

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=args.learning_rate) # setting learning rate to learning rate in args

        # self.dqn, self.target, self.optimizer = accelerator.prepare(self.dqn, self.target, self.optimizer)
        
        self.state = self.init_game_setting    
            
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.dqn.load_model()
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        return self.env.reset()
        # pass
    
    
    def make_action(self, observation, num_actions=4, test=True): # test=True for epsilon greedy
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        # print("State shape before DQN:", observation.shape)  # Should be [batch_size, channels, height, width]

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        q_vals = self.dqn(obs_tensor)
        # print("Q vals: ", q_vals)

        if test and random.random() < self.epsilon: # explore
            action = random.randint(0, num_actions-1)
            # print("Random action: ", action)
        else:
            action = torch.argmax(q_vals, dim=1).item()
            # print("Greedy action: ", action)
        return action
    
    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        
        self.D.append((state, action, reward, next_state, done))        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """

        return random.sample(self.D, self.batch_size) 
        

    def train(self, n_episodes=100000):
        """
        Implement your training algorithm here
        """
        rewards = []
        avg_rewards = []
        for episode in tqdm(range(n_episodes), desc='Training'):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.make_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward

                self.push(state, action, reward, next_state, done)

            if len(self.D) >= self.batch_size:  # Train only if there's enough data for a batch
                self.update()

            # print(f"Episode {episode + 1}/{n_episodes} finished with reward: {episode_reward}")
            # Add current episode's reward to list of rewards
            rewards.append(episode_reward)

            # Calculate average reward over last 100 episodes
            
            if len(rewards) >= 100:
                avg_reward = sum(rewards[-100:]) / 100
                avg_rewards.append(avg_reward)
                # print(f"Average reward over last 100 episodes: {avg_reward}")
            else:
                avg_rewards.append(0)
        
            if episode % 1000 == 0:
                while self.epsilon >= 0.1:
                    self.epsilon = self.epsilon * 0.9
        # self.eps_start = 1.0
        # self.eps_final = 0.01
        # self.eps_decay = 2000000
        # self.eps = 1.0

        # if self.global_step > 100000:
        #     self.eps -= (self.eps_start - self.eps_final) / self.eps_decay
        #     self.eps = max(self.eps, self.eps_final)
            # Move to the next state
            state = next_state
            
        plot = Plot(avg_rewards) # plot training
        plot.plot()

        print("Training complete. Saving model...")
        self.dqn.save_model('models/latest.pt')
        # if args.test_dqn:
            #you can load your model here
            # print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            ###########################
            # self.q_net.load_state_dict(torch.load('./dqn_breakout.pth'))
    def update(self):  
        batch = self.replay_buffer()
        states, actions, rewards, next_states, dones = zip(*batch)
        
        state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device) # combining into np.arrays for speed
        # print("State tensor shape: ", state_tensor.shape)
        action_tensor = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        reward_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        # print("next state tensor shape: ", next_state_tensor.shape)
        done_tensor = torch.tensor(np.array(dones), dtype=torch.bool).to(self.device)

        q = self.dqn(state_tensor)
        # print("q: ", q)

        q_for_actions = torch.gather(q, 1, action_tensor.unsqueeze(1)).squeeze(1)
        # print("q for actions: ", q_for_actions)

        with torch.no_grad():  # Don't compute gradients for the target Q-values
            next_state_values = self.target(next_state_tensor)  # Shape: (batch_size, num_actions)
            # print("next state q values: ", next_state_values)
            max_next_q_values, _ = torch.max(next_state_values, dim=1)
            # print("max next q values: ", max_next_q_values)
        target_q_values = reward_tensor + (1 - done_tensor.float()) * self.gamma * max_next_q_values
        # print("target q values: ", target_q_values)
        # Step 6: Calculate the loss
        loss_fn = torch.nn.HuberLoss()
        loss = loss_fn(q_for_actions, target_q_values)

        # Step 7: Backpropagation
        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()             # Compute the gradients
        self.optimizer.step()        # Update the parameters

        return loss.item()  # You can return the loss for monitoring