#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(8,8), stride=(4,4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2)

        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, num_actions)

        # for dueling DQN
        # self.state_value1 = nn.Linear(3136, 1024)
        # self.state_value2 = nn.Linear(1024, 1024)
        # self.state_value3 = nn.Linear(1024, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # x = torch.Tensor(x)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.flatten(x)

        action_value = self.relu(self.action_value1(x))
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.action_value3(action_value)

        # do the same with state_values for Dueling DQN

        return action_value 
        # dueling: state_value + (action_value - action_value.mean())

    def save_model(self, weights_filename='models/latest.pt'):
        torch.save(self.state_dict(), weights_filename)

    def load_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}.")
        except:
            print(f"No weights file available at {weights_filename}")
