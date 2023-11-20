'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-10-28 15:02:47
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-11-21 00:01:35
FilePath: \AI_Trader\main\model\model1.py
Description: This is a template model for AI Trader, you can copy this file and rename it to your own model.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) class.

    Args:
    - n_observations (int): number of observations in the input.
    - n_actions (int): number of actions in the output.
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.softmax(self.layer3(x), dim=1)