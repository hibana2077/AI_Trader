'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-10-28 15:02:47
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-11-16 23:15:49
FilePath: \AI_Trader\main\model\model1.py
Description: This is a model(MLP) for AI Trader
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BLOCK(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(BLOCK, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

class DQN(nn.Module):
    """
    Deep Q-Network model with multiple blocks.
    
    Args:
    - n_observations (int): number of observations in the input.
    - n_actions (int): number of actions in the output.
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.block1 = BLOCK(n_observations, n_actions)
        self.block2 = BLOCK(n_observations, n_actions)
        self.block3 = BLOCK(n_observations, n_actions)
        self.block4 = BLOCK(n_observations, n_actions)
        self.fusion = nn.Linear(4*n_actions, n_actions)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x = torch.cat((x1,x2,x3,x4), dim=1)
        return self.fusion(x)