'''
Author: hibana2077 hibana2077@gmaill.com
Date: 2023-05-31 11:14:02
LastEditors: hibana2077 hibana2077@gmaill.com
LastEditTime: 2023-05-31 11:38:05
FilePath: /AI_Trader/main/utils/Tradingenv.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import gym
import numpy as np
import pandas as pd
import arrow
import random
import sys
from plotly.graph_objects import Figure, Scatter, Candlestick
from gym.envs.registration import register
from gym import error, spaces, utils


"""
init
    - data: pd.DataFrame
    - fee: float
    - slippage: float
    - initial_balance: float
    -
reset
step
render
"""

"""
position:
    + -> long
    - -> short
    0 -> no position
"""

class Tradingenv(gym.Env):
    def __init__(self,data:pd.DataFrame,fee:float,initial_balance:float,slippage:float) -> None:
        
        self.init_date = data.index[0]

        self.data = data
        self.fee = fee
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.slippage = slippage
        self.position = 0.0
        self.entry_price = 0.0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0,high=1,shape=(2,))
        self.reset()
        
    def _caculate_profit(self):
        """計算收益"""
        if self.position > 0:
            profit = (self.data.Close-self.entry_price)/self.entry_price
        elif self.position < 0:
            profit = (self.entry_price-self.data.Close)/self.entry_price
        else:
            profit = 0

        #total

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0

        return observation,info
    
    def step(self,action):
        return observation,reward,done,info
    
    def render(self,mode='human'):
        """Using plotly to render the chart"""
        pass
