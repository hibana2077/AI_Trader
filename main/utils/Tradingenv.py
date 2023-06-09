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
        
    def _caculate_profit(self,closeprice:int):
        """計算收益"""
        profit = self.position * (closeprice - self.entry_price)
        #if -(short) -> -15 * (90 - 120) = 450
        #if +(long) -> 15 * (120 - 90) = 450
        #既可以計算多單也可以計算空單，還不用加abs
        #totalfee
        totalfee = self.fee * (self.position * (self.entry_price + closeprice))

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0

        return observation,info
    
    def step(self,action):
        return observation,reward,done,info
    
    def render(self,mode='human'):
        """Using plotly to render the chart"""
        pass
