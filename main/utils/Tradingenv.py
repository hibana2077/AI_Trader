
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
    - position: float
    - entry_price: float
reset
    return observation,info
        observation: np.array([balance,position])
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
    def __init__(self,data:pd.DataFrame,fee:float,initial_balance:float,slippage:float,mode:str) -> None:
        
        self.init_date = data.index[0]

        self.data = data
        self.fee = fee
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.slippage = slippage
        self.position = 0.0
        self.entry_price = 0.0
        self.idx = 0
        self.mode = mode #"Orderbook" or "Candlestick"

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
        #totalfee = 0.0004 * (15 * (90 + 120)) = 0.0004 * 15 * 210 = 1.26
        #totalfee = 0.0004 * (15 * (120 + 90)) = 0.0004 * 15 * 210 = 1.26
        return profit - totalfee

    def reset(self):
        '''
        Reset the state of the environment and returns an initial observation.
        Returns:
            observation (list): the initial observation.
            info (dict): diagnostic information useful for debugging.'''
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.idx = 0

        #data of price
        if self.mode == "Orderbook":#current price, bid, ask
            idx_data = self.data.iloc[self.idx]
            bid = idx_data["bid"] #list
            ask = idx_data["ask"] #list
            closeprice = (bid[0] + ask[0]) / 2
            observation = np.array([self.balance,self.position,closeprice,bid,ask])
        elif self.mode == "Candlestick":#open, high, low, close, volume
            idx_data = self.data.iloc[self.idx]
            openprice = idx_data["open"]
            highprice = idx_data["high"]
            lowprice = idx_data["low"]
            closeprice = idx_data["close"]
            volume = idx_data["volume"]
            observation = np.array([self.balance,self.position,openprice,highprice,lowprice,closeprice,volume])
        else:
            raise ValueError("mode should be set as Orderbook or Candlestick")
        info = {}
        return observation,info
    
    def step(self,action):
        '''
        The agent takes a step in the environment.

        Parameters:
            action (list): an action provided by the environment

        Returns:
            observation (list): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): diagnostic information useful for debugging. 
        '''
        #action field: [action,size]
        #action: 0 -> hold, 1 -> buy, 2 -> sell
        #size: (size of position)
        act,size = action
        #data of price
        if self.mode == "Orderbook":#current price, bid, ask
            idx_data = self.data.iloc[self.idx]
            bid = idx_data["bid"] #list
            ask = idx_data["ask"] #list
            closeprice = (bid[0] + ask[0]) / 2
        elif self.mode == "Candlestick":#open, high, low, close, volume
            idx_data = self.data.iloc[self.idx]
            openprice = idx_data["open"]
            highprice = idx_data["high"]
            lowprice = idx_data["low"]
            closeprice = idx_data["close"]
            volume = idx_data["volume"]
        else:
            raise ValueError("mode should be set as Orderbook or Candlestick")
        #caculate profit
        if self.position == 0:#no position
            if act == 1:#buy
                if size*closeprice > self.balance:
                    reward = 0

        return observation,reward,done,info
    
    def render(self,mode='human'):
        """Using plotly to render the chart"""
        pass
