import gym
import numpy as np
import pandas as pd
from gym import spaces
from backtesting import Backtest, Strategy

def process_observations(df:pd.DataFrame):
    # Add your own logic here
    return df.values

# Strategy is for calculating reward
def SIGNAL(df):
    return df['SIGNAL']

def QUANTITY(df):
    return df['QUANTITY']

def REVERSING(df):
    return df['REVERSING']

class Scalping_Strategy(Strategy):
    def init(self):
        super().init() 
        self.signal = self.I(SIGNAL, self.data)
        self.quantity = self.I(QUANTITY, self.data)
        self.reversing = self.I(REVERSING, self.data)

    def next(self):
        super().next()

        size_pre = round((float(self.quantity[-1])*self.equity)/float(self.data.Close[-1]))
        if self.reversing == 1:
            self.position.close()

        if self.signal == 1:
            self.buy(size=size_pre)
        elif self.signal == 2:
            self.sell(size=size_pre)

class CryptoTradingEnv(gym.Env):
    def __init__(self, df:pd.DataFrame):
        self.df:pd.DataFrame = df
        self.current_step:int = 0
        self.window_size:int = 50
        self.init_cash:float = 10000
        self.commision:float = 0.0005
        self.df['SIGNAL'] = 0
        self.df['QUANTITY'] = 0
        self.df['REVERSING'] = 0
        
        # Improved observation space *len need to add a number of other indicators(final equity, trades...)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.window_size, len(self.df.columns)), dtype=np.float32)
        
        # Continuous action space ((buy , sell , hold), quantity of percentage, reversing trade)
        # Reversing trade will close current trade and open a new trade with opposite direction
        # (0,0,0) = hold , (1,1,0) = buy 100% , (2,0.5,1) = sell 50% and reversing trade
        self.action_space = spaces.Box(low=0, high=2, shape=(3,), dtype=np.float32)
        
    def step(self, action):
        # Execute trade based on action
        # action = [action(int), quantity(float), reversing(int)]
        now_index = self.current_step + self.window_size
        self.df.loc[now_index, 'SIGNAL'] = action[0]
        self.df.loc[now_index, 'QUANTITY'] = action[1]
        self.df.loc[now_index, 'REVERSING'] = action[2]

        temp_data = self.df.iloc[self.current_step:now_index+1]

        bt = Backtest(temp_data, Scalping_Strategy, cash=self.init_cash, commission=self.commision, exclusive_orders=True)
        stats = bt.run()
        
        # Calculate reward with custom logic
        reward = self.calculate_reward(action)
        
        info = {}
        
        # Check if episode is done
        self.current_step += 1
        if self.current_step > len(self.df)-1:
            done = True
        else:
            done = False
            
        return self._next_observation(), reward, done, info
    
    def _next_observation(self):
        # Return processed observations
        obs = process_observations(self.df.iloc[self.current_step])
        return obs
            
    # Custom reward function
    def calculate_reward(self, action): 
        reward = 0
        
        return reward
    
    def reset(self):
        self.current_step = 0
        return self._next_observation()
