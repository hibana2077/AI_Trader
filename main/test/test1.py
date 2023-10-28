'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-10-28 15:06:17
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-10-28 23:31:57
FilePath: \AI_Trader\main\test\test1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import random
import torch
import numpy as np
import pandas as pd
from model.model1 import LSTMClassifier as Model1
from utils.tradingENV import CryptoTradingEnv
from collections import deque

# 初始化
df = pd.read_csv("../../data/SUIUSDT_1h.csv")
env = CryptoTradingEnv(df=df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model1(input_shape=env.observation_space.shape, num_actions=env.action_space.n)  # 假設你的模型初始化像這樣
memory = deque(maxlen=10000)  # 經驗回放緩衝區

# Offline learning階段
print("Offline learning...")
for _ in range(1000):  # 隨機探索1000次
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        memory.append((obs, action, reward, next_obs, done))
        obs = next_obs

# Online learning階段
print("Online learning...")
for episode in range(100):  # 100個交易周期
    obs = env.reset()
    done = False
    while not done:
        # 使用模型選擇動作
        predit = model(obs)#return is [[x,x,x],x,[x,x]]
        action_pre = torch.argmax(predit[0]).item()
        quantity = predit[1].item()
        reversing = torch.argmax(predit[2]).item() # 1 or 0
        action = [action, quantity, reversing]
        
        # 進行一步
        next_obs, reward, done, _ = env.step(action)

        # 儲存到經驗回放緩衝區
        memory.append((obs, action, reward, next_obs, done))

        # 從經驗回放中隨機抽取小批量樣本
        batch = random.sample(memory, min(len(memory), 64))
        
        # 準備訓練數據
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
        obs_batch = np.array(obs_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_obs_batch = np.array(next_obs_batch)
        done_batch = np.array(done_batch)

        # 訓練模型（您需要實現這個部分，這裡僅為示例）
        model.train(obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)

        # 更新觀察值
        obs = next_obs
