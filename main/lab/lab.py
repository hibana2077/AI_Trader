# import ccxt.pro as ccxtpro
# import pandas as pd
# import numpy as np
# import asyncio

# exchange = ccxtpro.binance({
#     'options': {
#         'tradesLimit': 1000,
#         'ordersLimit': 1000,
#         'OHLCVLimit': 1000,
#     },
# })

# data = dict()

# async def close_exchange(exchange:ccxtpro.Exchange):
#     """Close exchange"""
#     await exchange.close()
#     # logger.info('exchange closed.')
#     print('exchange closed.')

# async def get_orderbook(data:dict,exchange:ccxtpro.Exchange,data_length:int):
#     """Get orderbook"""
#     for idx in range(data_length):
#         try:
#             orderbook = await exchange.watch_order_book('SUI/USDT')
#             asks = orderbook['asks']
#             bids = orderbook['bids']
#             time = orderbook['datetime']
#             data[time] = {'asks':asks,'bids':bids}
#         except Exception as e:
#             print(e)
#             print('Something went wrong, please check the log.')
#             return
#     return data

# # asyncio.get_event_loop().run_until_complete(get_orderbook(data=data,exchange=exchange,data_length=10))

# data = asyncio.get_event_loop().run_until_complete(get_orderbook(data=data,exchange=exchange,data_length=10))
# print(len(data))
# df = pd.DataFrame.from_dict(data,orient='index')
# file_name = 'test.csv'
# df.to_csv(file_name)
# print(df.index)
# asyncio.get_event_loop().run_until_complete(close_exchange(exchange=exchange))
# asyncio.get_event_loop().close()

# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt

# # 生成範例數據
# data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# # 使用 KMeans 演算法，假設有4個集群
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(data)

# # 預測每個數據點的集群
# pred = kmeans.predict(data)

# # 繪製結果
# plt.scatter(data[:, 0], data[:, 1], c=pred, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.show()


# import gym
# from stable_baselines3 import PPO

# env = gym.make('MountainCar-v0')
# env.reset()

# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)

# episodes = 10

# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action, _ = model.predict(state)
#         n_state, reward, done, info = env.step(action)
#         score += reward
#         state = n_state
#     print('Episode:{} Score:{}'.format(episode, score))
#     env.close()