
from collections import deque
import random
import gym
import numpy as np
from keras import models, layers, optimizers

"""
"""


class DQN(object):
    def __init__(self,update_freq=200,replay_size=2000,model_layers=[2,100,3]):
        self.step = 0
        self.update_freq = update_freq  # 模型更新頻率
        self.replay_size = replay_size  # 訓練集大小
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.layer_info = model_layers

    def create_model(self):
        """創建一個神經網路"""
        STATE_DIM, ACTION_DIM = self.layer_info[0], self.layer_info[-1]
        model = models.Sequential()
        model.add(layers.Dense(self.layer_info[1], input_dim=STATE_DIM, activation='relu'))
        for i in range(2,len(self.layer_info)-1):
            model.add(layers.Dense(self.layer_info[i], activation='relu'))
        model.add(layers.Dense(ACTION_DIM, activation="linear"))
        model.compile(loss='mean_squared_error',
                        optimizer=optimizers.Adam(0.001))
        return model

    def summary(self):
        '''輸出模型摘要資訊'''
        self.model.summary()

    def act(self, s, epsilon=0.1):
        """預測動作"""
        # 一開始用隨機的動作探索環境
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]),verbose=str(0))[0])

    def save_model(self, file_path='MountainCar-v0-dqn.h5'):
        """保存模型"""
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward):
        """歷史紀錄，position >= 0.4时给額外的reward，快速收斂"""
        if next_s[0] >= 0.4:
            reward += 1
        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=64, lr=1, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        # 每 update_freq 步，將 model 的权重赋值给 target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))
 
        # 传入网络进行训练
        self.model.fit(s_batch, Q, verbose=str(0))
