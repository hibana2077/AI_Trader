'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-11-25 13:45:50
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-11-26 09:33:35
FilePath: \AI_Trader\main\test\a2c.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import gymnasium as gym

from stable_baselines3 import A2C,PPO

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=20_000,progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()