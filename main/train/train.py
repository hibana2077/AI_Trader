'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-11-16 21:29:59
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-11-25 11:13:39
FilePath: \AI_Trader\main\train\train.py
Description: This is a training script for DQN agent (CLI)
'''
import math
import random
import argparse
import datetime
import matplotlib
import gym_trading_env

import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
import pandas_ta as ta

from collections import namedtuple, deque
from gym_trading_env.downloader import download
from itertools import count
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils.webhook_notify as notify
import feature.data_processing as data_process

# Arguments
parser = argparse.ArgumentParser(description='Train a DQN agent on a single stock or crypto currency')
parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Symbol to train on (BTC/USDT, AAPL, etc)')
parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to train on (5m, 15m, 1h, 4h, 1d, 1w, 1M)')
parser.add_argument('--start_date', type=str, default='2023-10-10', help='Start date to train on (YYYY-MM-DD)')
parser.add_argument('--exchange', type=str, default='binance', help='Exchange to train on (binance, bitfinex, bitmex, bitstamp, coinbasepro, huobi, kraken, kucoin, okex)')
parser.add_argument('--discord_webhook_url', type=str, default='', help='Discord webhook url to send training results to')
parser.add_argument('--dir', type=str, default='../../data', help='Directory to store data in (default: ../../data)')
parser.add_argument('--initial_position', type=float, default=0, help='Initial position (default: 0)')
parser.add_argument('--trading_fees', type=float, default=0.02/100, help='Trading fees (default: 0.0002)')
parser.add_argument('--positions', type=list, default=[-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8], help='List of positions to train on (default: [-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8])')
parser.add_argument('--window_size', type=int, default=0, help='Window size (default: 0)')
args = parser.parse_args()

# Constants
DIR = args.dir
SYMBOL = args.symbol # BTC/USDT
EXCHANGE = args.exchange # binance
TIMEFRAME = args.timeframe # 1h
POSITIONS = args.positions # [-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]
DISCORD_WEBHOOK_URL = args.discord_webhook_url
INITIAL_POSITION = args.initial_position
TRADING_FEES = args.trading_fees
WINDOWS_SIZE = args.window_size if args.window_size > 0 else None

# Download BTC/USDT historical data from Binance and stores it to directory ./data/binance-BTCUSDT-1h.pkl
download(exchange_names = [EXCHANGE],
    symbols= [SYMBOL],
    timeframe= TIMEFRAME,
    dir = DIR,
    since= datetime.datetime(year= 2023, month= 10, day= 10),
)
# Import your fresh data
df = pd.read_pickle(DIR+"/"+EXCHANGE+"-"+SYMBOL.replace("/","")+"-"+TIMEFRAME+".pkl")

# Handle more feature
df = data_process.custom_ta(df)

env = gym.make("TradingEnv", df = df, positions = POSITIONS, initial_position=INITIAL_POSITION, trading_fees=TRADING_FEES, window=WINDOWS_SIZE)

_,info = env.reset()
print(info.keys())

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module): # you can customize this to your liking

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Return %')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    if show_result:
        plt.savefig('result.png')
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 5

start_time = time()

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            Portfolio_Return:str = env.unwrapped.get_metrics()['Portfolio Return']
            episode_durations.append(float(Portfolio_Return.replace("%","")))
            plot_durations()
            break

end_time = time()
print('Complete !')
print(f'Training took {(end_time - start_time)/60} minutes')
plot_durations(show_result=True)

plt.ioff()
plt.show()

notify.send_text_report(DISCORD_WEBHOOK_URL, end_time - start_time, env.unwrapped.get_metrics())
notify.send_image_report(DISCORD_WEBHOOK_URL, "result.png")