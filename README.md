# AI Trader

## Introduction

This is a side project I am working on to learn more about reinforcement learning and stock trading. The goal is to create a reinforcement learning agent that can learn to trade stocks in a simulated environment. The agent will be trained on historical stock data and will be able to make decisions on whether to buy, sell, or hold a stock. The agent will be rewarded for making good decisions and penalized for making bad decisions. The agent will be trained using the [OpenAI Gym](https://gym.openai.com/) toolkit, and Deep Q-Learning will useing the [TensorFlow](https://www.tensorflow.org/) library. Data will be collected from [Binance](https://www.binance.com/en) using the [ccxt](https://github.com/ccxt/ccxt) library.

## File Structure

```
.
├── data
│   ├── binance
│   │   ├── BTCUSDT
│   │   │   ├── 1m
│   │   │   │   ├── 2017-2018.csv
│   │   │   │   ├── 2018-2019.csv
|   |   |   |   ...
│   │   │   ├── 5m
│   │   │   ├── 15m
│   │   │   ├── 30m
│   │   │   ├── 1h
│   │   │   ├── 4h
│   │   │   └── 1d
│   │   ...
|   ...
├── main
│   ├── utils
│   │   ├── collect_data.py
│   │   ├── preprocess_data.py
│   │   └── TradingEnv.py
|   ├── train
│   │   ├── train.py
│   │   └── model.py
|   ├── test
│   │   └── test.py
|   ├── run
│   │   ├── run.py
|   |   ├── Dockerfile
|   |   └── docker-compose.yml
├── README.md
├── requirements.txt
```
