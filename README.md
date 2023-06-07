<!--
 * @Author: hibana2077 hibana2077@gmaill.com
 * @Date: 2023-05-31 09:37:37
 * @LastEditors: hibana2077 hibana2077@gmaill.com
 * @LastEditTime: 2023-05-31 10:14:58
 * @FilePath: /AI_Trader/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
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
|   |   ├── DQN.py
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
|   ├── lab
│   │   └── lab.py
├── README.md
└── requirements.txt
```
