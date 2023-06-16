<!--
 * @Author: hibana2077 hibana2077@gmaill.com
 * @Date: 2023-05-31 09:37:37
 * @LastEditors: hibana2077 hibana2077@gmaill.com
 * @LastEditTime: 2023-05-31 10:14:58
 * @FilePath: /AI_Trader/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# AI Trader

![python](https://img.shields.io/badge/python-3.10-blue?style=plastic-square&logo=python)
![tensorflow](https://img.shields.io/badge/tensorflow-2.6.0-FF6F00?style=plastic-square&logo=tensorflow)
![pandas](https://img.shields.io/badge/pandas-1.3.3-150458?style=plastic-square&logo=pandas)
![numpy](https://img.shields.io/badge/numpy-1.21.2-013243?style=plastic-square&logo=numpy)
![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-0.21.0-0081A5?style=plastic-square&logo=openai)
![fastapi](https://img.shields.io/badge/fastapi-0.85.1-009688?style=plastic-square&logo=fastapi)
![Guvicorn](https://img.shields.io/badge/Guvicorn-0.19.0-499848?style=plastic-square&logo=Gunicorn)
![mongodb](https://img.shields.io/badge/mongodb-4.4.6-47A248?style=plastic-square&logo=mongodb)
![docker](https://img.shields.io/badge/docker-20.10.8-2496ED?style=plastic-square&logo=docker)


## Introduction

This is a side project I am working on to learn more about reinforcement learning and stock trading. The goal is to create a reinforcement learning agent that can learn to trade stocks in a simulated environment. The agent will be trained on historical stock data and will be able to make decisions on whether to buy, sell, or hold a stock. The agent will be rewarded for making good decisions and penalized for making bad decisions. The agent will be trained using the [OpenAI Gym](https://gym.openai.com/) toolkit, and Deep Q-Learning will useing the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) library. The agent will be trained on historical stock data from [Binance](https://www.binance.com/en) and will be able to trade on the [Binance](https://www.binance.com/en) exchange.

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
|   |   ├── TradingEnv.py
│   │   └── Transformer.py
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