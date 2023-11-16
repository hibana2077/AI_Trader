<!--
 * @Author: hibana2077 hibana2077@gmaill.com
 * @Date: 2023-05-31 09:37:37
 * @LastEditors: hibana2077 hibana2077@gmail.com
 * @LastEditTime: 2023-11-16 23:42:18
 * @FilePath: /AI_Trader/README.md
 * @Description: This page is README.md
-->
# AI Trader

![python](https://img.shields.io/badge/python-3.10-blue?style=plastic-square&logo=python)
![pandas](https://img.shields.io/badge/pandas-1.3.3-150458?style=plastic-square&logo=pandas)
![numpy](https://img.shields.io/badge/numpy-1.21.2-013243?style=plastic-square&logo=numpy)
![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-0.21.0-0081A5?style=plastic-square&logo=openai)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.1-EE4C2C?style=plastic-square&logo=pytorch)
![fastapi](https://img.shields.io/badge/fastapi-0.85.1-009688?style=plastic-square&logo=fastapi)
![Guvicorn](https://img.shields.io/badge/Guvicorn-0.19.0-499848?style=plastic-square&logo=Gunicorn)
![mongodb](https://img.shields.io/badge/mongodb-4.4.6-47A248?style=plastic-square&logo=mongodb)
![docker](https://img.shields.io/badge/docker-20.10.8-2496ED?style=plastic-square&logo=docker)

## Introduction

AI Trader is a cutting-edge, customizable training framework for exploring the fusion of artificial intelligence and financial trading. This project leverages the power of Python, PyTorch, and OpenAI Gym to create a dynamic environment where a reinforcement learning agent is trained with historical stock data.

The framework focuses on Deep Q-Learning, allowing users to tailor their trading strategies and observe how the AI agent adapts to buying, selling, or holding stocks based on market data. With the integration of tools like Pandas, NumPy, FastAPI, Gunicorn, MongoDB, and Docker, AI Trader is not just a theoretical model but a practical, scalable solution for those interested in the future of automated trading. It stands as an ideal platform for enthusiasts and professionals alike to experiment and refine AI-driven trading strategies.

## Features

- **Customizable**: AI Trader is designed to be flexible and easy to use. Users can customize the training environment by adjusting parameters like the stock ticker, the number of training episodes, and the number of days per episode. The framework also allows users to specify the number of days to skip between each episode, which can be useful for training on longer time periods.
- **Multiple Training Interfaces**: AI Trader offers three different training interfaces: a command-line interface, a web interface, and Notebook interface. The command-line interface is ideal for users who want to quickly train an agent and view the results. The web interface is designed for users who want to train an agent and view the results in a browser. The Notebook interface is designed for users who want to directly interact with the training environment and tune the agent's hyperparameters.