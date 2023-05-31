'''
Author: hibana2077 hibana2077@gmaill.com
Date: 2023-05-31 11:38:25
LastEditors: hibana2077 hibana2077@gmaill.com
LastEditTime: 2023-05-31 11:39:38
FilePath: /AI_Trader/main/utils/collect_data.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import ccxt
import pandas as pd
import logging
import time

#setting loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('logs/Tradingenv.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


