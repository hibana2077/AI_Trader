'''
Author: hibana2077 hibana2077@gmaill.com
Date: 2023-05-31 11:38:25
LastEditors: hibana2077 hibana2077@gmaill.com
LastEditTime: 2023-05-31 11:56:16
FilePath: /AI_Trader/main/utils/collect_data.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import ccxt
import pandas as pd
import logging
import os
import time

#setting loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#setting file dir
def set_file_dir():
    """Match programe file setting to current file dir"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logger.info('now in {}'.format(os.getcwd()))
    
#setting exchange
def set_exchange():
    """Set exchange"""
    exchange_list = ["binanceusdm","bybit","bitget","okex5"]
    logger.info('exchange list: {}'.format(exchange_list))
    exchange = input('please choose exchange: ')
    return exchange

#main function
def main():
    try:
        set_file_dir()
        exchange = getattr(ccxt, set_exchange())()
        logger.info('exchange: {}'.format(exchange))

    except Exception as e:
        logger.error(e)
        logger.error('Something went wrong, please check the log.')
        return

#main 
if __name__ == '__main__':
    main()