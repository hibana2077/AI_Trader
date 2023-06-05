import ccxt.pro as ccxtpro
import pandas as pd
import logging
import asyncio
import time as t
from datetime import datetime

#setting debug mode
DEBUG = False

#setting loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def close_exchange(exchange:ccxtpro.Exchange):
    """Close exchange"""
    await exchange.close()
    logger.info('exchange closed.')

async def get_orderbook(data:dict,exchange:ccxtpro.Exchange,data_length:int):
    """Get orderbook"""
    s = t.time()
    for idx in range(data_length):
        try:
            orderbook = await exchange.watch_order_book('SUI/USDT')
            if DEBUG:
                logger.info(orderbook)
            asks = orderbook['asks']
            bids = orderbook['bids']
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if DEBUG:print(time)
            data[time] = {'asks':asks,'bids':bids}
            if idx % 100 == 0:
                logger.info(f'progress: {idx/data_length*100:.2f}% , {idx}/{data_length} , ETA: {((t.time()-s)/idx*(data_length-idx))/60:.2f}min')
                s = t.time()
        except Exception as e:
            logger.error(e)
            logger.error('Something went wrong, please check the log.')
            return
    return data

def main():
    """Main function"""
    data = dict()
    logger.info('initializing exchange...')
    exchange = ccxtpro.binance({
        'options': {
            'tradesLimit': 1000,
            'ordersLimit': 1000,
            'OHLCVLimit': 1000,
        },
    })
    logger.info('exchange initialized.')
    logger.info('collecting data...')
    asyncio.get_event_loop().run_until_complete(get_orderbook(data=data,exchange=exchange,data_length=100000))
    logger.info('data collected.')
    df = pd.DataFrame.from_dict(data,orient='index')
    file_name = 'test.csv'
    df.to_csv(file_name)
    asyncio.get_event_loop().run_until_complete(close_exchange(exchange=exchange))
    asyncio.get_event_loop().close()

if __name__ == '__main__':
    main()