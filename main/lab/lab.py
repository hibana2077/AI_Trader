# import ccxt.pro as ccxtpro
# import pandas as pd
# import numpy as np
# import asyncio

# exchange = ccxtpro.binance({
#     'options': {
#         'tradesLimit': 1000,
#         'ordersLimit': 1000,
#         'OHLCVLimit': 1000,
#     },
# })

# data = dict()

# async def close_exchange(exchange:ccxtpro.Exchange):
#     """Close exchange"""
#     await exchange.close()
#     # logger.info('exchange closed.')
#     print('exchange closed.')

# async def get_orderbook(data:dict,exchange:ccxtpro.Exchange,data_length:int):
#     """Get orderbook"""
#     for idx in range(data_length):
#         try:
#             orderbook = await exchange.watch_order_book('SUI/USDT')
#             asks = orderbook['asks']
#             bids = orderbook['bids']
#             time = orderbook['datetime']
#             data[time] = {'asks':asks,'bids':bids}
#         except Exception as e:
#             print(e)
#             print('Something went wrong, please check the log.')
#             return
#     return data

# # asyncio.get_event_loop().run_until_complete(get_orderbook(data=data,exchange=exchange,data_length=10))

# data = asyncio.get_event_loop().run_until_complete(get_orderbook(data=data,exchange=exchange,data_length=10))
# print(len(data))
# df = pd.DataFrame.from_dict(data,orient='index')
# file_name = 'test.csv'
# df.to_csv(file_name)
# print(df.index)
# asyncio.get_event_loop().run_until_complete(close_exchange(exchange=exchange))
# asyncio.get_event_loop().close()

x,y=[1,4]
print(x,y)