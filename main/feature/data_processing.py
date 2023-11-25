
import numpy as np
import pandas as pd
import pandas_ta as ta

def custom_ta(df):
    """
    Apply a series of technical analysis indicators to the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with the technical analysis indicators applied.
    """
    df['feature_open'] = df['open']
    df['feature_high'] = df['high']
    df['feature_low'] = df['low']
    df['feature_close'] = df['close']
    df['feature_volume'] = df['volume']
    df['feature_ma120'] = ta.sma(df['close'], length=120)

    df = df.drop(columns=['date_close'])
    df = df.dropna()

    return df