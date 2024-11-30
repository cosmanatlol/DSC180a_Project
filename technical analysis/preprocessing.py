import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta


def preprocess(stock_symbols, start, end, train = False):
    strat = ta.Strategy(
    name = 'Best Strategy Ever',
    ta = [
        {'kind':'ema', 'length': 10, 'col_names': 'ema_10'},
        {'kind':'ema', 'length': 25, 'col_names': 'ema_25'},
        {'kind':'ema', 'length': 50, 'col_names': 'ema_50'},
        {'kind':'ema', 'length': 100, 'col_names': 'ema_100'},
        {'kind':'hma', 'length': 10, 'col_names': 'hma_10'},
        {'kind':'hma', 'length': 25, 'col_names': 'hma_25'},
        {'kind':'hma', 'length': 50, 'col_names': 'hma_50'},
        {'kind':'hma', 'length': 100, 'col_names': 'hma_100'},
        {'kind':'macd', 'col_names': ('macd', 'macd_h', 'macd_s')},
        {'kind':'rsi', 'col_names': 'rsi'},
        {'kind':'mom', 'col_names': 'momentum'},
        {'kind':'bbands','std': 1, 'col_names': ('BBL', 'BBM', 'BBU', 'BBB', 'BBP')},
        {'kind': 'ao', 'col_names': 'ao'},
        {'kind':'adx', 'col_names': ('adx', 'dmp', 'dmn',)},
        {'kind':'chop', 'col_names': 'chop'},
        ]
    )
    stock_dic = {}
    for x in stock_symbols:
        cur = yf.download(x, start=start, end=end).reset_index()
        cur.ta.strategy(strat)
        cur['pct_change'] = cur['Close'].pct_change()
        if train:
            #0 means sell, 1 means buy
            pct_rec = lambda x: 1 if x > 0 else 0 
            cur['pct_change_shift'] = cur['pct_change'].shift(-1)
            cur['target'] = cur['pct_change_shift'].apply(pct_rec)
        stock_dic[x] = recommendation(cur.dropna())

    return stock_dic

def recommendation(df):
    pd.options.mode.chained_assignment = None 
    ma_rec = lambda x: 2 if x > 0.1 else 1 if x > 0.02 else -2 if x < -0.1 else -1 if x < -0.02 else 0
    df['ema_rec_short'] = ((df['ema_10'] - df['Close'])/df['Close']).apply(ma_rec) + ((df['ema_25'] - df['Close'])/df['Close']).apply(ma_rec)
    df['ema_rec_long'] = ((df['ema_50'] - df['Close'])/df['Close']).apply(ma_rec) + ((df['ema_100'] - df['Close'])/df['Close']).apply(ma_rec)
    df['hma_rec_short'] = ((df['hma_10'] - df['Close'])/df['Close']).apply(ma_rec) + ((df['hma_25'] - df['Close'])/df['Close']).apply(ma_rec)
    df['hma_rec_long'] = ((df['hma_50'] - df['Close'])/df['Close']).apply(ma_rec) + ((df['hma_100'] - df['Close'])/df['Close']).apply(ma_rec)
    rsi_rec = lambda x: 2 if x < 30 else 1 if x < 40 else -2 if x > 70 else -1 if x > 60 else 0
    df['rsi_rec'] = df['rsi'].apply(rsi_rec)
    df['macd_rec'] = (df['macd_h'] > .3).astype(int) + (df['macd_h'] > 0).astype(int) - (df['macd_h'] < 0).astype(int) - (df['macd_h'] < -0.3).astype(int)
    df['mom_rec'] = (df['momentum'] > 5).astype(int) + (df['momentum'] > 0.5).astype(int) - (df['momentum'] < -0.5).astype(int) - (df['momentum'] < -5).astype(int)
    df['bbands_rec'] = (df['BBU'] < df['Close']).astype(int) - (df['BBL'] > df['Close']).astype(int)
    df['ao_rec'] = (df['ao'] > 150).astype(int) + (df['ao'] > 0.5).astype(int) - (df['ao'] < -0.5).astype(int) - (df['ao'] < -5).astype(int)
    df['chop'] = df['chop']/100
    return df
    
