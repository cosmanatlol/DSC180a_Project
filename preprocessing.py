import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta


def preprocess_train(stock_symbols, start, end):
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
        {'kind':'bbands', 'col_names': ('BBL', 'BBM', 'BBU', 'BBB', 'BBP')},
        {'kind': 'ao', 'col_names': 'ao'},
        {'kind':'adx', 'col_names': ('adx', 'dmp', 'dmn',)},
        {'kind':'chop', 'col_names': 'chop'},
        ]
    )
    stock_dic = {}
    for x in stock_symbols:
        cur = yf.download(x, start=start, end=end).reset_index()
        cur.ta.strategy(strat)
        #5 is strong buy, 4 is buy, 3 is weak buy, 2 is weak sell, 1 is sell, 0 is strong sell
        pct_rec = lambda x: 5 if x > 0.04 else 4 if x > 0.02 else 3 if x > 0 else 0 if x < -0.04 else 1 if x < -0.02 else 2 
        #pct_rec = lambda x: 1 if x > 0 else 0 
        cur['pct_change'] = cur['Close'].pct_change().shift(-1)
        cur['target'] = cur['pct_change'].apply(pct_rec)
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
    df['macd_rec'] = (df['macd_h'] > 3).astype(int) + (df['macd_h'] > 0.8).astype(int) - (df['macd_h'] < -0.8).astype(int) - (df['macd_h'] < -3).astype(int)
    df['mom_rec'] = (df['momentum'] > 30).astype(int) + (df['momentum'] > 6).astype(int) - (df['momentum'] < -6).astype(int) - (df['momentum'] < -30).astype(int)
    df['bbands_rec'] = (df['BBU'] < df['Close']).astype(int) - (df['BBL'] > df['Close']).astype(int)
    df['ao_rec'] = (df['ao'] > 30).astype(int) + (df['ao'] > 6).astype(int) - (df['ao'] < -6).astype(int) - (df['ao'] < -30).astype(int)
    df['chop'] = df['chop']/100
    return df
    
