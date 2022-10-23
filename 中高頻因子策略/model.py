# %%
import time
import math
import timeit
import os.path
import glob

import numpy as np
import pandas as pd
import pyfolio as pf
import mlfinlab as ml
import seaborn as sns
import featuretools as ft
import pandas_ta as ta

from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from dateutil import parser
from tqdm import tqdm_notebook
from binance.client import Client
from datetime import timedelta, datetime
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# %%
names = ["BTC"]
types = ['tick', 'volume', 'dollar']
start_date = "2021-07-21"
end_date = "2021-8-21"
date_range = pd.date_range(start_date, end_date, freq="D")
parent_dir = "./data"

for name in names:
    ticker = name + "USDT"
    path = os.path.join(parent_dir,  ticker + "/" + types[1])
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

# %%
df = df[(df.date_time > 1626825600930) &
         (df.date_time < 1626998400000)]
df['log_ret'] = np.log(df['close']).diff()
df['maker_percent'] = df['cum_maker_volume'].div(df['volume'])
print(df.head().to_markdown())

# %%
def reversal_factor(log_ret, volume):
    weight_volume = volume / volume.sum()
    wma = np.dot(log_ret[1:], weight_volume.shift(1)[1:])

    return np.sum(wma)

def create_handmade_features(i):
    print(i)
    # df_fun = create_HLCV(i)
    features = pd.DataFrame(index=df.index)

    high = df.high.rolling(i).max()
    low = df.low.rolling(i).min()
    close = df.close.rolling(i).apply(lambda rows: rows.iloc[-1])

    features[f'cum_buy_volume_fold_{i}'] = df.cum_buy_volume / df.cum_buy_volume.rolling(i).mean()
    features[f'cum_maker_volume_fold_{i}'] = df.cum_maker_volume / df.cum_maker_volume.rolling(i).mean()
    features[f'maker_percent_{i}'] = df.maker_percent.rolling(i).median()
    features[f'close_loc_{i}'] = (high - close) / (high - low)
    features[f'close_change_{i}'] = df.log_ret.diff(i)  # Momentum
    # features[f'close_bias_{i}'] = (close - df.close.rolling(i).mean()) / df.close.rolling(i).std()  # 乖離
    # features[f'reversal_{i}'] = df.rolling(i).apply(lambda x: reversal_factor(df.loc[x.index, "log_ret"], df.loc[x.index, "volume"]))# 反轉因子1
    features[f'rsi_{i}'] = ta.rsi(df.close, length=i)  # 反轉因子2
    features[f'sharpe_8_{i}'] = df.log_ret.diff(8).rolling(i).apply(lambda x: x.mean() / x.std(), raw=True) # Sharpe Ratio
    features[f'sharpe_13_{i}'] = df.log_ret.diff(13).rolling(i).apply(lambda x: x.mean() / x.std(), raw=True) # Sharpe Ratio

    return features

def create_bunch_of_features():
    periods = [2, 5, 8, 13, 21, 34, 55, 89]
    results = Parallel(n_jobs=-1)(delayed(create_handmade_features)(period) for period in periods)
    bunch_of_features = pd.DataFrame(index=df.index)
    for result in results:
        bunch_of_features = bunch_of_features.join(result)

    return bunch_of_features

bunch_of_features = create_bunch_of_features()
print(bunch_of_features.head().to_markdown())

# %%
ic = bunch_of_features.corrwith(df.close.pct_change(-1), method='spearman')
ic_plot = ic.sort_values(ascending=False).plot.barh(title='Strength of Correlation(IC)', figsize=(20, 16))
ic_plot.figure.savefig('./fig/corr.png')

# %%
corr_matrix = bunch_of_features.corr()
clustermap_plot = sns.clustermap(corr_matrix, cmap='coolwarm', linewidth=1, method='ward')
clustermap_plot.savefig('./fig/clustermap.png')

# %%
deselected_features = [
    'sharpe_13_2', 'sharpe_8_2',
    'close_change_2'
]

selected_features = bunch_of_features.drop(labels=deselected_features, axis=1)

for feature in selected_features.columns:
    df[feature] = selected_features[feature].fillna(0)

# %%
# Re-compute sides
df['side'] = np.nan

# generate side information
long_signals = (df['maker_percent_2'] > df['maker_percent_5']) & (df['maker_percent_5'] > df['maker_percent_8'])
short_signals = (df['maker_percent_2'] < df['maker_percent_5']) & (df['maker_percent_5'] < df['maker_percent_8'])

# assign side information
df.loc[long_signals, 'side'] = 1
df.loc[short_signals, 'side'] = -1

# Remove Look ahead biase by lagging the signal
df['side'] = df['side'].shift(1)
print(df['side'].value_counts())

# %%

def get_vol(close, lookback=100):
    # daily vol re-indexed to close
    df0 = close / close.shift(1) - 1  # bar return
    df0 = df0.ewm(span=lookback).std()

    return df0


def get_horizons(prices, delta=40):
    t1 = prices.index.searchsorted(prices.index + delta)
    t1 = t1[t1 < prices.shape[0]]
    t1 = prices.index[t1]
    t1 = pd.Series(t1, index=prices.index[:t1.shape[0]])
    return t1


def get_touches(prices, events, factors=[1, 1]):

    out = events[['t1']].copy(deep=True)
    if factors[0] > 0:
        thresh_uppr = factors[0] * events['threshold']
    else:
        thresh_uppr = pd.Series(index=events.index)  # no uppr thresh
    if factors[1] > 0:
        thresh_lwr = -factors[1] * events['threshold']
    else:
        thresh_lwr = pd.Series(index=events.index)  # no lwr thresh

    for loc, t1 in events['t1'].iteritems():

        df0 = prices[loc:t1]                             # path prices
        df0 = (df0 / prices[loc] - 1) * events.side[loc]  # path returns
        out.loc[loc, 'stop_loss'] = df0[df0 < thresh_lwr[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'take_profit'] = df0[df0 > thresh_uppr[loc]].index.min() # earliest take profit

    return out


def get_labels(touches):
    out = touches.copy(deep=True)
    # pandas df.min() ignores NaN values
    first_touch = touches[['stop_loss', 'take_profit']].min(axis=1)
    for loc, t in first_touch.iteritems():
        if pd.isnull(t):
            out.loc[loc, 'label'] = 0
        elif t == touches.loc[loc, 'stop_loss']:
            out.loc[loc, 'label'] = 1
        else:
            out.loc[loc, 'label'] = 2
    return out


# %%
df['threshold'] = get_vol(close=df['close'], lookback=40)
df['t1'] = get_horizons(df['close'])
df['t1'].ffill(inplace=True)
df['t1'] = df['t1'].astype(int)


touches = get_touches(df.close, df[['t1', 'threshold', 'side']], [1, 1])
touches = get_labels(touches)
touches.iloc[-40:, -1] = 0
print(touches.tail())
print(touches['label'].value_counts())

# %%
print(touches.head().to_markdown())
print(df.head().to_markdown())

# %%
X = df.loc[:, ~df.columns.isin(['date_time', 'tick_num', 'open', 'side', 'threshold', 't1'])]
y = touches['label']

# %%
X_train = X.loc[:len(X)*8//10]
y_train = y.loc[:len(y)*8//10]
X_valid = X.loc[len(X)*8//10 + 1:]
y_valid = y.loc[len(y)*8//10 + 1:]

# %%
import catboost
print(catboost.__version__)

model = catboost.CatBoostClassifier(iterations=2000,
                           task_type="GPU",
                           learning_rate=0.03,
                           depth=11,
                           eval_metric='MultiClass',
                           use_best_model=True,
                           l2_leaf_reg=6)


# %%
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

# %%
print('CatBoost model is fitted: ' + str(model.is_fitted()))
print('CatBoost model parameters:')
print(model.get_params())

# %%
# Meta-label
# Performance Metrics
y_pred_rf = model.predict_proba(X_valid)[:, 1]
y_pred = model.predict(X_valid)
print(classification_report(y_valid, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_valid, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_valid, y_pred))


#%%
y_pred = pd.Series(y_pred[:, 0])
print(y_pred.value_counts())

# %%
# Feature Importance
title = 'Feature Importance:'
figsize = (15, 5)

feat_imp = pd.DataFrame({'Importance': model.feature_importances_})
feat_imp['feature'] = X.columns
feat_imp.sort_values(by='Importance', ascending=False, inplace=True)
feat_imp = feat_imp

feat_imp.sort_values(by='Importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title=title, figsize=figsize)
plt.xlabel('Feature Importance Score')
plt.show()
plt.savefig('./fig/feature_importance.png')

# %%
y_train_pred = pd.Series(model.predict(X_train)[:,0])


# %%
print(X_valid.head().to_markdown())
print(y_valid.head().to_markdown())
df_train = pd.concat([X_train, y_train_pred], axis=1)
df_validate = pd.concat([X_valid, y_pred], axis=1)

# %%
fig = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
ax1.plot(df_train['close'][100:150], label='close')
ax2.plot(df_train['predict'][100:150], label='y_valid')
plt.show()







