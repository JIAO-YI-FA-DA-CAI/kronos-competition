# %%
import pandas as pd
import numpy as np

#%%
df_backtest = pd.read_csv('./dfs/backtest.csv')

# %%
fund = 10000 #初始有10000元本金
money = 10000 #每次投入10000元
first = 0
BS = None
timeList = []
buy = []
sell = []
sellshort = []
buytocover = []
profit_list = [0]
profit_fee_list = [0]
feeRate = 0.0002

for i in range(len(df_backtest)):

    if i == len(df_backtest) - 1:
        break

    if BS == None:
        profit_list.append(0)
        profit_fee_list.append(0)
        
        if df_backtest.loc[i - 20:i, 'predict'].isin([2]).any():

            if (df_backtest.loc[i, 'maker_percent_2'] > df_backtest.loc[i, 'maker_percent_5']) & (
                    df_backtest.loc[i, 'maker_percent_5'] > df_backtest.loc[i, 'maker_percent_8']):
                temp = df_backtest.loc[i, 'close']
                tempSize = money / df_backtest.loc[i, 'close']
                BS = 'B'
                t = i + 1
                buy.append(t)


            # elif (df_backtest.loc[i, 'maker_percent_2'] < df_backtest.loc[i, 'maker_percent_5']) & (
            #         df_backtest.loc[i, 'maker_percent_5'] < df_backtest.loc[i, 'maker_percent_8']) and df_backtest.loc[i, 'rsi_2'] > df_backtest.loc[i, 'rsi_5']:
            #     temp = df_backtest.loc[i, 'close']
            #     tempSize = -money / df_backtest.loc[i, 'close']
            #     BS = 'S'
            #     t = i + 1
            #     sellshort.append(t)

    elif BS == 'B':
        profit = tempSize * (df_backtest['close'][i] - temp)
        profit_list.append(profit)

        exit_singal1 = df_backtest.loc[i, 'predict'] == 1
  

        if exit_singal1:
            pl_round = tempSize * (df_backtest['close'][i+1] - df_backtest['close'][t])
            profit_fee = profit - money*feeRate - (money+pl_round)*feeRate
            profit_fee_list.append(profit_fee)
            sell.append(i+1)
            BS = None
        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)
    # elif BS == 'S':
    #     profit = tempSize * (temp - df_backtest['close'][i])
    #     profit_list.append(profit)
    #
    #     exit_singal1 = df_backtest.loc[i, 'predict'] == 1
    #     exit_signal2 = (df_backtest.loc[i, 'maker_percent_2'] > df_backtest.loc[i, 'maker_percent_5']) & (
    #                 df_backtest.loc[i, 'maker_percent_5'] > df_backtest.loc[i, 'maker_percent_8'])
    #
    #     if exit_singal1:
    #         buytocover.append(profit)
    #         BS = None
    
#%%
equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=df_backtest.index)
print(equity)
equity.plot(grid=True, figsize=(12, 6));
# %%
