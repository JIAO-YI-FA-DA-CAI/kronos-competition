import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import talib
import random
import csv

import warnings
warnings.filterwarnings("ignore")

fed_meeting_timelist = ['2022-1-26 19:24', '2022-3-16 19:23', '2022-5-4 19:16', '2022-6-15 19:25', '2022-7-27 19:23', '2022-9-21 19:11']
eco_predict_timelist = ['2021-3-17 17:59', '2021-6-16 17:59', '2021-9-22 17:59', '2021-12-15 18:59']

event_2021_3_17 = pd.read_csv('event/event_2021_3_17.csv', parse_dates=True, index_col='Unnamed: 0')
event_2021_6_16 = pd.read_csv('event/event_2021_6_16.csv', parse_dates=True, index_col='Unnamed: 0')
event_2021_9_22 = pd.read_csv('event/event_2021_9_22.csv', parse_dates=True, index_col='Unnamed: 0')
event_2021_12_15 = pd.read_csv('event/event_2021_12_15.csv', parse_dates=True, index_col='Unnamed: 0')
event_2022_1_26 = pd.read_csv('event/event_2022_1_26.csv', parse_dates=True, index_col='Unnamed: 0')
event_2022_3_16 = pd.read_csv('event/event_2022_3_16.csv', parse_dates=True, index_col='Unnamed: 0')
event_2022_5_4 = pd.read_csv('event/event_2022_5_4.csv', parse_dates=True, index_col='Unnamed: 0')
event_2022_6_15 = pd.read_csv('event/event_2022_6_15.csv', parse_dates=True, index_col='Unnamed: 0')
event_2022_7_27 = pd.read_csv('event/event_2022_7_27.csv', parse_dates=True, index_col='Unnamed: 0')
event_2022_9_21 = pd.read_csv('event/event_2022_9_21.csv', parse_dates=True, index_col='Unnamed: 0')

events = [event_2021_3_17, event_2021_6_16, event_2021_9_22, event_2021_12_15, 
          event_2022_1_26, event_2022_3_16, event_2022_5_4, event_2022_6_15, event_2022_7_27, event_2022_9_21]

for event in events:
    event['buy_sell_ratio'] = event['buy_volume'] / event['sell_volume'] - 1
    event['cum_Volume'] = event['Volume'].cumsum()
    event['cum_buy_volume'] = event['buy_volume'].cumsum()
    event['cum_sell_volume'] = event['sell_volume'].cumsum()
    event['cum_buy_sell_ratio'] = event['cum_buy_volume'] / event['cum_sell_volume'] - 1
    event['buy_sell_twap_ratio'] = (event['buy_twap'] / event['sell_twap'] - 1) * 100
    event['cum_buy_sell_twap_ratio'] = event['buy_sell_twap_ratio'].cumsum() / [k+1 for k in range(len(event['buy_sell_twap_ratio']))]

# for i,j in zip(range(len(event.columns)), event.columns):
#     print(i,j)

def save_csv(list_, path):
    with open(path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list_)

def backtest(event, event_name, K_loss=0.02, start_time=5, stop_time=25, _print=False, _plot=False):

    fund = 10000
    money = 10000
    feeRate = 0

    BS = None
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    high = -10**8
    low = 10**8
    times = 0

    df_arr = np.array(event)

    for i in range(len(df_arr)):

        if i == len(df_arr)-1:
            break
            
        ## twap 比率
        if i <= start_time:
            entryLong = False
            entrySellShort = False
        else:
            if df_arr[i,14] >= 0 and df_arr[i,16] >= 0:
                entryLong = True
                entrySellShort = False
            elif df_arr[i,14] < 0 and df_arr[i,16] < 0:
                entryLong = False
                entrySellShort = True

            if times >= stop_time:
                exitShort = True
                exitBuyToCover = True
            else:
                exitShort = False
                exitBuyToCover = False
                
        ## 停利停損邏輯
        if BS == 'B':
            stopLossMoving = df_arr[i,9] <= high * (1-K_loss)
            
        elif BS == 'S':
            stopLossMoving = df_arr[i,7] >= low * (1+K_loss)
            
        if BS == None:
            profit_list.append(0)
            profit_fee_list.append(0)
            
            if entryLong:
                tempSize = money / df_arr[i+1,7]
                BS = 'B'
                t = i+1
                buy.append(t)
                high = df_arr[i,0]
                times = 1

            elif entrySellShort:
                tempSize = money / df_arr[i+1,9]
                BS = 'S'
                t = i+1
                sellshort.append(t)
                low = df_arr[i,0]
                times = 1
                
        elif BS == 'B':
            profit = tempSize * (df_arr[i+1,0] - df_arr[i,0])
            profit_list.append(profit)
            times += 1
            
            if df_arr[i,0] > high:
                high = df_arr[i,0]

            if exitShort or i == len(df_arr)-2 or stopLossMoving:
                pl_round = tempSize * (df_arr[i+1,9] - df_arr[t,7])
                profit_fee = profit - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                BS=None
                high = -10**8

                # Realized PnL
                profit_fee_realized = pl_round - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list_realized.append(profit_fee_realized)
                break

            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                
        elif BS == 'S':
            profit = tempSize * (df_arr[i,0] - df_arr[i+1,0])
            profit_list.append(profit)
            times += 1

            if df_arr[i,0] < low:
                low = df_arr[i,0]
            
            if exitBuyToCover or i == len(df_arr)-2 or stopLossMoving:
                pl_round = tempSize * (df_arr[t,9] - df_arr[i+1,7])
                profit_fee = profit - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list.append(profit_fee)
                buytocover.append(i+1)
                BS=None
                low = 10**8

                # Realized PnL
                profit_fee_realized = pl_round - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list_realized.append(profit_fee_realized)
                break

            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)         
                
    equity = pd.DataFrame({'profit':np.cumsum(profit_fee_list)}, index=event.index[:len(profit_fee_list)])
    equity['equity'] = equity['profit'] + fund
    equity['drawdown_percent'] = (equity['equity'] / equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax()
    
    ret = equity['equity'][-1]/equity['equity'][0] - 1
    mdd = abs(equity['drawdown_percent'].min())
    if mdd != 0:
        calmarRatio = ret / mdd
    else:
        calmarRatio = np.nan
    tradeTimes = len(buy)+len(sellshort)

    if _print == True: 
        print(f'<{event_name}>')
        print(f'return: {np.round(ret,4)*100}%')
        # print(f'mdd: {np.round(mdd,4)*100}%')
        print(f'calmarRatio: {np.round(calmarRatio,2)}')
        print(f'tradeTimes: {tradeTimes}')

    if _plot == True:
        equity.plot(figsize=(12,6), ylabel='PnL')
        event['Close'].plot(alpha=0.7, grid=True, secondary_y=True)
        plt.title(f'Equity Curve {event_name}')
        plt.ylabel('BTC Spot Price')
        plt.show();

    return equity


# %%

# events = [event_2021_3_17, event_2021_6_16, event_2021_9_22, event_2021_12_15, 
#           event_2022_1_26, event_2022_3_16, event_2022_5_4, event_2022_6_15, event_2022_7_27, event_2022_9_21]

# events_name = ['event_2021_3_17', 'event_2021_6_16', 'event_2021_9_22', 'event_2021_12_15', 
#                'event_2022_1_26', 'event_2022_3_16', 'event_2022_5_4', 'event_2022_6_15', 'event_2022_7_27', 'event_2022_9_21']

events = [event_2021_3_17, event_2021_6_16, event_2021_9_22, event_2021_12_15]
events_name = ['event_2021_3_17', 'event_2021_6_16', 'event_2021_9_22', 'event_2021_12_15']

simulated = 100000
random.seed(random.randint(0, 13))
K_loss_list = [0.02, 0.04]
start_time_list = [random.randint(5, 720) for i in range(simulated)]
stop_time_list = [random.randint(60, 2880) for i in range(simulated)]

for K_loss in K_loss_list:
    for start_time, stop_time in zip(start_time_list, stop_time_list):
        if start_time < stop_time:
            nice_output = 0
            for event, event_name in zip(events, events_name):
                ret, calmarRatio = backtest(event=event, event_name=event_name, K_loss=K_loss, start_time=start_time, stop_time=stop_time)
                if ret > 0 and calmarRatio > 1.2:
                    nice_output += 1
                    if nice_output >= 3:
                        save_csv(list_=[start_time, stop_time, K_loss], path='eventOutput_2021.csv')
                        print(start_time, stop_time, K_loss)


# %%


params = pd.read_csv('eventOutput_2021.csv', header=None)

start_time_list = np.array(params[0])
stop_time_list = np.array(params[1])

events = [event_2021_3_17, event_2021_6_16, event_2021_9_22, event_2021_12_15]
events_name = ['event_2021_3_17', 'event_2021_6_16', 'event_2021_9_22', 'event_2021_12_15']

for start_time, stop_time in zip(start_time_list, stop_time_list):
    if start_time < stop_time:
        nice_output = 0
        for event, event_name in zip(events, events_name):
            ret, calmarRatio = backtest(event=event, event_name=event_name, K_loss=0.02, start_time=start_time, stop_time=stop_time)
            if ret > 0 and calmarRatio > 2:
                nice_output += 1
                if nice_output >= 3:
                    print(start_time, stop_time)

start_time = 30
stop_time = 120
K_loss = 0.02

# for event, event_name in zip(events, events_name):
#     ret, calmarRatio = backtest(event=event, event_name=event_name, K_loss=K_loss, start_time=start_time, stop_time=stop_time)
#     print(event_name, ret, calmarRatio)

for event, event_name in zip(events, events_name):
    equity = backtest(event=event, event_name=event_name, K_loss=K_loss, start_time=start_time, stop_time=stop_time)
    equity.to_csv(f'final_event_result/{event_name}.csv')

