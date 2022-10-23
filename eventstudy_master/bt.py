import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import talib

def backtest(event=pd.DataFrame(), event_name='event_2022_7_27', _print=False, _plot=False, length=15, NumStd=1.5, K_profit=0.06, K_loss=0.03):
    # 'Open', 'High', 'Low', 'Close', 'Volume', 
    # 'twap', 'buy_volume','buy_twap', 'sell_volume', 'sell_twap',
    # 'buy_sell_ratio', 'cum_Volume','cum_buy_volume', 'cum_sell_volume', 'cum_buy_sell_ratio'
    fund = 10000
    money = 10000
    feeRate = 0
    # event = event_2022_7_27

    # 參數
    # length = 15
    # NumStd = 1.5
    # K_profit = 0.06
    # K_loss = 0.03

    event['MA'] = event['twap'].rolling(window=length, center=False).mean()
    event['STD'] = event['twap'].rolling(window=length, center=False).std()
    event['upLine'] = event['MA'] + NumStd*event['STD']
    event['downLine'] = event['MA'] - NumStd*event['STD']

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

    df_arr = np.array(event)
    time_arr = np.array(event.index)

    for i in range(len(df_arr)):

        if i == len(df_arr)-1:
            break
            
        ## 進場邏輯
        entryLong = df_arr[i,16] > 0 and df_arr[i,17]>df_arr[i,15]# (ma 15 微分大於0) 且 (ma5>ma15)
        entrySellShort = df_arr[i,16] < 0 and df_arr[i,17]<df_arr[i,15]
        entryCondition = True
        
        ## 出場邏輯
        exitShort = df_arr[i,16] < 0 and df_arr[i,17]<df_arr[i,15]
        exitBuyToCover = df_arr[i,16] > 0 and df_arr[i,17]>df_arr[i,15]

        ## 停利停損邏輯
        if BS == 'B':
            pass
            # stopProfit = df_arr[i,9] >= df_arr[i,7] * (1+K_profit)
            # stopLossMoving = df_arr[i,9] <= high * (1-K_loss)
            # stopLoss = df_arr[i,9] <= df_arr[t,7] * (1-K_loss)
            
        elif BS == 'S':
            pass
            # stopProfit = df_arr[i,7] <= df_arr[i,9] * (1-K_profit)
            # stopLossMoving = df_arr[i,7] >= low * (1+K_loss)
            # stopLoss = df_arr[i,7] >= df_arr[t,9] * (1+K_loss)
            
        if BS == None:
            profit_list.append(0)
            profit_fee_list.append(0)
            
            if entryLong and entryCondition:
                tempSize = money / df_arr[i+1,7]
                BS = 'B'
                t = i+1
                buy.append(t)
                t1 = time_arr[i+1]
                high = df_arr[i,0]

            elif entrySellShort and entryCondition:
                tempSize = money / df_arr[i+1,9]
                BS = 'S'
                t = i+1
                sellshort.append(t)
                t1 = time_arr[i+1]
                low = df_arr[i,0]
                
        elif BS == 'B':
            profit = tempSize * (df_arr[i+1,0] - df_arr[i,0])
            profit_list.append(profit)
            t2 = time_arr[i+1]
            
            if df_arr[i,0] > high:
                high = df_arr[i,0]

            if exitShort or i == len(df_arr)-2 or stopLossMoving or stopProfit:
            # A = False
            # if A:
                pl_round = tempSize * (df_arr[i+1,9] - df_arr[t,7])
                profit_fee = profit - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                BS=None
                high = -10**8

                # Realized PnL
                profit_fee_realized = pl_round - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list_realized.append(profit_fee_realized)
                
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                t1 = time_arr[i+1]
                
        elif BS == 'S':
            profit = tempSize * (df_arr[i,0] - df_arr[i+1,0])
            profit_list.append(profit)
            t2 = time_arr[i+1]

            if df_arr[i,0] < low:
                low = df_arr[i,0]
            
            if exitBuyToCover or i == len(df_arr)-2 or stopLossMoving or stopProfit:
            # A = False
            # if A:
                pl_round = tempSize * (df_arr[t,9] - df_arr[i+1,7])
                profit_fee = profit - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list.append(profit_fee)
                buytocover.append(i+1)
                BS=None
                low = 10**8

                # Realized PnL
                profit_fee_realized = pl_round - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list_realized.append(profit_fee_realized)
    
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                t1 = time_arr[i+1]            
    print("ok")
    print("profit_fee_list")
    equity = pd.DataFrame({'profit':np.cumsum(profit_fee_list)}, index=event.index)
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

    if len(profit_fee_list_realized) != 0:
        winRate = len([i for i in profit_fee_list_realized if i > 0]) / len(profit_fee_list_realized)
    else:
        winRate = np.nan

    if abs(np.mean([i for i in profit_fee_list_realized if i < 0])) != 0:
        winLossRatio = np.mean([i for i in profit_fee_list_realized if i > 0]) / abs(np.mean([i for i in profit_fee_list_realized if i < 0]))
    else:
        winLossRatio = np.nan

    if _print == True: 
        print(f'<{event_name}>')
        print(f'return: {np.round(ret,4)*100}%')
        print(f'mdd: {np.round(mdd,4)*100}%')
        print(f'calmarRatio: {np.round(calmarRatio,2)}')
        print(f'tradeTimes: {tradeTimes}')
        print(f'winRate: {np.round(winRate,4)*100}%')
        print(f'winLossRatio: {np.round(winLossRatio,2)}')
        print()

    if _plot == True:
        equity.plot(figsize=(12,6), ylabel='PnL')
        event['Close'].plot(alpha=0.7, grid=True, secondary_y=True)
        plt.title(f'Equity Curve {event_name}')
        plt.ylabel('BTC Spot Price')
        plt.show();

    return equity



def backtest(event=pd.DataFrame(), event_name='event_2022_7_27', _print=False, _plot=False, length=15, NumStd=1.5, K_profit=0.06, K_loss=0.03):
    # 'Open', 'High', 'Low', 'Close', 'Volume', 
    # 'twap', 'buy_volume','buy_twap', 'sell_volume', 'sell_twap',
    # 'buy_sell_ratio', 'cum_Volume','cum_buy_volume', 'cum_sell_volume', 'cum_buy_sell_ratio'
    fund = 10000
    money = 10000
    feeRate = 0
    # event = event_2022_7_27

    # 參數
    # length = 15
    # NumStd = 1.5
    # K_profit = 0.06
    # K_loss = 0.03

    event['MA'] = event['twap'].rolling(window=length, center=False).mean()
    event['STD'] = event['twap'].rolling(window=length, center=False).std()
    event['upLine'] = event['MA'] + NumStd*event['STD']
    event['downLine'] = event['MA'] - NumStd*event['STD']

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

    df_arr = np.array(event)
    time_arr = np.array(event.index)

    for i in range(len(df_arr)):

        if i == len(df_arr)-1:
            break
            
        ## 進場邏輯
        entryLong = df_arr[i,16] > 0 and df_arr[i,17]>df_arr[i,15]# (ma 15 微分大於0) 且 (ma5>ma15)
        entrySellShort = df_arr[i,16] < 0 and df_arr[i,17]<df_arr[i,15]
        entryCondition = True
        
        ## 出場邏輯
        exitShort = df_arr[i,16] < 0 and df_arr[i,17]<df_arr[i,15]
        exitBuyToCover = df_arr[i,16] > 0 and df_arr[i,17]>df_arr[i,15]

        ## 停利停損邏輯
        if BS == 'B':
            pass
            # stopProfit = df_arr[i,9] >= df_arr[i,7] * (1+K_profit)
            # stopLossMoving = df_arr[i,9] <= high * (1-K_loss)
            # stopLoss = df_arr[i,9] <= df_arr[t,7] * (1-K_loss)
            
        elif BS == 'S':
            pass
            # stopProfit = df_arr[i,7] <= df_arr[i,9] * (1-K_profit)
            # stopLossMoving = df_arr[i,7] >= low * (1+K_loss)
            # stopLoss = df_arr[i,7] >= df_arr[t,9] * (1+K_loss)
            
        if BS == None:
            profit_list.append(0)
            profit_fee_list.append(0)
            
            if entryLong and entryCondition:
                tempSize = money / df_arr[i+1,7]
                BS = 'B'
                t = i+1
                buy.append(t)
                t1 = time_arr[i+1]
                high = df_arr[i,0]

            elif entrySellShort and entryCondition:
                tempSize = money / df_arr[i+1,9]
                BS = 'S'
                t = i+1
                sellshort.append(t)
                t1 = time_arr[i+1]
                low = df_arr[i,0]
                
        elif BS == 'B':
            profit = tempSize * (df_arr[i+1,0] - df_arr[i,0])
            profit_list.append(profit)
            t2 = time_arr[i+1]
            
            if df_arr[i,0] > high:
                high = df_arr[i,0]

            if exitShort or i == len(df_arr)-2 or stopLossMoving or stopProfit:
            # A = False
            # if A:
                pl_round = tempSize * (df_arr[i+1,9] - df_arr[t,7])
                profit_fee = profit - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                BS=None
                high = -10**8

                # Realized PnL
                profit_fee_realized = pl_round - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list_realized.append(profit_fee_realized)
                
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                t1 = time_arr[i+1]
                
        elif BS == 'S':
            profit = tempSize * (df_arr[i,0] - df_arr[i+1,0])
            profit_list.append(profit)
            t2 = time_arr[i+1]

            if df_arr[i,0] < low:
                low = df_arr[i,0]
            
            if exitBuyToCover or i == len(df_arr)-2 or stopLossMoving or stopProfit:
            # A = False
            # if A:
                pl_round = tempSize * (df_arr[t,9] - df_arr[i+1,7])
                profit_fee = profit - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list.append(profit_fee)
                buytocover.append(i+1)
                BS=None
                low = 10**8

                # Realized PnL
                profit_fee_realized = pl_round - money*feeRate - (money+pl_round)*feeRate
                profit_fee_list_realized.append(profit_fee_realized)
    
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                t1 = time_arr[i+1]            
    print("ok")
    print("profit_fee_list")
    equity = pd.DataFrame({'profit':np.cumsum(profit_fee_list)}, index=event.index)
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

    if len(profit_fee_list_realized) != 0:
        winRate = len([i for i in profit_fee_list_realized if i > 0]) / len(profit_fee_list_realized)
    else:
        winRate = np.nan

    if abs(np.mean([i for i in profit_fee_list_realized if i < 0])) != 0:
        winLossRatio = np.mean([i for i in profit_fee_list_realized if i > 0]) / abs(np.mean([i for i in profit_fee_list_realized if i < 0]))
    else:
        winLossRatio = np.nan

    if _print == True: 
        print(f'<{event_name}>')
        print(f'return: {np.round(ret,4)*100}%')
        print(f'mdd: {np.round(mdd,4)*100}%')
        print(f'calmarRatio: {np.round(calmarRatio,2)}')
        print(f'tradeTimes: {tradeTimes}')
        print(f'winRate: {np.round(winRate,4)*100}%')
        print(f'winLossRatio: {np.round(winLossRatio,2)}')
        print()

    if _plot == True:
        equity.plot(figsize=(12,6), ylabel='PnL')
        event['Close'].plot(alpha=0.7, grid=True, secondary_y=True)
        plt.title(f'Equity Curve {event_name}')
        plt.ylabel('BTC Spot Price')
        plt.show();

    return equity


