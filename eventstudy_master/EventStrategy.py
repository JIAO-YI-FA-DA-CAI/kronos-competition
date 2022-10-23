import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import talib
from scipy import stats
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('../') # setting path

import os
import sys
import numpy as np
import datetime
import event_study.event_study_sdk as es
from event_study.event_study_sdk import excelExporter
# ==================================================================
import os 
os.chdir('c:\\Users\\casper\\Desktop\\Casper\\kronos competition\\kronos-competition\\get_data')
df_btc1 = pd.read_csv('BTC_spotPrice_2021_1.csv', index_col='Datetime', parse_dates=True)
df_btc2 = pd.read_csv('BTC_spotPrice_2021_2.csv', index_col='Datetime', parse_dates=True)
df_btc3 = pd.read_csv('BTC_spotPrice_2022_1.csv', index_col='Datetime', parse_dates=True)
df_btc4 = pd.read_csv('BTC_spotPrice_2022_2.csv', index_col='Datetime', parse_dates=True)

df = pd.concat([df_btc1,df_btc2,df_btc3,df_btc4])
df.index = df.index+dt.timedelta(minutes=1)

df['return']=df['Close'].pct_change()
df['return_ema']=talib.EMA(df['return'], timeperiod=30)
df=df.dropna(how='any')
df.index.name='date'
df.to_csv("BTC_Return.csv")

#===========================================================
os.chdir('c:\\Users\\casper\\Desktop\\Casper\\kronos competition\\kronos-competition\\event_study')
import event_study.event_study_sdk.utils as esu
events=["CoreCPI","CPI","CorePPI","PPI","Nonfarm","InitialClaim"]
df1 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[0], parse_dates=True)
df2 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[1], parse_dates=True)
df3 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[2], parse_dates=True)
df4 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[3], parse_dates=True)
df5 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[4], parse_dates=True)
df6 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[5], parse_dates=True)
def than_expected(df1):
    df1['type']=0
    df1['type+']=df1['type'].mask(df1['data']>df1['predict'],1)
    df1['type-']=df1['type'].mask(df1['data']<df1['predict'],-1)
    df1['type']=df1['type+']+df1['type-']
    df1['end_date']=(df1['date']+dt.timedelta(days=2))
    df1['start_date']=(df1['date']-dt.timedelta(days=2))
    return df1
df1=than_expected(esu.economy_excel_timestamp(df1,df,date_type='chinense'))
df2=than_expected(esu.economy_excel_timestamp(df2,df,date_type='chinense'))
df3=than_expected(esu.economy_excel_timestamp(df3,df,date_type='chinense'))
df4=than_expected(esu.economy_excel_timestamp(df4,df,date_type='chinense'))
df5=than_expected(esu.economy_excel_timestamp(df5,df,date_type='chinense'))
df6=than_expected(esu.economy_excel_timestamp(df6,df,date_type='number'))

events=["會議紀要","經濟預測","聲明","官員談話","新聞發布會"]
df7 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[0], parse_dates=True)
df8 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[1], parse_dates=True)
df9 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[2], parse_dates=True)
df10 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[3], parse_dates=True)
df11 = pd.read_excel('ReleaseTime.xlsx',sheet_name=events[4], parse_dates=True)
df7=esu.fed_excel_timestamp(df7,df,date_type='fed')
df8=esu.fed_excel_timestamp(df8,df,date_type='fed')
df9=esu.fed_excel_timestamp(df9,df,date_type='fed')
df10=esu.fed_excel_timestamp(df10,df,date_type='fed')
df11=esu.fed_excel_timestamp(df11,df,date_type='fed_press_conference')
df12=df11[['date1']].rename(columns={'date1':'date'})
df12['date']=pd.to_datetime(df12['date'].dt.strftime('%Y-%m-%d %H:%M'))
df13=df11[['date2']].rename(columns={'date2':'date'})
df13['date']=pd.to_datetime(df13['date'].dt.strftime('%Y-%m-%d %H:%M'))

#===========================================================
def to_kline(symbol1='BTCUSDT',symbol2='BTCUSDT',date='2022-01-25',no_action=False):
    os.chdir('c:\\Users\\casper\\Desktop\\Casper\\kronos competition\\kronos-competition\\get_data\\data')
    filename=f'.\{symbol1}\{symbol2}-trades-{date}.csv'
    tick = pd.read_csv(filename, header=None)
    if(no_action):
        return tick
    else:
        tick.columns = ['ID', 'price', 'size', 'size_quote', 'trade_time', 'is_buyer_mm', 'ignore']
        tick['time'] = [dt.datetime.fromtimestamp(i/1000) for i in tick['trade_time']]
        tick = tick[['price', 'size', 'size_quote', 'time', 'is_buyer_mm']]
        tick.index = tick['time']

        tick_sell = tick[tick['is_buyer_mm']==True].drop(['is_buyer_mm', 'time'], axis=1)
        tick_buy = tick[tick['is_buyer_mm']==False].drop(['is_buyer_mm', 'time'], axis=1)

        twap = tick['size_quote'].resample(rule='1T').sum() / tick['size'].resample(rule='1T').sum()
        buy_volume = tick_buy['size'].resample(rule='1T').sum()
        buy_twap = tick_buy['size_quote'].resample(rule='1T').sum() / tick_buy['size'].resample(rule='1T').sum()
        sell_volume = tick_sell['size'].resample(rule='1T').sum()
        sell_twap = tick_sell['size_quote'].resample(rule='1T').sum() / tick_sell['size'].resample(rule='1T').sum()

        kline = pd.concat([twap, buy_volume, buy_twap, sell_volume, sell_twap], axis=1)
        kline.columns = ['twap', 'buy_volume', 'buy_twap', 'sell_volume', 'sell_twap']
        
        # to utc+0
        kline.index = [i-dt.timedelta(hours=8) for i in kline.index]
        kline.index=kline.index+dt.timedelta(minutes=1)
        return kline

events=["CoreCPI","CPI","CorePPI","PPI","Nonfarm","InitialClaim"]
# for i,df_event in enumerate([df1,df2,df3,df4,df5,df6]):
for i,df_event in enumerate([df6]):
    i=i+5
    print(events[i])
    corr_dict_all=pd.DataFrame()

    events_all=[]
    for event in df_event.index:
        dates=pd.DataFrame()
        start_date=df_event['start_date'].loc[event,]
        end_date=df_event['end_date'].loc[event,]
        es_date=df_event['date'].loc[event,]
        print(f"{start_date} ~ {es_date} ~ {end_date}",end=' ')
        for date in pd.date_range(start=start_date,end=end_date, freq='D'):
            temp = to_kline(symbol1='BTCUSDT',
                            symbol2='BTCUSDT',
                            date=date.strftime('%Y-%m-%d'))
            dates=pd.concat([dates,temp])
        dates=dates.loc[start_date:end_date]
        dates['type']=df_event['type'].loc[event,]
        dates=dates.merge(df, how='left',left_index=True,right_index=True)

        mins=1440*2
        time_windows=np.arange(30,mins,30)
        corr_dict={}
        for time_window in time_windows:
            # print(f"{time_window}",end='=>')
            corr=(pd.DataFrame(dates.loc[:es_date,'Close'][::-1].iloc[:time_window].values,dates.loc[es_date:,'Close'].iloc[:time_window].values).reset_index().corr()).iloc[0,1]
            corr_dict[time_window]=corr
        # print(f"near event {time_window:4} mins corr: {corr:5.2f}")
        corr_dict=pd.Series(corr_dict)
        corr_dict_all=pd.concat([corr_dict_all,corr_dict],axis=1)
        events_all.append(df_event['date'].iloc[event])
    corr_dict_all.columns=events_all
    corr_dict_all['mean']=abs(corr_dict_all).mean(axis=1)
    corr_dict_all.to_csv(f"{events[i]}_correlation.csv")