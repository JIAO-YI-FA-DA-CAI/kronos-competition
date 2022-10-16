from binance.client import Client
import datetime
import pandas as pd

api_key = "emptyKey"
api_secret = "emptySecret"
client = Client(api_key, api_secret)

### This code is for UTC +0
# https://python-binance.readthedocs.io/en/latest/

def listToDataframe(dataUnprocess):
    data_df = pd.DataFrame(dataUnprocess)
    data_df = data_df.drop(columns=[i for i in range(6, 12)])
    data_df.columns = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    data_df['Datetime'] = data_df['TimeStamp'].apply(lambda x: datetime.datetime.fromtimestamp(x//1000).strftime("%Y-%m-%d %H:%M:%S"))
    data_df['Datetime'] = pd.to_datetime(data_df['Datetime']) - datetime.timedelta(hours=8)
    data_df['Date'] = data_df['Datetime'].apply(lambda x: str(x)[:10])
    data_df['Time'] = data_df['Datetime'].apply(lambda x: str(x)[11:])
    data_df['Time'] = data_df['Time'].apply(lambda x: str(x)[:5])
    data_df = data_df.set_index('Datetime')
    data_df = data_df.drop(columns=['TimeStamp', 'Date', 'Time'])
    data_df = data_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data_df    
    
def getData(symbol="LTCBTC", interval=Client.KLINE_INTERVAL_1MINUTE, startTime="1 day ago UTC"):
    all_df = pd.DataFrame()
    nowTime = datetime.date.today()
    #nowTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    while True:
        endTime = pd.to_datetime(startTime)+datetime.timedelta(days=1)
        endTime = endTime.strftime("%Y-%m-%d")
        # fetch klines
        klines = client.get_historical_klines(symbol, interval, startTime, endTime)    
        if klines:
            temp_data_df = listToDataframe(klines)
            all_df = pd.concat([all_df, temp_data_df])
            # all_df = all_df.append(temp_data_df)
       
        if pd.to_datetime(startTime) < pd.to_datetime(nowTime):
            startTime = endTime
            print(startTime)
        else:
            print("end:", startTime)
            break    
    all_df = all_df.sort_index()
    all_df = all_df.drop_duplicates()
    all_df = all_df.dropna()
    
    return all_df

symbol = 'BTCUSDT'
startTime = '2021-02-28 0:0:0'
filename = 'BTC_spotPrice.csv'

df = getData(symbol=symbol, startTime=startTime)

df1 = df.loc[:'2021-6']
df2 = df.loc['2021-7':'2021-12']
df3 = df.loc['2022-1':'2022-6']
df4 = df.loc['2022-7':]


df1.to_csv('BTC_spotPrice_2021_1.csv')
df2.to_csv('BTC_spotPrice_2021_2.csv')
df3.to_csv('BTC_spotPrice_2022_1.csv')
df4.to_csv('BTC_spotPrice_2022_2.csv')
