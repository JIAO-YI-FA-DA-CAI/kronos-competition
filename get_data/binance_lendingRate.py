import requests
import time
import datetime
import pytz
import hmac
import hashlib
import pandas as pd

tz = pytz.timezone('UTC')

class BinanceSpotClient:
    _EndPoint = 'https://api.binance.com'
    
    def __init__(self, api_key, api_secret):
        self._api_key = api_key
        self._api_secret = api_secret
        self.header = {'X-MBX-APIKEY': self._api_key}

    def _addSign(self, param, recvWindow=60000):
        timestamp = int((datetime.datetime.now(tz) - datetime.datetime.utcfromtimestamp(0).replace(tzinfo=tz)).total_seconds() * 1000)
        param['timestamp'] = timestamp
        param['recvWindow'] = recvWindow
        hashString = ''
        for key in param.keys():
            if param[key]:
                if type(param[key]) == list:
                    for p in param[key]:
                        hashString += key + '=' + str(p) + '&'
                else:
                    hashString += key + '=' + str(param[key]) + '&'
        hashString = hashString[:-1]
        signature = hmac.new(bytes(self._api_secret , 'latin-1'),
                             msg = bytes(hashString , 'latin-1'),
                             digestmod = hashlib.sha256).hexdigest()
        param['signature'] = signature

        return param

    def _process_response(self, response):
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        if type(data) == dict:
            if 'code' in data.keys():
                raise Exception(str(data))
            else:
                return data
        else:
            return data
        
    def _get(self, path, params):
        r = requests.get(self._EndPoint+path, headers=self.header, params=self._addSign(params))

        return self._process_response(response=r)

    def strToTimestamp(self, dt_str):
        if dt_str:
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            timestamp = time.mktime(dt.timetuple()) * 1000
            return int(timestamp)
        else:
            return None


def get_interestRateHistory(SpotClient, symbol, vipLevel, startTime, endTime):

    try:
        r = SpotClient._get(path = '/sapi/v1/margin/interestRateHistory', 
                params = {
                    'asset':symbol, 
                    'vipLevel': vipLevel,
                    'startTime': startTime,
                    'endTime': endTime,
                    'recvWindow': 60000
                    })
    except Exception as e:
        print('somethings wrong!', e)
        print('sleeping for 2s... will retry')
        time.sleep(2)
        get_interestRateHistory(SpotClient, symbol, vipLevel)

    return r

# enter your key
api_key = ''
api_secret = ''

SpotClient = BinanceSpotClient(api_key = api_key, api_secret = api_secret)

now_timestamp = SpotClient.strToTimestamp(str(datetime.datetime.now()).split('.')[0])
vip_list = [None, 1, 2, 3, 4, 5, 6, 7, 8 ,9]
data_df = pd.DataFrame()
symbol = 'USDT'

once_data_df = pd.DataFrame()

for vip_num in vip_list:

    startTime = SpotClient.strToTimestamp(str(datetime.datetime.now() - datetime.timedelta(days=28)).split('.')[0])
    result = get_interestRateHistory(SpotClient, symbol=symbol, vipLevel=vip_num, startTime=startTime, endTime=now_timestamp)
    temp_data_df = pd.DataFrame(result)
    temp_data_df.index = temp_data_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x//1000).strftime("%Y-%m-%d %H:%M:%S"))
    vipNum = temp_data_df.iloc[0]['vipLevel']
    temp_data_df[f'vip{vipNum}'] = temp_data_df['dailyInterestRate']
    temp_data_df = temp_data_df.drop(['asset', 'timestamp', 'dailyInterestRate', 'vipLevel'], axis=1)
    once_data_df = pd.concat([once_data_df, temp_data_df], axis=1)

data_df = pd.concat([data_df, once_data_df])

while pd.to_datetime(data_df.index[-1]) >= pd.to_datetime('2021-2-28'):

    endTime = SpotClient.strToTimestamp(data_df.index[-1])
    startTime = datetime.datetime.fromtimestamp(endTime/1000) - datetime.timedelta(days=28)
    startTime = SpotClient.strToTimestamp(str(startTime).split('.')[0])
    once_data_df = pd.DataFrame()

    for vip_num in vip_list:
        result = get_interestRateHistory(SpotClient, symbol=symbol, vipLevel=vip_num, startTime=startTime, endTime=endTime)
        temp_data_df = pd.DataFrame(result)
        temp_data_df.index = temp_data_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x//1000).strftime("%Y-%m-%d %H:%M:%S"))
        vipNum = temp_data_df.iloc[0]['vipLevel']
        temp_data_df[f'vip{vipNum}'] = temp_data_df['dailyInterestRate']
        temp_data_df = temp_data_df.drop(['asset', 'timestamp', 'dailyInterestRate', 'vipLevel'], axis=1)
        once_data_df = pd.concat([once_data_df, temp_data_df], axis=1)

    data_df = pd.concat([data_df, once_data_df])

data_df = data_df[~data_df.index.duplicated(keep='first')]
data_df.to_csv(f'{symbol}_dailyInterestRate.csv')
