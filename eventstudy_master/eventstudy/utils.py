import numpy as np
import pandas as pd
from scipy.stats import t

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from collections import defaultdict
from csv import DictReader

# All model must returns : (residuals: list, df: int, var: float)

# TODO: sortir la computation des résiduals des fonctions de modélisation. Juste leur faire calculer les prédictions. Sortir aussi windowsize estimation size et tout le reste, et aussi le secReturns qui doit être rataché à l'event study pas la fonction de modélisation.

def to_table(columns, asterisks_dict=None, decimals=None, index_start=0):

    if decimals:
        if type(decimals) is int:
            decimals = [decimals] * len(columns)

        for key, decimal in zip(columns.keys(), decimals):
            if decimal:
                columns[key] = np.round(columns[key], decimal)

    if asterisks_dict:
        columns[asterisks_dict["where"]] = map(
            add_asterisks, columns[asterisks_dict["pvalue"]], columns[asterisks_dict["where"]]
        )

    df = pd.DataFrame.from_dict(columns)
    df.index += index_start
    return df


def add_asterisks(pvalue, value=None):
    if value == None:
        value = pvalue

    if pvalue < 0.01:
        asterisks = f"{str(value)} ***"
    elif pvalue < 0.05:
        asterisks = f"{str(value)} **"
    elif pvalue < 0.1:
        asterisks = f"{str(value)} *"
    else:
        asterisks = f"{str(value)}"
    return asterisks


def plot(time, CAR, *, AR=None, CI=False, var=None, df=None, confidence=0.90):

    fig, ax = plt.subplots()
    ax.plot(time, CAR)
    ax.axvline(
        x=0, color="black", linewidth=0.5,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if CI:
        delta = np.sqrt(var) * t.ppf(confidence, df)
        upper = CAR + delta
        lower = CAR - delta
        ax.fill_between(time, lower, upper, color="black", alpha=0.1)

    if AR is not None:
        ax.vlines(time, ymin=0, ymax=AR)

    if ax.get_ylim()[0] * ax.get_ylim()[1] < 0:
        # if the y axis contains 0
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    return fig


def get_index_of_date(data, date: np.datetime64, n: int = 4):
    # assume the date exist and there is only one of it in the dataset
    # assume date are in index

    for i in range(n + 1):
        index = np.where(data == date)[0]
        if len(index) > 0:
            return index[0]
        else:
            date = date + np.timedelta64(1, "D")

    # return None if there is no row corresponding to this date or n days after.
    return None


def OLD_read_csv(path):
    data = defaultdict(list)
    with open("./returns.csv", "r") as f:
        reader = DictReader(f)
        for row in reader:
            for col, value in row.items():
                data[col].append(value)

    return data


def read_csv(
    path,
    format_date: bool = False,
    date_format: str = "%Y-%m-%d",
    date_column: str = "date",
    row_wise: bool = False,
):

    df = pd.read_csv(path, skipinitialspace=True)

    if format_date:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)

    if row_wise:
        data = list()
        for row in df.itertuples(index=False):
            row = dict(row._asdict())
            if format_date:
                row[date_column] = np.datetime64(row[date_column])
            data.append(row)
    else:  # column_wise
        data = dict()
        for col in df.columns:
            data[col] = df[col].values

    return data


#=========================================
def economy_excel_timestamp(
    df1,df_btc,
    date_type='number',utc=8):
    df1.columns=['big_time','small_time','data','predict','past']
    fix=pd.Timedelta(2,'d')
    if(date_type=='number'):
        df1['date']=(pd.to_datetime('1900-01-01')+df1['big_time'].map(dt.timedelta)-fix).astype('str')+" "+df1['small_time'].astype('str')
    if(date_type=='chinense'):
        df_big_time=df1['big_time'].str.extract("(.*)年(.*)月(.*)日")
        df_big_time.columns=['年','月','日']
        df1['date']=df_big_time['年']+"-"+df_big_time['月']+"-"+df_big_time['日']+" "+df1['small_time'].astype('str')
    if(date_type=='fed_press_conference'):
        col=['big_time', 'small_time', 'h1', 'm1', 's1', 'h2', 'm2', 's2']
        temp=df1.loc[1:,:].copy().fillna(0)
        temp.columns=col
        temp['date']=(pd.to_datetime('1900-01-01')+temp['big_time'].map(dt.timedelta)-fix).astype('str')+" "+temp['small_time'].astype('str')
        temp['date']=pd.to_datetime(temp['date'])
        h1=temp['h1'].map(lambda x:pd.Timedelta(x,'H'),na_action="ignore")
        m1=temp['m1'].map(lambda x:pd.Timedelta(x,'m'),na_action="ignore")
        s1=temp['s1'].map(lambda x:pd.Timedelta(x,'s'),na_action="ignore")
        h2=temp['h2'].map(lambda x:pd.Timedelta(x,'H'),na_action="ignore")
        m2=temp['m2'].map(lambda x:pd.Timedelta(x,'m'),na_action="ignore")
        s2=temp['s2'].map(lambda x:pd.Timedelta(x,'s'),na_action="ignore")
        temp['date1']=(temp['date']+h1+m1+s1).astype('str')
        temp['date2']=(temp['date']+h2+m2+s2).astype('str')
        temp['date']=(temp['date']).astype('str')
        df1=temp
    df1['date']=pd.to_datetime(df1['date'])-pd.Timedelta(utc,'h')
    df1=df1[['date','data','predict','past']]
    df1=df1[df1['date']>df_btc.index[0]]
    df1=df1[df1['date']<df_btc.index[-1]]
    return df1

def fed_excel_timestamp(
    df1,df_btc,
    date_type='fed',utc=8):
    fix=pd.Timedelta(2,'d')
    if(date_type=='fed_press_conference'):
        col=['big_time', 'small_time', 'h1', 'm1', 's1', 'h2', 'm2', 's2']
        temp=df1.loc[1:,:].copy().fillna(0)
        temp.columns=col
        temp['date']=(pd.to_datetime('1900-01-01')+temp['big_time'].map(dt.timedelta)-fix).astype('str')+" "+temp['small_time'].astype('str')
        temp['date']=pd.to_datetime(temp['date'])
        h1=temp['h1'].map(lambda x:pd.Timedelta(x,'H'),na_action="ignore")
        m1=temp['m1'].map(lambda x:pd.Timedelta(x,'m'),na_action="ignore")
        s1=temp['s1'].map(lambda x:pd.Timedelta(x,'s'),na_action="ignore")
        h2=temp['h2'].map(lambda x:pd.Timedelta(x,'H'),na_action="ignore")
        m2=temp['m2'].map(lambda x:pd.Timedelta(x,'m'),na_action="ignore")
        s2=temp['s2'].map(lambda x:pd.Timedelta(x,'s'),na_action="ignore")
        temp['date1']=(temp['date']+h1+m1+s1)
        temp['date2']=(temp['date']+h2+m2+s2)
        df1=temp[['date','date1','date2']]
        df1=df1.applymap(lambda x:x-pd.Timedelta(utc,'h'))
    else:
        df1.columns=['big_time','small_time']
        df1['date']=(pd.to_datetime('1900-01-01')+df1['big_time'].map(dt.timedelta)-fix).astype('str')+" "+df1['small_time'].astype('str')
        df1['date']=pd.to_datetime(df1['date'])-pd.Timedelta(utc,'h')
        df1=df1[['date']]
    df1=df1[df1['date']>df_btc.index[0]]
    df1=df1[df1['date']<df_btc.index[-1]]
    return df1 
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import talib
from scipy import stats
from sklearn.linear_model import LinearRegression
def ttest(front_df,batchsize=100):
    epochs  = int(len(front_df) / batchsize)

    # front_tt,back_tt=es.ttest(front_df),es.ttest(back_df)
    ttest = pd.DataFrame()
    for epoch in range(epochs): 
        sample = front_df.iloc[batchsize*epoch:batchsize*(epoch+1)]['AR'].values
        mean=front_df.iloc[batchsize*epoch:batchsize*(epoch+1)]['Expected_Return'].mean()
        t, p_value = stats.ttest_1samp(sample, mean)
        if p_value <= 0.01:
            significance = '***'
        elif 0.01 < p_value <= 0.05:
            significance = '**'
        elif 0.05 < p_value <= 0.1:
            significance = '*'
        else:
            significance = ''  
        temp=pd.DataFrame(np.array([epoch,t,p_value, significance]).reshape((1,4)))
        ttest = pd.concat([ttest,temp]).reset_index(drop=True)
    ttest.columns=['epoch','T-score', 'P-value','Sig.']
    ttest = ttest.set_index('epoch')
    ttest[['T-score', 'P-value']]=ttest[['T-score', 'P-value']].astype("float").applymap(lambda x: '%.4f' % x)
    return ttest
# event study hist
def es_hist(front_df,back_df):
    front_df['Close'].plot(title='before event', figsize=(8,4))
    plt.show()

    back_df['Close'].plot(title='after event', figsize=(8,4))
    plt.show()

    front_df['Return'].plot(kind='hist', bins=30, alpha = 0.8, color='black', figsize=(8, 4), legend=True, label='frontEvent')
    back_df['Return'].plot(kind='hist', bins=30, alpha = 0.6, color='orange', figsize=(8, 4), legend=True, label='backEvent', title='Return Histogram')
    plt.show()

    front_df['AR'].plot( color='black', figsize=(8, 4), legend=True, label='frontEvent_AR')
    back_df['AR'].plot(color='red', figsize=(8, 4), legend=True, label='backEvent_AR', title='AR')
    plt.show()
    front_df['CAR'].plot( color='black', figsize=(8, 4), legend=True, label='frontEvent_CAR', title='CAR')
    back_df['CAR'].plot(color='red', figsize=(8, 4), legend=True, label='backEvent_CAR')
    plt.show()

# event study fama french
def es_ARs(train_df,test_df):
    model = LinearRegression()
    model.fit(train_df['Return_ma'].values.reshape(-1,1),train_df['Return'].values.reshape(-1,1))
    test_df['Expected_Return'] = model.predict(test_df['Return_ma'].values.reshape(-1,1))
    test_df['AR']=test_df['Return_ma']-test_df['Expected_Return']
    test_df['CAR']=test_df['AR'].cumsum()
    return test_df

# event study stat
def es_stat(front_df,back_df,
            frontLength,backLength):
    # front stat
    mean1 = front_df['Return'].mean()
    std1 = front_df['Return'].std()
    skew1 = front_df['Return'].skew()
    kurt1 = front_df['Return'].kurt()
    rsi1 = talib.RSI(front_df['Close'], frontLength-1).iloc[-1]
    adx1 = talib.ADX(front_df['High'], front_df['Low'], front_df['Close'], frontLength/2-1).iloc[-1]
    atr1 = talib.ATR(front_df['High'], front_df['Low'], front_df['Close'], frontLength-1).iloc[-1]

    # back stat
    mean2 = back_df['Return'].mean()
    std2 = back_df['Return'].std()
    skew2 = back_df['Return'].skew()
    kurt2 = back_df['Return'].kurt()
    rsi2 = talib.RSI(back_df['Close'], backLength-1).iloc[-1]
    adx2 = talib.ADX(back_df['High'], back_df['Low'], back_df['Close'], backLength/2-1).iloc[-1]
    atr2 = talib.ATR(back_df['High'], back_df['Low'], back_df['Close'], backLength-1).iloc[-1]
    
    before={
        "mean":mean1,
        "std":std1,
        "skew":skew1,        
        "kurt":kurt1,
        "rsi":rsi1,
        "atr":atr1,
    }
    after={
        "mean":mean2,
        "std":std2,
        "skew":skew2,        
        "kurt":kurt2,
        "rsi":rsi2,
        "atr":atr2,
    }
    return before,after

# event study date
def es_date(df=pd.DataFrame(),
            eventTime='2022-5-4 19:17', 
            trainMinute=2880,
            frontMinute=2880, backMinute=2880, 
            kbar=5,
            betalength=30):

    if not(df.empty):
        frontLength  = int(frontMinute / kbar)
        backLength = int(backMinute / kbar)

        d1 = df.resample(rule=str(kbar)+'T', closed='left', label='left').first()[['Open']]
        d2 = df.resample(rule=str(kbar)+'T', closed='left', label='left').max()[['High']]
        d3 = df.resample(rule=str(kbar)+'T', closed='left', label='left').min()[['Low']]
        d4 = df.resample(rule=str(kbar)+'T', closed='left', label='left').last()[['Close']]
        d5 = df.resample(rule=str(kbar)+'T', closed='left', label='left').sum()[['Volume']]
        df_ = pd.concat([d1,d2,d3,d4,d5], axis=1)

        df_['Return'] = df['Close'].pct_change().fillna(0)
        df_=df_.dropna()
        df_['Return_ma']=talib.MA(df_['Return'], timeperiod=betalength).shift(1)        
        df_=df_.reset_index()
        df_=df_.reset_index().rename(columns={"index":"Relative_day"})
        df_=df_.set_index('Datetime')
        idx=df_.loc[eventTime,'Relative_day']
        df_['Relative_day']-=idx
        try:
            train_df = df_.loc[:eventTime].iloc[-(trainMinute+frontLength):-frontLength]
            front_df = df_.loc[:eventTime].iloc[-frontLength:]
            back_df = df_.loc[eventTime:].iloc[:backLength]    

            return train_df,front_df,back_df,frontLength,backLength
        except:
            return "event date has no enough train kindles"
    else:
        return "no df pass"