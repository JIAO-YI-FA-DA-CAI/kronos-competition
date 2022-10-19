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