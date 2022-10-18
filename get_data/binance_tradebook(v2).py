import pandas as pd
import zipfile
import urllib.request
import time
import os
names = ["BTC"]

start_date = "2022-10-01"
end_date = "2022-10-03"
date_range = pd.date_range(start_date, end_date, freq="D")

parent_dir = "./data/"
for name in names:
    print(name)
    ticker = name + "USDT"
    path = os.path.join(parent_dir, ticker)

    if not os.path.exists(path):
        os.mkdir(path)

    for date in date_range:
        base_url = "https://data.binance.vision/data/spot/daily/trades/"
        url = base_url + ticker + "/" + ticker + "-trades-" + date.date().isoformat() + ".zip"
        print(url)
        file_path = os.path.join(path, ticker + ".zip")
        file_name = os.path.join(path, ticker + "-1m-" + date.date().isoformat()[:-3] + ".csv")
        #
        if not os.path.exists(file_name):
            try:
                urllib.request.urlretrieve(url, file_path)
            except:
                print(date)
                continue
        #
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path)
            os.remove(file_path)

        time.sleep(0.1)