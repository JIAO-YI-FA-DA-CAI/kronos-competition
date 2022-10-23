from multiprocessing import Process, Pool
import mlfinlab as ml
import pandas as pd
import os

transform_func_dict = {
    'tick': ml.data_structures.get_tick_bars,
    'volume': ml.data_structures.get_volume_bars,
    'dollar': ml.data_structures.get_dollar_bars,
}

def read_ticks(fp):
    # read tick data from file
    cols = list(map(str.lower, ['ID', 'price', 'size', 'cost', 'timestamp', 'maker_flag', 'best_flag']))
    df = (pd.read_csv(fp, header=None)
          .rename(columns=dict(zip(range(len(cols)),cols)))
          .assign(dates=lambda df: (pd.to_datetime(df['timestamp'], unit='ms').dt.date))
          .assign(time=lambda df: (pd.to_datetime(df['timestamp'], unit='ms').dt.time))
          .set_index('dates')
          .drop_duplicates())
    return df


def transform_data(*args):
    print(f"store in {args[3]}")
    df = transform_func_dict[args[0]](args[1], args[2])
    df.to_csv("./data/BTCUSDT/" + args[0] + "/" + args[3] + ".csv", index=False)

def run(type):
    names = ["BTC"]
    start_date = "2021-07-21"
    end_date = "2021-11-21"
    date_range = pd.date_range(start_date, end_date, freq="D")
    parent_dir = "./data"
    for name in names:
        ticker = name + "USDT"
        path = os.path.join(parent_dir,  ticker + "/trades")

        for prev_date, date in zip(date_range, date_range[1:]):
            print(ticker + "-trades-" + date.date().isoformat() + ".csv")
            df_prev = read_ticks(os.path.join(path, ticker + "-trades-" + prev_date.date().isoformat() + ".csv"))
            df = read_ticks(os.path.join(path, ticker + "-trades-" + date.date().isoformat() + ".csv"))

            df_base = df[['timestamp', 'price', 'size', 'maker_flag']]
            switcher = {
                'tick': len(df_prev)/86400,
                'volume':  df_prev['size'].quantile(0.75),
                'dollar': df_prev['cost'].quantile(0.75),
            }
            transform_data(type, df_base, switcher[type], ticker + "-" + type + "-" + date.date().isoformat())

if __name__ == '__main__':
    pool = Pool(3)
    pool.map(run, ['tick', 'volume', 'dollar'])
    print("Finished")


    # p_tickbar = Process(target=transform_data,  args=("tick", df_base, 1000, ticker + "-tick-" + date.date().isoformat()))
    # p_volumebar = Process(target=transform_data, args=("volume", df_base, df_base['size'].quantile(0.75), ticker + "-volume-" + date.date().isoformat()))
    # p_dollarbar = Process(target=transform_data, args=("dollar", df_base, df['cost'].quantile(0.75), ticker + "-dollar-" + date.date().isoformat()))
    #
    # process_list.append(p_tickbar)
    # process_list.append(p_volumebar)
    # process_list.append(p_dollarbar)
    #
    # for p in process_list:
    #     p.start()
    # for p in process_list:
    #     p.join()







