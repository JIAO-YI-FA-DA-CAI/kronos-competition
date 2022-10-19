from multiprocessing import Process
from Team.zhou import mlfinlab as ml
import pandas as pd
import os

transform_func_dict = {
    'tick': ml.data_structures.get_tick_bars,
    'volume': ml.data_structures.get_volume_bars,
    'dollar': ml.data_structures.get_dollar_bars
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
    df.to_csv("./data/" + args[3] + ".csv", index=False)


if __name__ == '__main__':

    names = ["BTC"]
    start_date = "2022-10-01"
    end_date = "2022-10-03"
    date_range = pd.date_range(start_date, end_date, freq="D")
    parent_dir = "./data/"
    for name in names:
        ticker = name + "USDT"
        path = os.path.join(parent_dir, ticker)

        process_list = []
        for date in date_range:
            print(ticker + "-trades-" + date.date().isoformat() + ".csv")
            df = read_ticks(os.path.join(path, ticker + "-trades-" + date.date().isoformat() + ".csv"))
            df_base = df[['timestamp', 'price', 'size', 'maker_flag']]

            print(f"size quantile: \n"
                  f"0.95:{df_base['size'].quantile(0.95)}\n"
                  f"0.75:{df_base['size'].quantile(0.75)}\n"
                  f"0.5:{df_base['size'].quantile(0.5)}\n"
                    f"0.25:{df_base['size'].quantile(0.25)}\n"
                    f"0.05:{df_base['size'].quantile(0.05)}\n")
            print(f"cost quantile: \n"
                  f"0.95:{df['cost'].quantile(0.95)}\n"
                    f"0.75:{df['cost'].quantile(0.75)}\n"
                    f"0.5:{df['cost'].quantile(0.5)}\n"
                    f"0.25:{df['cost'].quantile(0.25)}\n"
                    f"0.05:{df['cost'].quantile(0.05)}\n")

            p_tickbar = Process(target=transform_data,  args=("tick", df_base, 1000, ticker + "-tick-" + date.date().isoformat() + ".csv"))
            p_volumebar = Process(target=transform_data, args=("volume", df_base, df_base['size'].quantile(0.75), ticker + "-volume-" + date.date().isoformat() + ".csv"))
            p_dollarbar = Process(target=transform_data, args=("dollar", df_base, df['cost'].quantile(0.75), ticker + "-dollar-" + date.date().isoformat() + ".csv"))
            
            process_list.append(p_tickbar)
            process_list.append(p_volumebar)
            process_list.append(p_dollarbar)

            p_tickbar.start()
            p_volumebar.start()
            p_dollarbar.start()

            p_tickbar.join()
            p_volumebar.join()
            p_dollarbar.join()


        # df_volumebar.to_csv("./" + date.date().isoformat() + "-volumebar.csv", index=False)
        # df_dollarbar.to_csv("./" + date.date().isoformat() +"-dollarbar.csv", index=False)
        # df_tickbar.to_csv("./" + date.date().isoformat() + "-tickbar.csv", index=False)





