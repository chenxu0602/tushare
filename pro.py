import tushare as ts
import pandas as pd
from pathlib import Path
import concurrent.futures
import os, sys
import time
from datetime import date, datetime

output_dir = Path("hk_stocks_{}".format(date.today().strftime("%Y-%m-%d")))
if not output_dir.exists():
    print(f"Output dir {output_dir} doesn't exist, creating one ...")
    os.makedirs(output_dir)

try:
    df_listed = pd.read_csv("listed.csv")
except Exception as e:
    print(f"Couldn't find the listed tickers!")
    sys.exit(1)
else:
    print(f"{len(df_listed)} listed tickers.")

try:
    df_delisted = pd.read_csv("delisted.csv")
except Exception as e:
    print(f"Couldn't find the delisted tickers!")
    sys.exit(1)
else:
    print(f"{len(df_delisted)} listed tickers.")

listed = set(sorted(df_listed.ts_code.unique()))
delisted = set(sorted(df_delisted.ts_code.unique()))

token = '065d9c4ba0cbd6d584d6f6d804ccbc946ef0827bf084ed50329ee5ea'
pro = ts.pro_api(token)

def download(ticker, output):
    print(f"Downloading {ticker} ...")
    df = pro.hk_daily(ts_code=ticker)
    if not df.empty:
        df.to_csv(output, index=False)
    return df


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_url = {}
    for tk in listed:
        output = output_dir / tk
        if output.exists():
            print(f"{output} exists, skip ...")
            continue
        future_to_url[executor.submit(download, tk, output)] = tk
        time.sleep(1)

    for tk in delisted:
        if tk in listed: 
            print(f"{tk} delisted is in listed, skip ...")
            continue
        output = output_dir / tk
        if output.exists():
            print(f"{output} exists, skip ...")
            continue
        future_to_url[executor.submit(download, tk, output)] = tk
        time.sleep(1)

    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")
        else:
            print(f"{url} has {len(data)} lines.")

