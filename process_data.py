import pandas as pd
from pathlib import Path
import concurrent.futures
import os, sys
import matplotlib.pyplot as plt


input_dir = Path("hk_stocks")

frames = []

def load_data(name):
    try:
        df = pd.read_csv(name, parse_dates=["trade_date"])
    except Exception as exc:
        print(f"Couldn't load data in {name}!")
        return pd.DataFrame()
    else:
        frames.append(df)
        return df

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    future_to_url = {}
    for name in input_dir.glob("*.HK"):
        future_to_url[executor.submit(load_data, name)] = name

    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")
        else:
            print(f"{url} has {len(data)} lines.")

if not frames:
    print(f"No data frames!")
    sys.exit(1)


smoothed = []

def smooth(df):
    df.set_index("trade_date", inplace=True)
    df2 = pd.DataFrame({"smoothed_close": 
        df.close.ewm(span=10, adjust=False).mean()})
    df = df.merge(df2, left_index=True, right_index=True, how="left")
    df.reset_index(inplace=True)
    smoothed.append(df)
    return df

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    future_to_url = {}
    for i in range(len(frames)):
        future_to_url[executor.submit(smooth, frames[i])] = i

    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")
        else:
            print(f"{url} has {len(data)} lines.")

output_dir = Path("hk_stocks_smoothed")
if not output_dir.exists():
    print(f"Creating output dir: {output_dir}")
    output_dir.mkdir()

def save(df):
    code = df.ts_code.unique()
    if not code.size == 1:
        print(f"Wrong ts code!")
        return pd.DataFrame()
    output = output_dir / code[0]
    print(f"Saving to {output}.")
    df.to_csv(output, index=False)
    return df

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    future_to_url = {}
    for i in range(len(smoothed)):
        future_to_url[executor.submit(save, smoothed[i])] = i

    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")
