import pandas as pd
from pathlib import Path
import concurrent.futures
import os, sys
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


input_dir = Path("hk_stocks_smoothed")

start = datetime.strptime("2014-01-01", "%Y-%m-%d")
end = datetime.strptime("2018-12-31", "%Y-%m-%d")

frames = []

def load_data(name):
    try:
        df = pd.read_csv(name, parse_dates=["trade_date"])
    except Exception as exc:
        print(f"Couldn't load data in {name}!")
        return pd.DataFrame()
    else:
        df.trade_date = pd.to_datetime(df.trade_date)
        df.set_index("trade_date", inplace=True)
        df.sort_index(inplace=True)
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


smoothed = {}

def process(df):
    if df.index.min() <= start and df.index.max() >= end:
        code = df.ts_code.unique()[0]
        s = df.loc[(df.index >= start) & (df.index <= end), "smoothed_close"]
        smoothed[code] = s

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    future_to_url = {}
    for i in range(len(frames)):
        future_to_url[executor.submit(process, frames[i])] = i

    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")


if not frames:
    print(f"No data frames!")
    sys.exit(1)


df = pd.DataFrame(smoothed)
df.fillna(method="ffill", inplace=True)
df = np.log(df).diff().fillna(0)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df.values
X = scaler.fit_transform(X)

pca = PCA(n_components=0.75)
y = pca.fit_transform(X)
