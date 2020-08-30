import pandas as pd
from pathlib import Path
import concurrent.futures
import os, sys
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


input_dir = Path("hk_stocks_smoothed")

train_start = datetime.strptime("2016-01-01", "%Y-%m-%d")
train_end = datetime.strptime("2018-01-01", "%Y-%m-%d")

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

def process(df, start, end):
    if df.index.min() <= start and df.index.max() >= end:
        code = df.ts_code.unique()[0]
        s = df.loc[(df.index >= start) & (df.index <= end), "smoothed_close"]
        smoothed[code] = s

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    future_to_url = {}
    for i in range(len(frames)):
        future_to_url[executor.submit(
          process, frames[i], train_start, train_end)] = i

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
pca.fit(X)

y = pca.transform(X)
df_train = pd.DataFrame(data=y, index=df.index, columns=[f"f{i+1}" for i in range(y.shape[1])]) 

def load_history(ticker, start, end):
    df = load_data(input_dir / ticker)
    s = df.loc[(df.index >= start) & (df.index <= end), "smoothed_close"]
    long_smoothed[ticker] = s

long_smoothed = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    future_to_url = {}
    for tk in df.columns:
        future_to_url[executor.submit(
          load_history, tk, "2012-01-01", train_end)] = i

    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")

df_long = pd.DataFrame(long_smoothed)
df_long.fillna(method="ffill", inplace=True)
df_long = np.log(df_long).diff().fillna(0)

y_predict = pca.transform(scaler.transform(df_long.values))
df_predict = pd.DataFrame(data=y_predict, index=df_long.index, columns=[f"f{i+1}" for i in range(y_predict.shape[1])]) 
df_predict.index.name = "date"

output = Path("pca_factors.csv")
df_predict.round(4).to_csv(output, index=True)