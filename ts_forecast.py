import numpy as np
import pandas as pd
from pathlib import Path
import os, sys
import matplotlib.pyplot as plt
import tensorflow as tf


inputfile = Path("pca_factors.csv")
df = pd.read_csv(inputfile, parse_dates=["date"], index_col=["date"])

factor = "f4"

split_time = 1200
df_train = df.iloc[:split_time]
df_valid = df.iloc[split_time:]
x_train = df_train[factor].values 
x_valid = df_valid[factor].values 

def windowed_dataset_dense(series, window_size, batch_size, shuffle_buffer):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

window_size = 128
batch_size = 16
shuffle_buffer_size = 1000

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
train_set_dense = windowed_dataset_dense(x_train, window_size, batch_size, shuffle_buffer_size)
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

scale = tf.Variable(100.)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * scale)
])

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(20, input_shape=[window_size], activation="relu"),
#     tf.keras.layers.Dense(10, activation="relu"),
#     tf.keras.layers.Dense(1),
# ])

def last_time_step_mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer, 
              metrics=["mae", last_time_step_mse])
history = model.fit(train_set, epochs=30, callbacks=[tensorboard_callback])