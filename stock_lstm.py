import keras.optimizers
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM

import joblib

# %%

stock_name = 'GOOGL'
google_ticker = yf.Ticker(stock_name)
data = google_ticker.history(period='max')

data = data.iloc[::, :4]

# %%
plt.figure(figsize=(20, 20))
count = 1
for i in data.columns:
    plt.subplot(len(data.columns), 1, count)
    plt.plot(data.index, data[i])
    plt.title(f"Time vs {i}")
    plt.xlabel("Time Period")
    plt.ylabel(i)
    count += 1

plt.tight_layout()
plt.show()

# %%

X, y = data['Open'].to_numpy().reshape((-1, 1)), data['Close'].to_numpy().reshape((-1, 1))

print(X.shape, y.shape)

# %%

force_train = True
file_name = f"D:/PyCharm_Projects/StockMarket/lstm_{stock_name}.model"

if (not os.path.isfile(file_name)) or force_train:
    model = Sequential([
        CuDNNLSTM(units=100, return_sequences=True, input_shape=(X.shape[0], 1), name='LSTM-1'),
        Dropout(0.15, name='DROPOUT-1'),

        CuDNNLSTM(units=1000, return_sequences=True, name='LSTM-2'),
        Dropout(0.15, name='DROPOUT-2'),

        CuDNNLSTM(units=1000, return_sequences=True, name='LSTM-3'),
        Dropout(0.15, name='DROPOUT-3'),

        CuDNNLSTM(units=100, return_sequences=True, name='LSTM-4'),
        Dropout(0.15, name='DROPOUT-4'),

        Dense(units=1, name='DENSE-1')
    ], name='SEQUENTIAL-1')

    model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.mean_squared_logarithmic_error,
                  metrics=[tf.metrics.mean_absolute_percentage_error])
    history = model.fit(X, y, epochs=150, use_multiprocessing=True, shuffle=False)

    joblib.dump(model, filename=file_name)
elif os.path.isfile(file_name) and (not force_train):
    model = joblib.load(file_name)
    history = model.history

# %%

# noinspection PyUnboundLocalVariable
model.summary()
# noinspection PyUnboundLocalVariable
loss = history.history['loss']
mae = history.history['mean_absolute_percentage_error']

metrics = [loss, mae]

plt.close()
plt.figure(figsize=(12, 12))
count = 1
for i in ['Loss', 'MAPE']:
    plt.subplot(2, 1, count)
    plt.plot(metrics[count - 1], label=f'min:{np.min(metrics[count - 1]):.4f}\nmax:{np.max(metrics[count - 1]):.4f}')
    plt.legend()
    plt.title(f'Epoch vs {i}')
    plt.xlabel('Epoch')
    plt.ylabel(i)
    count += 1

plt.tight_layout()
plt.show()

# %%


test_val = 100
val = np.array(test_val).reshape((-1, 1))
y_pred = model.predict(val)
print(test_val, y_pred)

