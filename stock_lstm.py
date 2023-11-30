import yfinance as yf

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM

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

# %% LSTM
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
history = model.fit(X, y, epochs=250, use_multiprocessing=True, shuffle=False)
# %% LSTM - Summary and Visualization

# noinspection PyUnboundLocalVariable
model.summary()
# noinspection PyUnboundLocalVariable
loss = history.history['loss']
mae = history.history['mean_absolute_percentage_error']

metrics = [loss, mae]

plt.close()
plt.figure(figsize=(12, 12))
count = 1
for i in ['MSLE', 'MAPE']:
    plt.subplot(2, 1, count)
    plt.plot(metrics[count - 1], label=f'min:{np.min(metrics[count - 1]):.4f}\nmax:{np.max(metrics[count - 1]):.4f}')
    plt.legend()
    plt.title(f'Epoch vs {i}')
    plt.xlabel('Epoch')
    plt.ylabel(i)
    count += 1

plt.tight_layout()
plt.show()

# %% Test - Predictions LSTM and its Visualization
test_val = y.copy()
y_pred = model.predict(test_val)
y_pred = np.array(y_pred).reshape((-1))
print(y_pred)

plt.figure(figsize=(10, 10))
plt.title('LSTM - Actual vs Predictions')
plt.plot(np.array(y).reshape((-1)), label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
