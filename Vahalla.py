import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random
import sys
import io
import os
import datetime
from bs4 import BeautifulSoup
import requests as rq
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras


def main():
    start_date = '2016-01-01'
    stockDataFrame = pd.read_csv('/Users/royschor/Desktop/Core Course/archive/aapl.us.txt', usecols=['Date', 'Close'], dtype={'Date': 'str', 'Close': 'float'}, parse_dates=['Date'], index_col='Date')
    # Reads in all the data, then slices it to only take the data after the start date (Janurary 1st of 2016 and onward)
    stockDataFrame = stockDataFrame.loc[start_date:]
    minClosePrice = stockDataFrame['Close'].min()
    maxClosePrice = stockDataFrame['Close'].max()

    # stockDataFrame['Close'].plot()
    # plt.title("Closing Prices of AAPL Stock Over Time")
    # plt.xlabel("Stock Dates from 2016 onward")
    # plt.ylabel("Stock Value in Dollars")
    # plt.ylim(float(int(minClosePrice - 5.0)), float(int(maxClosePrice + 5.0)))
    # plt.show()

    closing_prices = stockDataFrame['Close'].values.reshape(-1, 1)
    # This normalizes the Closing prices to be btwn 0-1, based off of the max price
    normalized_prices = closing_prices / closing_prices.max()

    # This takes 80% of the normalized data to be trained set and 20% to be the test
    train_size = int(len(normalized_prices) * 0.8)
    train_data, test_data = normalized_prices[:train_size], normalized_prices[train_size:]

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(train_data[:-1], train_data[1:], epochs=10, batch_size=1, verbose=2)

    test_loss = model.evaluate(test_data[:-1], test_data[1:], verbose=0)
    print('Test loss:', test_loss)
    test_predictions = model.predict(test_data[:-1])

    # plt.plot(test_data[1:], label='Test Data')
    plt.plot(test_predictions, label='Predictions yeah right')
    plt.plot(train_data, label='Training')
    # plt.ylim(float(int(minClosePrice - 5.0)), float(int(maxClosePrice + 5.0)))
    plt.legend()
    plt.show()

if __name__ == "__main__":
  main()
