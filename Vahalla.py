import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import requests as rq
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


def main():
    """
    start_date = '2014-01-01'
    end_date = '2016-01-01'
    stockDataFrame = pd.read_csv('/Users/royschor/Desktop/Core Course/archive/aapl.us.txt', usecols=['Date', 'Close'], dtype={'Date': 'str', 'Close': 'float'}, parse_dates=['Date'], index_col='Date')
    # Reads in all the data, then slices it to only take the data after the start date
    stockDataFrame = stockDataFrame.loc[start_date:end_date]
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

    # This takes the first 80% of the normalized data to be trained set and 
    # the last 20% to be the test
    train_size = int(len(normalized_prices) * 0.8)
    train_data, test_data = normalized_prices[:train_size], normalized_prices[train_size:]

    # Splits the total data in half (x & y)
    # Takes x and splits it into train and test, first half is train second half is test
    # Does the same with y, y is the second half of the data while x is first
    x_data = normalized_prices[:int(len(normalized_prices)/2)]
    y_data = normalized_prices[int(len(normalized_prices)/2):]
    x_train = x_data[:int(len(x_data)/2)]
    x_test = x_data[int(len(x_data)/2):]
    y_train = y_data[:int(len(x_data)/2)]
    y_test = y_data[int(len(x_data)/2):]

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    #model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(train_data[:-1], train_data[1:], epochs=3, verbose=2)
    
    # model.fit(x_train, y_train, epochs=3, verbose=2)

    test_loss = model.evaluate(test_data[:-1], test_data[1:], verbose=0)
    # test_loss = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_loss)
    test_predictions = model.predict(test_data[:-1])

    plt.plot(test_data[1:], label='Test Data')
    plt.plot(test_predictions, label='Test Predictions')
    plt.plot(train_data, label='Training')
    # plt.ylim(float(int(minClosePrice - 5.0)), float(int(maxClosePrice + 5.0)))
    plt.legend()
    plt.show()
    """

#############################################################################################
############################### Try Number 5 or so!! ########################################
#############################################################################################
    start_date_1 = '2013-12-31'
    end_date_1 = '2016-01-01'
    stockDataFrame_1 = pd.read_csv('/Users/royschor/Desktop/Core Course/archive/aapl.us.txt', usecols=['Date', 'Close'], dtype={'Date': 'str', 'Close': 'float'}, parse_dates=['Date'], index_col='Date')
    # Reads in all the data, then slices it to only take the data after the start date
    stockDataFrame_1 = stockDataFrame_1.loc[start_date_1:end_date_1]
    minClosePrice_1 = stockDataFrame_1['Close'].min()
    maxClosePrice_1 = stockDataFrame_1['Close'].max()

    # we want to split the data into different window sizes (4 for train, then 1 for test)
    # how do we do it?
    split_window_data = stockDataFrame_1['Close'].values.reshape(-1, 5)
    print(stockDataFrame_1['Close'].values.reshape(-1, 5))

if __name__ == "__main__":
  main()


def on_epcoh_end(epoch, _):
   print("Print something after each epoch to see where we at")