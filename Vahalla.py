import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import datetime
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
# from tensorflow.keras.optimizers import RMSprop


def main():
    """
    # Start and end date of the data we will reading in
    start_date_1 = '2013-12-31'
    end_date_1 = '2016-01-01'
    # File we read in, we only read the Date and Close values from the CSV
    stockDataFrame = pd.read_csv('/Users/royschor/Desktop/Core Course/archive/aapl.us.txt', usecols=['Date', 'Close'], dtype={'Date': 'str', 'Close': 'float'}, parse_dates=['Date'], index_col='Date')
    # Reads in all the data, then slices it to only take the data after the start date
    stockDataFrame = stockDataFrame.loc[start_date_1:end_date_1]

    #### Uncomment this for a graph of the stock value itself
    # minClosePrice = stockDataFrame['Close'].min()
    # maxClosePrice = stockDataFrame['Close'].max()
    # plt.plot(stockDataFrame.index, stockDataFrame['Close'])
    # plt.title("AAPL Stock Value from 2014 to 2016")
    # plt.xlabel("Dates (Year-Month)")
    # plt.ylabel("Stock Values in Points")
    # plt.ylim(float(int(minClosePrice - 5.0)), float(int(maxClosePrice + 5.0)))
    # plt.show()

    # This normalizes the Closing prices to be btwn 0-1, based off of the max price
    stockDataFrame['Close'] = stockDataFrame['Close'].apply(lambda x: x/stockDataFrame['Close'].max())

    just_close_vals = stockDataFrame['Close'].values
    raw_training_data = []
    raw_x_train_data = []
    raw_y_train_data = []
    eighty_percent = int(len(just_close_vals) * 0.8)

    # This is for the training data which is for the first 80% of the data
    for index in range(5, eighty_percent, 5):
      x_train_vals = just_close_vals[index - 5:index - 1]
      y_train_vals = just_close_vals[index - 1]

      raw_training_data.append(x_train_vals)
      # Need to add the three 0's, because when we np.array it we do matric multiplication 
      # and we need the dimension to be equal for the multiplication to work
      raw_training_data.append([y_train_vals, 0, 0, 0])
      raw_x_train_data.append(x_train_vals)
      raw_y_train_data.append([y_train_vals])

    # This is for the last 20% of the data which is the testing data
    raw_test_data = []
    raw_x_test_data = []
    raw_y_test_data = []
    for index in range(eighty_percent, len(just_close_vals), 5):
      x_test_vals = just_close_vals[index - 5:index - 1]
      y_test_vals = just_close_vals[index - 1]

      raw_test_data.append(x_test_vals)
      raw_test_data.append([y_test_vals, 0, 0, 0])
      raw_x_test_data.append(x_test_vals)
      raw_y_test_data.append([y_test_vals])

    # Need to reshape the data using np arrays
    training_data = np.array(raw_training_data)
    x_train = np.array(raw_x_train_data)
    y_train = np.array(raw_y_train_data)
    test_data = np.array(raw_test_data)
    x_test = np.array(raw_x_test_data)
    y_test = np.array(raw_y_test_data)
    
    # LSTM Paramater layout:
    # model.add(keras.layers.LSTM(hidden_nodes, input_shape=(window, num_features), consume_less="mem"))

    model = Sequential()
    # Shape is 4 --> 1 as we need 4 input neurons (1 for each close value) outputing 1 neuron (the fifth close value)
    model.add(LSTM(64, input_shape=(4, 1)))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=3, verbose=2, validation_data=(x_test, y_test))

    # model.add(Dropout(0.2)) --> Unsure if needed, dropout is where random neurons are ignored in training temporarily

    # test_loss = model.evaluate(y_train, y_test, verbose=0)
    # print('Test loss:', test_loss)

    # test_predictions = model.predict(y_test)
    # print(test_predictions)

    plt.title("Network Results")
    plt.plot(y_test, label='y Test Data')
    # plt.plot(test_predictions, label='Network Test Predictions')
    # plt.plot(test_data, label='Test Data')
    # plt.plot(training_data, label='Training')
    plt.legend()
    plt.show()

    # I think maybe x_train needs to be split in half to be the x and y and then the validation/ test remains the remaining 20%

    """
    # File we read in
    stockDataFrame = pd.read_csv('/Users/royschor/Desktop/Core Course/archive/aapl.us.txt')
    # we only take in the Date and Close values from the CSV
    stockDataFrame = stockDataFrame[['Date', 'Close']]
    # Converts all Dates of type string to type Datetime
    stockDataFrame['Date'] = stockDataFrame['Date'].apply(convertToDate)
    # need to remove the first column (index column) as its useless data/ space
    stockDataFrame.index = stockDataFrame.pop('Date')
    
    # Outputs the graph of the stock value
    # plt.title("AAPL Stock Value")
    # plt.plot(stockDataFrame.index, stockDataFrame['Close'])
    # plt.xlabel("Dates")
    # plt.ylabel("Stock Values in Points")
    # plt.show()

    startDate = stockDataFrame.index[0]
    endDate = stockDataFrame.index[-1]
    # print(stockDataFrame)

    windowDf = window_data(stockDataFrame, windowSize=4)
    # print(windowDf)

# This function splits the dataframe into windows to feed into the network
# Each row now includes a date, the previous 4 days of data, and the final column is the target prediction day's data
def window_data(data, windowSize=4):
  # Creates our temporary datafram
  windowed_data = pd.DataFrame()

  for index in range(windowSize, 0, -1):
    # appends to each row in df past 4 days of data, one day at a time and shifts the entire data doing so
    windowed_data[f'Target-{index}'] = data['Close'].shift(index)

  windowed_data['Target'] = data['Close']

  # Removes all rows that are missing some data - any incomplete windows that would mess up training
  return windowed_data.dropna()

# Each date in the dataframe is a string but we want it as a Date object
# thus this function converts a string to Datetime object
def convertToDate(stringDate):
  # splits based on hyphen as the string of date is seperated by hyphen
   splitValue = stringDate.split('-')
   year, month, day = int(splitValue[0]), int(splitValue[1]), int(splitValue[2])
   return datetime.datetime(year=year, month=month, day=day)

if __name__ == "__main__":
  main()