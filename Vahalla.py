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
    stock_data_frame = pd.read_csv('/Users/royschor/Desktop/Core Course/archive/aapl.us.txt')
    # we only take in the Date and Close values from the CSV
    stock_data_frame = stock_data_frame[['Date', 'Close']]
    # Converts all Dates of type string to type Datetime
    stock_data_frame['Date'] = stock_data_frame['Date'].apply(convert_to_date)
    # need to remove the first column (index column) as its useless data
    stock_data_frame.index = stock_data_frame.pop('Date')

    windowed_df = split_df_to_windowed_df(stock_data_frame, window_size=4)
    windowed_df = windowed_df.reset_index()

    # Now need to fix data to be numpy and reshape it to fit in LSTM
    df_npied = windowed_df.to_numpy()
    # grab only all the dates, first column is dates
    dates = df_npied[:, 0]
    # takes all past data points exluding date and target date data, so first and last
    middle_data_segment = df_npied[:, 1:-1]
    # reshaped by length of dates, the size of the middle part, and 1 for us as this is a univariate problem
    total_x_data = middle_data_segment.reshape((len(dates), middle_data_segment.shape[1], 1))
    total_y_data = df_npied[:, -1]
    # Unknown why I have to do this but found it fixed a bug
    total_x_data = total_x_data.astype(np.float32)
    total_y_data = total_y_data.astype(np.float32)
    # dates.shape = (8360), x.shape = (8360,4,1) (4 steps in past) (1 float variable), y.shape = (8360)
    print(dates.shape, total_x_data.shape, total_y_data.shape)
    
    # Now we create training, testing, and validation data
    # We will do 80% training, the remaining 20% is split 10-10 into validation and testing
    eighty_split = int(len(dates) * .8)
    ninety_split = int(len(dates) * .9)

    # Up until 80%
    dates_train, x_train, y_train = dates[:eighty_split], total_x_data[:eighty_split], total_y_data[:eighty_split]
    # between 80% - 90%
    dates_validation, x_validation, y_validation = dates[eighty_split:ninety_split], total_x_data[eighty_split:ninety_split], total_y_data[eighty_split:ninety_split]
    # 90% - end
    dates_test, x_test, y_test = dates[ninety_split:], total_x_data[ninety_split:], total_y_data[ninety_split:]

    # Creating the Network:
    

    # To see graphs uncomment the following:
    # show_total_stock_graph(stock_data_frame)
    show_data_split_graph(dates_train, y_train, dates_validation, y_validation, dates_test, y_test)


def show_data_split_graph(training_dates, y_train, validation_dates, y_validation, testing_dates, y_test):
  plt.title("Total Data Set Split into Training(80%), Validation(10%), Test(10%)")
  plt.plot(training_dates, y_train, color='red')
  plt.plot(validation_dates, y_validation, color='blue')
  plt.plot(testing_dates, y_test, color='green')
  plt.xlabel("Dates")

  plt.ylabel("Stock Values in Points")
  plt.legend(['Train', 'Validation', 'Test'])
  plt.show()

# Outputs the graph of the stock value
def show_total_stock_graph(stock_data_frame):
  plt.title("AAPL Stock Value")
  plt.plot(stock_data_frame.index, stock_data_frame['Close'])
  plt.xlabel("Dates")
  plt.ylabel("Stock Values in Points")
  plt.show()

# This function splits the dataframe into windows to feed into the network
# Each row now includes a date, followed by previous 4 days, and final column is prediction day's data
def split_df_to_windowed_df(data, window_size=4):
  # Creates our temporary datafram
  windowed_df = pd.DataFrame()

  for index in range(window_size, 0, -1):
    # appends to each row in df past 4 days of data, one day at a time and shifts the entire data doing so
    windowed_df[f'Target-{index}'] = data['Close'].shift(index)

  windowed_df['Target'] = data['Close']
  # Removes all rows that are missing some data - any incomplete windows that would mess up training
  return windowed_df.dropna()

# Each date in the dataframe is a string but we want it as a Date object
# thus this function converts a string to Datetime object
def convert_to_date(stringDate):
  # splits based on hyphen as the string of date is seperated by hyphen
   split_value = stringDate.split('-')
   year, month, day = int(split_value[0]), int(split_value[1]), int(split_value[2])
   return datetime.datetime(year=year, month=month, day=day)

if __name__ == "__main__":
  main()