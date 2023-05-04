import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LambdaCallback

def main():
    # File we read in
    stock_data_frame = pd.read_csv('AAPL.csv')
    # we only take in the Date and Close values from the CSV
    stock_data_frame = stock_data_frame[['Date', 'Close']]
    # Converts all Dates of type string to type Datetime
    stock_data_frame['Date'] = stock_data_frame['Date'].apply(convert_to_date)
    # need to remove the first column (index column) as its useless data
    stock_data_frame.index = stock_data_frame.pop('Date')

    windowed_df = split_df_to_windowed_df(stock_data_frame, window_size=4)

    # Here we slice the windowed dataframe greatly
    # We believe that the network training all the data actually harms its predictions 
    # as it is not training on the most volatile part (the recent history), 
    # thus we are now trying to only train on recent history (past 3 years not all 30+)
    start_date = "2018-05-04"
    end_date = "2023-05-03"
    windowed_df = windowed_df.loc[start_date:end_date]
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
    # dates.shape = (8360), total_x_data.shape = (8360,4,1) (4 steps in past) (1 float variable), total_y_data.shape = (8360)

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

    # Creating the Network 4,1 for 4 inputs dates one output
    model = Sequential()
    model.add(layers.LSTM(64, input_shape=(4,1)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Call back function to get the loss and val_mean_absolute_error to graph it and see our accuracy
    # history = []
    # history.appendloss_history()
    # history_callback = []
    # # history_callback.append(LambdaCallback(on_epoch_end=on_epoch_end))
    # my_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    # history = Histories()

    #define history so that I can plot the loss of this model
    #define history_callback so that I can plot the loss of this model
    # history = Histories()
    # what is the error here?
    # history_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    # history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20, callbacks=[history_callback])


    # model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20, callbacks=[history])
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20)
    training_predictions = model.predict(x_train).flatten()
    validation_predictions = model.predict(x_validation).flatten()
    test_predictions = model.predict(x_test).flatten()

    # To see graphs uncomment the following:
    # show_total_stock_graph(stock_data_frame)
    show_data_split_graph(dates_train, y_train, dates_validation, y_validation, dates_test, y_test)
    network_training_prediction_graph(dates_train, training_predictions, y_train)
    network_validation_prediction_graph(dates_validation, validation_predictions, y_validation)
    network_testing_prediction_graph(dates_test, test_predictions, y_test)
    all_predictions_graph(dates_train, dates_validation, dates_test, training_predictions, validation_predictions, test_predictions, y_train, y_validation, y_test)
    plot_loss(history)

    # stock_data_frame = stock_data_frame.loc[start_date:]
    # plt.title("Sliced AAPL Stock Value Timeline")
    # plt.plot(stock_data_frame.index, stock_data_frame['Close'])
    # plt.xlabel("Dates")
    # plt.ylabel("Stock Values in Points")
    # plt.show()

    # Two graphs one of loss over time and one of val mean absolute error over time


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


def all_predictions_graph(training_dates, validation_dates, testing_dates, training_predictions, validation_predictions, test_predictions, y_train, y_val, y_test):
  plt.title("Training, Validation, and Testing Predictions vs Real Observation")
  plt.plot(training_dates, training_predictions)
  plt.plot(training_dates, y_train)
  plt.plot(validation_dates, validation_predictions)
  plt.plot(validation_dates, y_val)
  plt.plot(testing_dates, test_predictions)
  plt.plot(testing_dates, y_test)
  plt.legend(['Training Predictions', 
              'Training Observations',
              'Validation Predictions',
              'Validation Observations',
              'Testing Predictions',
              'Testing Observations'])
  plt.xlabel("Dates")
  plt.ylabel("Stock Value in Points")
  plt.show()

def network_testing_prediction_graph(testing_dates, test_predictions, y_test):
  plt.title("Testing Predictions vs Observations of Network")
  plt.plot(testing_dates, test_predictions)
  plt.plot(testing_dates, y_test)
  plt.xlabel("Dates")
  plt.ylabel("Stock Value in Points")
  plt.legend(['Testing Predictions', 'Testing Observations'])
  plt.show()

def network_validation_prediction_graph(validation_dates, validation_predictions, y_val):
  plt.title("Validation Predictions vs Observations of Network")
  plt.plot(validation_dates, validation_predictions)
  plt.plot(validation_dates, y_val)
  plt.xlabel("Dates")
  plt.ylabel("Stock Value in Points")
  plt.legend(['Validation Predictions', 'Validation Observations'])
  plt.show()

def network_training_prediction_graph(training_dates, training_predictions, y_train):
  plt.title("Training Predictions vs Observations of Network")
  plt.plot(training_dates, training_predictions)
  plt.plot(training_dates, y_train)
  plt.xlabel("Dates")
  plt.ylabel("Stock Value in Points")
  plt.legend(['Training Predictions', 'Training Observations'])
  plt.show()

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

def plot_loss(history):
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()

if __name__ == "__main__":
  main()