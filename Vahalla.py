import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
#hopefully this works

class Histories(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []
    self.val_mean_absolute_error = []
    self.validation_loss = []

  def on_epoch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
    self.validation_loss.append(logs.get('val_loss'))
    self.val_mean_absolute_error.append(logs.get('val_mean_absolute_error'))

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
    start_date = "2019-01-01"
    end_date = "2023-05-03"
    windowed_df = windowed_df.loc[start_date:end_date]
    windowed_df = windowed_df.reset_index()

    # Below are the arrays that hold the Validation Mean Absolute Error for the various ranges we tested. We hardcode these values because we do not care about retraining the model every time we run the program, we just want to see the results of the model.
    # 1983 
    a = [6.252941608428955, 5.861586570739746, 5.516464710235596, 5.3440937995910645, 5.521788597106934, 5.176183700561523, 4.966674327850342, 5.287469387054443, 4.994562149047852, 4.909974575042725, 5.080941200256348, 4.807373046875, 4.756270408630371, 4.746894359588623, 4.294131755828857, 4.266901016235352, 4.119685173034668, 4.308830261230469, 4.13102912902832, 4.020274639129639]
    # 1993
    b = [14.882230758666992, 13.587896347045898, 13.125176429748535, 12.55322551727295, 12.01820182800293, 11.671340942382812, 11.780874252319336, 11.617316246032715, 11.571721076965332, 11.347923278808594, 10.826908111572266, 11.392051696777344, 10.95500659942627, 11.047951698303223, 10.692322731018066, 11.418281555175781, 10.75320816040039, 10.654683113098145, 10.62875747680664, 10.770564079284668]
    # 2003
    c = [38.34563064575195, 33.267860412597656, 31.9074649810791, 31.13399887084961, 30.729549407958984, 30.33388900756836, 30.10266876220703, 29.960704803466797, 29.78331756591797, 29.794527053833008, 29.59204864501953, 29.446773529052734, 29.51336669921875, 29.488868713378906, 29.33375358581543, 29.283695220947266, 29.077476501464844, 28.97437286376953, 29.091236114501953, 28.73244285583496]
    # 2013
    d = [134.06332397460938, 86.33783721923828, 32.76543045043945, 23.52359390258789, 20.861404418945312, 19.634998321533203, 18.70405387878418, 17.354795455932617, 16.856077194213867, 16.478525161743164, 15.877416610717773, 14.524008750915527, 14.446290969848633, 18.378307342529297, 15.712822914123535, 14.90718936920166, 14.6417818069458, 14.269576072692871, 15.563369750976562, 15.755496978759766]
    # 2018
    e = [148.7624053955078, 132.35946655273438, 77.95355224609375, 58.55264663696289, 46.0085563659668, 11.688681602478027, 4.65613317489624, 3.8759472370147705, 3.311195135116577, 3.3155882358551025, 3.2633109092712402, 3.154031753540039, 3.1059470176696777, 3.2320895195007324, 2.920069932937622, 2.9358115196228027, 2.836531400680542, 3.193765640258789, 3.5151216983795166, 3.4412786960601807]
    # 2019
    f = [148.73809814453125, 139.95709228515625, 105.03919982910156, 56.121337890625, 41.18218994140625, 26.27713394165039, 7.036997318267822, 3.9553627967834473, 3.3355133533477783, 3.240297317504883, 3.2375853061676025, 3.011812686920166, 2.994367837905884, 2.8576622009277344, 2.87736439704895, 3.430616617202759, 2.769928455352783, 3.188600778579712, 3.4058034420013428, 2.673654079437256]
    # 2020
    g = [143.92137145996094, 141.15023803710938, 134.99044799804688, 117.07787322998047, 80.38827514648438, 31.0433349609375, 13.052395820617676, 15.66572380065918, 12.865097999572754, 4.965411186218262, 3.1070821285247803, 3.7383804321289062, 3.3583579063415527, 3.8592076301574707, 4.0676493644714355, 3.2291860580444336, 3.0178701877593994, 3.096320390701294, 3.3127856254577637, 2.9465932846069336]
    # 2021
    h = [138.84365844726562, 136.8251190185547, 133.09474182128906, 126.93390655517578, 116.0492935180664, 101.43693542480469, 78.79503631591797, 52.47658920288086, 20.023515701293945, 7.861454963684082, 12.547298431396484, 9.992996215820312, 8.203194618225098, 8.4424409866333, 6.566117286682129, 7.862133026123047, 6.146325588226318, 4.938750743865967, 4.094830513000488, 2.7769486904144287]
    # 2022
    i = [149.21546936035156, 148.54150390625, 147.6330108642578, 146.36386108398438, 144.5149383544922, 141.56141662597656, 137.49472045898438, 131.77981567382812, 124.0371322631836, 113.24458312988281, 99.18407440185547, 83.5406265258789, 64.0949478149414, 44.30375671386719, 25.45188331604004, 10.653507232666016, 2.708097219467163, 5.997953414916992, 5.055028915405273, 4.500945091247559]
    # 2023
    j = [164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875, 164.70623779296875]
    data = [a, b, c, d, e, f, g, h, i, j]

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
    history = Histories()

    fits = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20, callbacks=[history])
    training_predictions = model.predict(x_train).flatten()
    validation_predictions = model.predict(x_validation).flatten()
    test_predictions = model.predict(x_test).flatten()

    # print(history.val_mean_absolute_error)

    # To see graphs uncomment the following:

    # show_total_stock_graph(stock_data_frame)
    # show_sliced_stock_graph(stock_data_frame, start_date, end_date)
    # show_data_split_graph(dates_train, y_train, dates_validation, y_validation, dates_test, y_test)
    # network_training_prediction_graph(dates_train, training_predictions, y_train)
    # network_validation_prediction_graph(dates_validation, validation_predictions, y_validation)
    # network_testing_prediction_graph(dates_test, test_predictions, y_test)
    # all_predictions_graph(dates_train, dates_validation, dates_test, training_predictions, validation_predictions, test_predictions, y_train, y_validation, y_test)
    # both_loss_graphs(fits, history)
    # graph_comparison_different_data_size(data)

    # print(training_predictions, validation_predictions, test_predictions)
    # print(len(training_predictions), len(validation_predictions), len(test_predictions))
    print('Actual Stock Price | Predicted Stock Price')
    print_two_arrays(y_test, test_predictions)

# this function takes two arrays a and b and prints a[0], b[0] \n a[1], b[1] \n etc
def print_two_arrays(a, b):
  for i in range(len(a)):
    print('      ', round(a[i],2), '     |        ', round(b[i],2))

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

def graph_comparison_different_data_size(data):
  for each in data:
   plt.plot(each)
  #  plt.text(0, 0, str(each[19]))
  #  print(each, '\n')

  plt.title("Comparison of Validation Mean Absolute Error of Different Data Sizes")
  plt.xlabel("Epochs")
  plt.xticks(np.arange(1, 21, 1))
  plt.ylabel("Mean Absolute Error")  
  plt.legend(['1983-2023\n'+str(round(data[0][19],2))+'%', '1993-2023\n'+str(round(data[1][19],2))+'%', '2003-2023\n'+str(round(data[2][19],2))+'%', '2013-2023\n'+str(round(data[3][19],2))+'%', '2018-2023\n'+str(round(data[4][19],2))+'%', '2019-2023\n'+str(round(data[5][19],2))+'%', '2020-2023\n'+str(round(data[6][19],2))+'%', '2021-2023\n'+str(round(data[7][19],2))+'%', '2022-2023\n'+str(round(data[8][19],2))+'%', 'YTD\n'+str(round(data[9][19],2))+'%'], title= 'Legend (Percent Validation Mean Absolute Error)')
  plt.show()


# Outputs two graphs of our Network's loss over time and mean Absolute Error loss
# Can't combine into one as scale is far different for both
def both_loss_graphs(model_history, history):
  plt.title("Loss over Time")
  plt.plot(model_history.epoch, history.losses)
  plt.xlabel("Epochs")
  plt.ylabel("Loss Value in Percent")
  plt.show()

  plt.title("Validation Mean Absolute Error Loss over Time")
  plt.plot(model_history.epoch, history.val_mean_absolute_error)
  plt.xlabel("Epochs")
  plt.ylabel("Absolute Error Loss Value in Percent")
  plt.show()

  plt.title("Validation Loss over Time")
  plt.plot(model_history.epoch, history.validation_loss)
  plt.xlabel("Epochs")
  plt.ylabel("Validation_loss Loss Value in Percent")
  plt.show()

# Outputs the graph of all our Network's predictions vs the real observation data combined
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

# Outputs the graph of our Networks testing predictions vs the real testing observation data
def network_testing_prediction_graph(testing_dates, test_predictions, y_test):
  plt.title("Testing Predictions vs Observations of Network")
  plt.plot(testing_dates, test_predictions)
  plt.plot(testing_dates, y_test)
  plt.xlabel("Dates")
  plt.ylabel("Stock Value in Points")
  plt.legend(['Testing Predictions', 'Testing Observations'])
  plt.show()

# Outputs the graph of our Networks validation predictions vs the real validation observation data
def network_validation_prediction_graph(validation_dates, validation_predictions, y_val):
  plt.title("Validation Predictions vs Observations of Network")
  plt.plot(validation_dates, validation_predictions)
  plt.plot(validation_dates, y_val)
  plt.xlabel("Dates")
  plt.ylabel("Stock Value in Points")
  plt.legend(['Validation Predictions', 'Validation Observations'])
  plt.show()

# Outputs the graph of our Networks training predictions vs the real training observation data
def network_training_prediction_graph(training_dates, training_predictions, y_train):
  plt.title("Training Predictions vs Observations of Network")
  plt.plot(training_dates, training_predictions)
  plt.plot(training_dates, y_train)
  plt.xlabel("Dates")
  plt.ylabel("Stock Value in Points")
  plt.legend(['Training Predictions', 'Training Observations'])
  plt.show()

# Outputs the graph of our three data sets split by section
def show_data_split_graph(training_dates, y_train, validation_dates, y_validation, testing_dates, y_test):
  plt.title("Total Data Set Split into Training(80%), Validation(10%), Test(10%)")
  plt.plot(training_dates, y_train, color='red')
  plt.plot(validation_dates, y_validation, color='blue')
  plt.plot(testing_dates, y_test, color='green')
  plt.xlabel("Dates")
  plt.ylabel("Stock Values in Points")
  plt.legend(['Train', 'Validation', 'Test'])
  plt.show()

# Outputs the graph of the stock value sliced to our testing dates
def show_sliced_stock_graph(stock_data_frame, start_date, end_date):
  stock_data_frame = stock_data_frame.loc[start_date:end_date]
  plt.title("Sliced AAPL Stock Value Timeline")
  plt.plot(stock_data_frame.index, stock_data_frame['Close'])
  plt.xlabel("Dates")
  plt.ylabel("Stock Values in Points")
  plt.show()

# Outputs the graph of the stock value
def show_total_stock_graph(stock_data_frame):
  plt.title("AAPL Stock Value")
  plt.plot(stock_data_frame.index, stock_data_frame['Close'])
  plt.xlabel("Dates")
  plt.ylabel("Stock Values in Points")
  plt.show()

if __name__ == "__main__":
  main()