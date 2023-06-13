<h1 align="center">Utilizing AI to Crack the Stock Market and its Patterns or Lack thereof</h1>
<h3 align="center"><em>By Roy Schor, Zane Hankin, and Ethan Glazer</em></h3>
<h6 align="center">Link to Blog Post (same information as below):</h6>
<h6 align="center">https://medium.com/@royschor/artificial-neural-networks-and-stocks-7d17474c14c8</h6>

<h5 align="center">.&emsp;&emsp;&emsp;.&emsp;&emsp;&emsp;.</h5>
<p><em>This post is for those interested in following our thought process and actual steps of researching, creating, and testing a neural network from scratch on stock prices. Ultimately, to predict future stock values.<br></em></p>

## Our Motivation
The stock market’s spikes and downfalls have mystified people since its inception, with nothing, machine or human, able to predict its future effectively. With an AI Network called Aladdin making investments and controlling over 7% of the entire world’s financial assets, which is almost the size of the U.S.’s total GDP, we thought, why not take a crack at it?

Besides, what else could engross investment-minded students more than the constant fluxes of money in the market. This was our attempt at using a Neural Network to learn a certain pattern in the stock market and beat the 10% yearly avg increase in the market.

Did we succeed? I guess you’ll have to read on and find out…

## Our Approach
It began with a rough roadmap and goals to hit etched into a Google Doc (so long are the days of paper.) We decided to use an LSTM (Long Short-Term Memory) Neural Network model as the basis for our project.

*Here we will give some basic guiding info; however, you can search tons of websites, books, and videos to delve deeper into these topics; many of what we used will be at the bottom in the works cited section!*

#### Now what is an LSTM?
Fair question.

LSTM stands for Long Short-Term Memory, a recurrent neural network. An LSTM is an artificial neural network architecture suited for processing sequential data, such as time series or, in our case, daily closing stock values. These networks are unique in their ability to handle long-term information more effectively by having a selective storing, updating, and removal process for the data. LSTMs achieve this by using memory cells that can store data for a more extended period of time. Along with the use of gates that control what information goes into these memory cells (**input gate**), what is allowed out and forgotten (**forget gate**), and then finally, what is propagated onto the next stage (**output gate.**)

We aimed to take this type of model and train it on a set of stock closing values from a particular stock. We would then ask the model to predict the 5th day’s closing value based on the previous 4.

For example, the model should predict Friday’s close given Alphabet’s stock closing value on Monday, Tuesday, Wednesday, and Thursday.
<h5 align="center">.&emsp;&emsp;&emsp;.&emsp;&emsp;&emsp;.</h5>

## Different Approaches

We went through 2 main approaches; the initial one failed drastically. This led us to spend a lot more time researching the topic in more depth and finally creating the second successful approach.

As always, we will start with the failed first approach.
# Initial Approach, Data Setup, and Neural Network
### DATA, Data, data
Every good neural network relies on an immense database. If a machine is only as good as the data it is taught on, your best bet is to use as deep a net of data as possible. At least, that is what we thought; we will understand why this is only sometimes the case later.

We scoured [Kaggle](https://www.kaggle.com/) until we found the right stock data set for us, and there was a lot to choose from. We landed on a dataset made by [Boris Marjanovic](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs?resource=download); very much thanks to him.

The data is laid out in dozens of CSV files. Each file contains the full history of a stock up until 2017. The CSV has the date, open, close, high, low, close, and volume of each stock ordered by dates. Our first step was processing and cleaning our data.

![giphy](https://user-images.githubusercontent.com/70181314/236553068-afeb76d3-b18b-42cd-8fb7-e518dd4bed79.gif)

Below you can see how the raw data looked like before we cleaned all the nonsense up:

<img width="683" alt="Screen Shot 2023-05-05 at 9 35 44 PM" src="https://user-images.githubusercontent.com/70181314/236553399-69718ed5-2ef0-4bc9-99dc-8c07a6a7ae92.png">

After some coding magic:
<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br># Start and end date of the data we will reading in - interchangeable<br>
        start_date_1 = '2013-12-31'<br>
        end_date_1 = '2016-01-01'<br>
        # File we read in, we only read the Date and Close values from the CSV<br>
        stockDataFrame = pd.read_csv('aapl.us.txt', usecols=['Date', 'Close'], dtype={'Date': 'str', 'Close':'float'}, parse_dates=['Date'], index_col='Date')<br>
        # Reads in all the data, then slices it to only take the data after the start date<br>
        stockDataFrame = stockDataFrame.loc[start_date_1:end_date_1]<br><br>
    </h6>
</h5>

We ended up with this:

<img width="190" alt="Screen Shot 2023-05-01 at 1 41 04 AM" src="https://user-images.githubusercontent.com/70181314/236553825-39246090-3e7b-45fe-bfa1-28c9cddf6722.png">

Now we are getting somewhere.

The goal was only to get the dates and close values of the stock, with a key goal of using code that is easily transferable to any file or amount of files put in a loop. We grabbed only the date and close values removing all the other data as we thought it nonsensical for our purposes.

It is important to note that the dates column was only for our graphing purposes, and the network would only use the close values. In addition, each close value represents one day of trading.

Here’s how AAPL, or Apple Inc., has been performing from the start, our chosen stock to test everything with:

<img width="639" alt="Full AAPL Stock Graph" src="https://user-images.githubusercontent.com/70181314/236553967-7e49a559-ad29-4faa-9886-b68958530e33.png">

After more brainstorming, we split the data into window subsets, the 4 and 1 combo we mentioned earlier. This is where our network receives four days of data to predict the fifth stock data.

## Creating the Network
Now that we had our data formatted correctly, we had an idea of how we wanted it to be fed into our network; the only thing left was building the network.

We kept the infrastructure fairly simple: a Sequential Model with four inputs and one output, an LSTM layer, and one Dense layer.
- Four input neurons were for the four input days, with a corresponding one output neuron.
<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br>model = Sequential()<br>
        model.add(LSTM(64, input_shape=(4, 1)))<br>
        model.add(Dense(1, activation='relu'))<br><br>
    </h6>
</h5>

Additionally, we compiled the model with the [Adam optimizer](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c) and minimized the mean squared error for our loss. However, We used the metric of mean absolute error to better measure the accuracy of our prediction in comparison to our historical data.

<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br>model.compile(optimizer='adam', loss='mean_squared_error', metrics=’mean_absolute_error')<br>
        model.fit(x_train, y_train, epochs=3, verbose=2, validation_data=(x_test, y_test))<br><br>
    </h6>
</h5>

That was it, all that we needed for our model! (or so we thought)

## Feeding the Network and Getting Results(ish)

We had split our data to feed it in an 80%-20% fashion. 80% would be our training data, and 20% would be our testing.

We then created our x_train and y_train by creating two arrays, with x_train being the first four days of data and then our fifth day as our y-training value. This is the 4–1 philosophy we believed in.

<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br>raw_training_data.append(x_train_vals)<br>
        raw_training_data.append(y_train_vals, 0, 0, 0)<br><br>
    </h6>
</h5>

Running the model delivered very strange, unexplainable results. We had reached a standstill. We did not know what was going on, and despite much effort, we needed help to figure out how to move forward.

<img width="632" alt="Screen Shot 2023-05-04 at 2 33 19 PM" src="https://user-images.githubusercontent.com/70181314/236555854-9495c6d9-bbb6-46e7-8594-7bf2439be87f.png">

That is when we stopped coding. For the next few days, we were in research mode; the **scientist** in computer scientist was at the forefront.

We won’t go too much into the research or what we learned. However, if you go to the end of this post, you can find all of our works cited with a short description of what the source is mainly about.

## Second Approach — the successful one
### Comment on Research and Finds

We looked to others for similar projects and how to adjust them. There were two articles that helped us mend our mistakes. The first was [this article](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/). Shoutout to Jason Brownlee, the author of the article, for better articulating how to predict time series data points using LSTMs. Due to this article, we took a step back and rethought our initial plan of an LSTM model: build short-term memory for our RNN to train on. The source that propelled our progress was [this youtube video](https://www.youtube.com/watch?v=CbTU92pbDKw&ab_channel=GregHogg) that gave us a more desirable approach to splitting up our data. Instead of arranging the data into arrays for x and y, we learned we should store the data for x and y together in a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

Through cherry-picking information, plenty of research, and newly found knowledge to lead us, we felt prepared to tackle the challenge.

### Data formatting
It all comes back to the data…

Instead of retaining only *‘Date’* and *‘Close,’* we appended the past 4 days to the same row of the prediction date:

<img width="542" alt="Picture of data layout" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/bb054846-2394-460b-852c-d96777814aaf">


Above, you can see that each date has 5 corresponding stock values. Each **‘Target-X’** is one of 4 previous stock days, with **‘Target’** being the current stock day and the prediction target day for our network.

Now we can slide forward one day between four-day splits and train our model to output the fifth day for the entirety of our data. We will call this the *“Sliding Window”* method. This makes it easier to visualize, organize and access the data as well as expand the amount of data we have. Instead of using 5 data points and then moving on to the next five, we can simply slide one day at a time, as seen below:

<img width="591" alt="Screen Shot 2023-05-04 at 5 24 13 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/6d0434e0-5999-4da8-8a81-76e3fe4ed3ca">

We used the following code to do so, with two helper functions; if you read through the comments of the code, it should give you a solid idea of what is going on:

<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br>def main():<br>
        &emsp;&emsp;# File we read in<br>
        &emsp;&emsp;stock_data_frame = pd.read_csv('AAPL.csv')<br>
        &emsp;&emsp;# we only take in the Date and Close values from the CSV<br>
        &emsp;&emsp;stock_data_frame = stock_data_frame[['Date', 'Close']]<br>
        &emsp;&emsp;# Converts all Dates of type string to type Datetime<br>
        &emsp;&emsp;stock_data_frame['Date'] = stock_data_frame['Date'].apply(convert_to_date)<br>
        &emsp;&emsp;# need to remove the first column (index column) as its useless data<br>
        &emsp;&emsp;stock_data_frame.index = stock_data_frame.pop('Date')<br><br>
        &emsp;&emsp;windowed_df = split_df_to_windowed_df(stock_data_frame, window_size=4)<br><br>
        &emsp;&emsp;# Here we slice the windowed data frame greatly<br>
        &emsp;&emsp;# We believe that the network training all the data actually harms its predictions <br>
        &emsp;&emsp;# as it is not training on the most volatile part (the recent history), <br>
        &emsp;&emsp;# thus we are now trying to only train on recent history (past 3 years not all 30+)<br>
        &emsp;&emsp;start_date = "1983-01-01"<br>
        &emsp;&emsp;end_date = "2023-05-03"<br>
        &emsp;&emsp;windowed_df = windowed_df.loc[start_date:end_date]<br>
        &emsp;&emsp;windowed_df = windowed_df.reset_index()<br><br>
        &emsp;&emsp;# This function splits the dataframe into windows to feed into the network<br>
        &emsp;&emsp;# Each row now includes a date, followed by previous 4 days, and final column is prediction day's data<br>
        &emsp;&emsp;def split_df_to_windowed_df(data, window_size=4):<br>
        &emsp;&emsp;&emsp;# Creates our temporary datafram<br>
        &emsp;&emsp;&emsp;windowed_df = pd.DataFrame()<br><br>
        &emsp;&emsp;&emsp;for index in range(window_size, 0, -1):<br>
        &emsp;&emsp;&emsp;&emsp;# appends to each row in df past 4 days of data, one day at a time and shifts the entire data doing so<br>
        &emsp;&emsp;&emsp;&emsp;windowed_df[f'Target-{index}'] = data['Close'].shift(index)<br><br>
        &emsp;&emsp;&emsp;windowed_df['Target'] = data['Close']<br>
        &emsp;&emsp;&emsp;# Removes all rows that are missing some data - any incomplete windows that would mess up training<br>
        &emsp;&emsp;&emsp;return windowed_df.dropna()<br><br><br>
        &emsp;&emsp;# Each date in the dataframe is a string but we want it as a Date object<br>
        &emsp;&emsp;# thus this function converts a string to Datetime object<br>
        &emsp;&emsp;def convert_to_date(stringDate):<br>
        &emsp;&emsp;&emsp;# splits based on hyphen as the string of date is seperated by hyphen<br>
        &emsp;&emsp;&emsp;split_value = stringDate.split('-')<br>
        &emsp;&emsp;&emsp;year, month, day = int(split_value[0]), int(split_value[1]), int(split_value[2])<br>
        &emsp;&emsp;&emsp;return datetime.datetime(year=year, month=month, day=day)<br>
    </h6>
</h5>

Now that our data frame was formatted into the sliding windows, we needed to reshape the data so that the LSTM could take it. The reshaping looked a little like this:

<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br># Now need to fix data to be numpy and reshape it to fit in LSTM<br>
        df_npied = windowed_df.to_numpy()<br>
        # grab only all the dates, first column is dates<br>
        dates = df_npied[:, 0]<br>
        # takes all past data points exluding date and target date data, so first and last<br>
        middle_data_segment = df_npied[:, 1:-1]<br>
        # reshaped by length of dates, the size of the middle part, and 1 for us as this is a univariate problem<br>
        total_x_data = middle_data_segment.reshape((len(dates), middle_data_segment.shape[1], 1))<br>
        total_y_data = df_npied[:, -1]<br>
        # this fixed a bug adding float32 conversion<br>
        total_x_data = total_x_data.astype(np.float32)<br>
        total_y_data = total_y_data.astype(np.float32)<br>
        # dates.shape = (8360), total_x_data.shape = (8360,4,1) (4 steps in past) (1 float variable), total_y_data.shape = (8360)<br><br>
    </h6>
</h5>

### Creating Training, Validation, and Test Data
We kept the same strategy as the first approach. An 80–10–10 split. 80% of the now reshaped data will be used for training, 10% for validation to tweak the network, and 10% for test data.

This was a pretty simple slicing process:

<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br># Now need to fix data to be numpy and reshape it to fit in LSTM<br>
        # Now we create training, testing, and validation data<br>
        # We will do 80% training, the remaining 20% is split 10-10 into validation and testing<br>
        eighty_split = int(len(dates) * .8)<br>
        ninety_split = int(len(dates) * .9)<br><br>
        # Up until 80%<br>
        dates_train, x_train, y_train = dates[:eighty_split], total_x_data[:eighty_split], total_y_data[:eighty_split]<br>
        # between 80% - 90%<br>
        dates_validation, x_validation, y_validation = dates[eighty_split:ninety_split], total_x_data[eighty_split:ninety_split], total_y_data[eighty_split:ninety_split]<br>
        # 90% - end<br>
        dates_test, x_test, y_test = dates[ninety_split:], total_x_data[ninety_split:], total_y_data[ninety_split:]<br><br>
    </h6>
</h5>

Below you can see the split of the data visually:

<img width="795" alt="Data Split Graph" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/ebb5aa19-826c-4df6-a27a-c2a2b777f94d">
