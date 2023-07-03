<h1 align="center">Utilizing AI to Crack the Stock Market and its Patterns or Lack thereof</h1>
<h3 align="center"><em>By Roy Schor, Zane Hankin, and Ethan Glazer</em></h3>
<h6 align="center">Link to Blog Post (same information as below simply looks better on Medium as opposed to Readme):</h6>
<h6 align="center">https://medium.com/@royschor/artificial-neural-networks-and-stocks-7d17474c14c8</h6>

<h5 align="center">.&emsp;&emsp;&emsp;.&emsp;&emsp;&emsp;.</h5>
<p><em>This post is for those interested in following our thought process and actual steps of researching, creating, and testing a neural network from scratch on stock prices. Ultimately, to predict future stock values.<br></em></p>

## Table of Contents
- [Our Motivation](https://github.com/RoySchor/Stock-Neural-Network#our-motivation)
- [Initial Approach, Data Setup, and Neural Network](https://github.com/RoySchor/Stock-Neural-Network#initial-approach-data-setup-and-neural-network)
- [Second Approach — the successful one](https://github.com/RoySchor/Stock-Neural-Network#second-approach--the-successful-one)
    - [Creating Training, Validation, and Test Data](https://github.com/RoySchor/Stock-Neural-Network#creating-training-validation-and-test-data)
    - [Predicting Values](https://github.com/RoySchor/Stock-Neural-Network#ready-to-predict)
- [Final Results](https://github.com/RoySchor/Stock-Neural-Network#final-results)
- [Conclusion and Takeaways](https://github.com/RoySchor/Stock-Neural-Network#conclusion-and-takeaways)
- [Key Terms](https://github.com/RoySchor/Stock-Neural-Network#key-terms)
- [Works Cited](https://github.com/RoySchor/Stock-Neural-Network#works-cited)

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
        <br># Now we create training, testing, and validation data<br>
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

### Creating and fitting the Network

To create the network, we took the same build we made for the first approach and made it a bit more complex. Tips from the Youtube Video by Greg Hogg and the article by Jason Brownlee helped greatly.

Our model had an LSTM layer of shape 4–1 for the 4 input dates used to output 1 prediction date. We then added 2 Dense layers and used the Adam optimizer. A lot of fiddling led us to set the learning_rate = 0.001; initially, it was at 0.01, but from research, we learned the best value is case specific and requires trial and error. Our loss and metrics remained the same as we had in the first approach.

<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br># Creating the Network 4,1 for 4 inputs dates one output<br>
        model = Sequential()<br>
        model.add(layers.LSTM(64, input_shape=(4,1)))<br>
        model.add(layers.Dense(32, activation='relu'))<br>
        model.add(layers.Dense(32, activation='relu'))<br>
        model.add(layers.Dense(1, activation='relu'))<br>
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])<br><br>
        fits = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20, callbacks=[history])<br>
        training_predictions = model.predict(x_train).flatten()<br>
        validation_predictions = model.predict(x_validation).flatten()<br>
        test_predictions = model.predict(x_test).flatten()<br><br>
    </h6>
</h5>

We also decided to add a callback. A callback for a model is a function that gets called at a set time period, which is flexible to whatever the coder sets. We wanted to grab the loss and metrics data at the end of each epoch and append it to the respective array. We created a class to do just that:

<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br>class Histories(Callback):<br>
        &emsp;&emsp;def on_train_begin(self, logs={}):<br>
        &emsp;&emsp;&emsp;self.losses = []<br>
        &emsp;&emsp;&emsp;self.val_mean_absolute_error = []<br>
        &emsp;&emsp;&emsp;self.validation_loss = []<br><br>
        &emsp;&emsp;def on_epoch_end(self, batch, logs={}):<br>
        &emsp;&emsp;&emsp;self.losses.append(logs.get('loss'))<br>
        &emsp;&emsp;&emsp;self.validation_loss.append(logs.get('val_loss'))<br>
        &emsp;&emsp;&emsp;self.val_mean_absolute_error.append(logs.get('val_mean_absolute_error'))<br><br>
    </h6>
</h5>

Now we really were set. Set to see what we had created and what it looked like visually.

## Ready To Predict
Initially, we thought of training the model with all of the stock data YTD. The more data, the merrier, right?

Well, no, not really. The problem we found was that initially, most stocks are very stagnant at a low value. This is until, if they succeed as a company, the stock jumps up and then is very fidgety but overall increasing. However, because the stock starts off very stagnant and remains as such for a long period of time, our model is trained on that stagnant growth. Thus, it believes the pattern will hold, messing up the predictions for the actual stock booms and busts. Below are the different predictions (model-created data) vs. observations (real data).

<img width="1038" alt="Initial Validation Prediction Graph" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/eb0f795d-6550-467f-8947-d5fd46b9d8c9">

<img width="1057" alt="Screen Shot 2023-06-12 at 10 09 52 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/327c67da-0658-4de9-9168-bc9a4e80ff40">

<img width="530" alt="Screen Shot 2023-05-05 at 12 56 44 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/48cacff9-ebec-4a3c-b611-49f6912759e9">

<img width="967" alt="Initial testing Prediction Graph" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/5f9adb07-4975-47cb-bb06-1b1e382573e3">

Each graph shows how our network’s guess compared to the real stock value. The first shows the overall validation prediction vs. the validation observation or real data points. Our network is unable to follow the initial jump, retaining the past pattern of slow increases.

This is then seen again in the second graph where we show all three types of predictions vs. actual data points. The network’s prediction, followed the observation very closely until the stagnation point as seen.

As seen, it fails to follow the essential pattern of the stock after it booms. The data from two years ago caused its projections to stall out after a certain point.

### Cutting Down the Data
From this, we decided to feed the network a smaller portion of data. We chose 1 year as we only wanted to detect patterns that would help predict tomorrow’s stock price.

<img width="639" alt="Full AAPL Stock Graph" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/e8afb490-4b46-4db5-83d2-288ae6bda1b1">

Above is the entire data for AAPL stocks from 1984–2017. We originally used this to train our model, but after realizing we needed to train on a much smaller data set to get the best results, we cut about 31 years of data, focusing on the two years between 2015 and 2017. Our performance was much better.

<img width="730" alt="Screen Shot 2023-05-05 at 12 58 59 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/0ba3566d-5ec3-4417-b064-9f5fb516bbd9">

You can see our Validation Prediction (green) and our Testing Prediction (purple) compared to our Validation Observations (red) and Testing Observations (Brown) perform much better than our previous model. The stagnation is far better.

Looking at the graph further, we noticed a **gap in our data.** We believed our source had a gap in its information storage. That was no good; the gap would cause our network not to predict well after it. We decided to use [Yahoo Finance](https://finance.yahoo.com/?guccounter=1&guce_referrer=aHR0cHM6Ly9tZWRpdW0uY29tL0Byb3lzY2hvci9hcnRpZmljaWFsLW5ldXJhbC1uZXR3b3Jrcy1hbmQtc3RvY2tzLTdkMTc0NzRjMTRjOA&guce_referrer_sig=AQAAAJ2Mqja8uZD5cOAZXVip5fSogWk2cjx6JDSs8OsCbJV9-MCiY0_Kyg_NVtZc5DOt2QPA9pjdDqmFYDxmL-Utk59umBdIn0CogSY1M0FmYUozhEUWAWfQFQHc1H0_nOj7QLIxiO2WsqHCyk1GC23ENkBfQXKB9cmxQzD-gu6vHDMU) which gives us more complete historical data to download and access.

After downloading [this dataset](https://finance.yahoo.com/quote/AAPL/history?period1=345427200&period2=1683072000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true), we re-trained our model with a variety of dates.

First, we ran our model with the entirety of the dataset. Below we received what was expected with a fairly rapid stagnation/declination, which makes sense after being trained on closing prices that increased fairly incrementally.

<img width="669" alt="Screen Shot 2023-05-05 at 1 02 58 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/0abb8b8b-4ddf-4a63-9b7c-f44863ddc2f4">

### Narrowing Training Size
With our new approach of narrowing down the window size to a smaller span of years, we decided to figure out which window size of data is our “Goldilocks,” so to say. We trained our model for 20 epochs on several timeframes and determined the window size with the lowest Validation Mean Absolute Error was from January 01, 2019 — May 03, 2023. See the graph of our comparisons below:

*Note: when we trained on the data from January 01, 2023 — May 03, 2023 (Year to Date, or YTD), our model could not determine a specific enough pattern. We believe it was too short of a period. This means that both too much and too little data harm the network. The key is finding the exact amount to get the best result.*

<img width="753" alt="Screen Shot 2023-05-05 at 1 07 20 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/7b683d2d-3b6e-4833-be4e-b795b9d43ac9">

We then looked further into our predictions after training the data, specifically on the data between January 01, 2019 — May 03, 2023.

Taking the following sliced stock from AAPL:

<h5 align="center">
    <h6 style="display: inline-block; text-align: left;">
        <br>start_date = "2019-01-01"<br>
        end_date = "2023-05-03"<br>
        windowed_df = windowed_df.loc[start_date:end_date]<br><br>
    </h6>
</h5>

<img width="714" alt="Screen Shot 2023-05-05 at 1 08 34 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/b146d001-8b61-4537-b874-45dcb1f8ec28">

We split the data into the following 80% Training, 10% Validation, and 10% Test:

<img width="719" alt="Screen Shot 2023-05-05 at 1 10 00 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/ac17cdee-c71b-4994-b2dc-cef05607a338">

## Final Results
Once we sent the data into our model to run, we printed out a comparison of the Actual Stock Price and our model’s Predicted Stock Price. It looked pretty good; although we overshot some prices, the predictions were incredibly close on average.

<img width="698" alt="Screen Shot 2023-05-05 at 1 11 05 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/0d002277-430f-4414-a327-efa9184d91bb">

To better visualize the losses that our model faced, we sought out to create some graphs:

**Look at the scale for many of them; the line seems far off and could be misleading. For some, the scale of stock points is on the micro level!**

<img width="731" alt="Screen Shot 2023-05-05 at 1 12 36 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/c736844c-1a16-4fc4-a622-bdfafa2ae732">

<img width="729" alt="Screen Shot 2023-05-05 at 1 14 09 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/306ddfc1-c3af-42d8-8be1-39ebf63bbf5a">

<img width="736" alt="Screen Shot 2023-05-05 at 1 14 15 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/7c8133c2-3b4c-4304-8e93-284171976b52">

<img width="730" alt="Screen Shot 2023-05-05 at 1 14 24 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/0a67efe0-204b-43b2-ac73-5ba87cc5ac1c">

<img width="717" alt="Screen Shot 2023-05-05 at 1 15 09 PM" src="https://github.com/RoySchor/Stock-Neural-Network/assets/70181314/f923b60b-df43-4aaf-b0c8-30076c808777">

## Conclusion and Takeaways
In the end, we were able to build a model that could extrapolate stock prices for X days into the future based on the AAPL stock. Since we were using historical data, we can determine that we did this with a solid success rate — *the rate changes depending on the period and stock.* This means that a network can predict a stock value in the short term, and there is somewhat of a pattern that is found.

#### Expansion
Some things we would hope to add to this project in the future include:

- Be able to automatically scrape stock data from Yahoo Finance by simply using a given stock ticker and its API.
- Make it user dependent by taking in X amount of money and simulating its increase or decrease in the stock market during a given period. In addition, to retain a profit that is no worse than the Dow Jones Industrial Average of 10% yearly.
- Improve our prediction by scraping news outlets to help minutely influence changes/drops in stock prices, which would influence the weights and biases of our network.

## Key Terms
#### Training Data
The data we trained our model on. This is represented by 80% of the data we selected from our dataset. We apply the “Window Sliding” method on this data for the model to learn what the 5th day’s closing price should be.

#### Validation Data
The data we validated our model with. This is represented by the next 10% of the data we selected from our dataset. We apply the “Window Sliding” method on this data for the model to learn what the 5th day’s closing price should be.

#### Testing Data
This is the data we test our model with. This is represented by the last 10% of the data we selected from our dataset. After training, we feed our model windows of four days, asking it to predict the fifth day.

#### Training, Validation, and Testing Predictions
For training and testing, this is the “fifth day” data that our model predicts for each 4-day sliding time frame. For validation, our model is fine-tuning the parameters of the network based on its predictions vs. real observation.

#### Training, Validation, and Testing Observations
These are the actual closing prices from the data we selected. It is what the company’s actual stock looked like during this window — used as a comparison value.

#### Training Loss
Indicates how bad the training predictions were compared to the observations. If perfect, the training loss is zero. As the model does worse, the training loss increases.

#### Validation Loss
Indicates how bad the validation predictions were compared to the observations.

#### Validation Mean Absolute Error
Assesses the error between the validation observation and the validation predictions. It is calculated by the sum of absolute errors divided by the sample size. So for us, it is the sum of absolute errors between validation predictions and observations, divided by the number of data points (days) in our selected window size.

## Works Cited
- [https://finance.yahoo.com/quote/AAPL/history?period1=345427200&period2=1683072000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true](https://finance.yahoo.com/quote/AAPL/history?period1=345427200&period2=1683072000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)

Second complete data set
- [https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs?resource=download](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs?resource=download)

Initial data set
- [https://keras.io/](https://keras.io/)

Every sub page related to anything keras was used for a lot of research info.
- [https://www.linkedin.com/pulse/can-artificial-intelligence-used-improve-stock-trading-bernard-marr/](https://www.linkedin.com/pulse/can-artificial-intelligence-used-improve-stock-trading-bernard-marr/)

Research article regarding AI and stocks.
- [https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

Articulating how to predict time series data points using LSTMs.
- [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

Pandas documentation center
- [https://www.thinkpolnews.com/the-powerful-ai-shaping-the-world-meet-aladdin-from-blackrock/#:~:text=An%20AI%20that%20is%20much,by%20multinational%20investment%20firm%20BlackRock.](https://www.thinkpolnews.com/the-powerful-ai-shaping-the-world-meet-aladdin-from-blackrock/#:~:text=An%20AI%20that%20is%20much,by%20multinational%20investment%20firm%20BlackRock.)

Article relating to Blackrock’s Aladdin and what it is/ does
- [https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)

Article regarding Adam optimizer and its benefits
- [https://www.youtube.com/watch?v=CbTU92pbDKw&ab_channel=GregHogg](https://www.youtube.com/watch?v=CbTU92pbDKw&ab_channel=GregHogg)

Similar stock prediction project
