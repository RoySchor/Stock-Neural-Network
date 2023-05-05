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
<div style="text-align: center;">
    <div style="display: inline-block; text-align: left;">
        <br># Start and end date of the data we will reading in - interchangeable<br>
        start_date_1 = '2013-12-31'<br>
        end_date_1 = '2016-01-01'<br>
        # File we read in, we only read the Date and Close values from the CSV<br>
        stockDataFrame = pd.read_csv('/Users/royschor/Desktop/Core Course/archive/aapl.us.txt', usecols=['Date', 'Close'], dtype={'Date': 'str', 'Close':         'float'}, parse_dates=['Date'], index_col='Date')<br>
        # Reads in all the data, then slices it to only take the data after the start date<br>
        stockDataFrame = stockDataFrame.loc[start_date_1:end_date_1]<br><br>
    </div>

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
<div style="text-align: center;">
    <div style="display: inline-block; text-align: left;">
        <br>model = Sequential()<br>
        model.add(LSTM(64, input_shape=(4, 1)))<br>
        model.add(Dense(1, activation='relu'))<br><br>
    </div>

Additionally, we compiled the model with the [Adam optimizer](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c) and minimized the mean squared error for our loss. However, We used the metric of mean absolute error to better measure the accuracy of our prediction in comparison to our historical data.

<p><br>model.compile(optimizer='adam', loss='mean_squared_error', metrics=’mean_absolute_error')<br>
model.fit(x_train, y_train, epochs=3, verbose=2, validation_data=(x_test, y_test))<br><br></p>

That was it, all that we needed for our model! (or so we thought)

## Feeding the Network and Getting Results(ish)

We had split our data to feed it in an 80%-20% fashion. 80% would be our training data, and 20% would be our testing.

We then created our x_train and y_train by creating two arrays, with x_train being the first four days of data and then our fifth day as our y-training value. This is the 4–1 philosophy we believed in.


