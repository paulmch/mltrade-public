import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder 
import boto3
from sklearn.model_selection import train_test_split
import requests
from io import StringIO
from datetime import datetime, timedelta
import pytz
import time
import sys
import logging
import ta
from sklearn.model_selection import train_test_split
import tensorflow as tf
from bayes_opt import BayesianOptimization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout,LeakyReLU,BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import traceback
import subprocess
import json
import gc
import os 
from ncps import wirings
from ncps.tf import LTC



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

try:
    test = os.environ["manual"]
    manual = True
    logger.info("manual mode")
except:
    manual = False


def one_hot_encode_column(data: pd.DataFrame, col, max_classes=None):
    df = data.copy()
    
    if max_classes is not None:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
         
        missing_cols = set(range(max_classes + 1)) - set(df[col].unique())
        for col_num in missing_cols:
            dummies[col +"_"+ str(col_num)] = 0
        
        # Custom sort function to ensure correct column order
        dummies = dummies[sorted(dummies.columns, key=lambda x: int(x.split("_")[1]))]
    else:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)

    df = pd.concat([df, dummies], axis=1)
    if max_classes is None:
        df = df.drop(columns=[col])

    return df

def read_data_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'], parse_dates=['Date'])


def fetch_and_concat_data(api_key):
    base_url = "https://api.twelvedata.com/time_series"
    #start_date = datetime(year=2023, month=12, day=1)
    start_date = datetime(year=2019, month=10, day=30)
    end_date = datetime.now()  # Or any other end date you want

    all_data_frames = []
    request_count = 0
    delay_between_calls = 60 / 53
    while start_date < end_date:
        params = {
            "symbol": "GDAXI",
            "interval": "5min",
            "format": "CSV",
            "apikey": api_key,
            "start_date": start_date.strftime("%m/%d/%Y 9:00"),
            "end_date": (start_date + timedelta(days=1)).strftime("%m/%d/%Y 17:31")
        }

        response = requests.get(base_url, params=params)
        request_count += 1

        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, delimiter=';')
            df = df.iloc[::-1]
            all_data_frames.append(df)
        else:
            print(f"Failed to fetch data for {start_date.strftime('%Y-%m-%d')}")

        # Check if rate limit is reached
        time.sleep(delay_between_calls)

        

        if request_count % 100 == 0:
            logger.info(f"Request count: {request_count}")
           

        start_date += timedelta(days=1)

    # Concatenate all data frames
    final_df = pd.concat(all_data_frames, ignore_index=True)

    # Remove duplicates based on the datetime column
    final_df.drop_duplicates(subset='datetime', keep='first', inplace=True)

    return final_df

# Function to calculate daily candles updated at every 5-minute interval
def calculate_updated_daily_candles(df):
    # Create an empty DataFrame for the updated daily candles
    updated_daily_candles = pd.DataFrame(columns=['open', 'high', 'low', 'close'], index=df.index)

    # Iterate over the 5-minute candles
    for current_time in df.index:
        # Filter data up to the current timestamp
        current_day_data = df[df.index.date == current_time.date()]
        up_to_current_time_data = current_day_data[current_day_data.index <= current_time]

        # Calculate updated daily candle
        updated_daily_candles.loc[current_time, 'open'] = current_day_data.iloc[0]['open']
        updated_daily_candles.loc[current_time, 'high'] = up_to_current_time_data['high'].max()
        updated_daily_candles.loc[current_time, 'low'] = up_to_current_time_data['low'].min()
        updated_daily_candles.loc[current_time, 'close'] = up_to_current_time_data.iloc[-1]['close']

    return updated_daily_candles
def pricelong(indexvalue, knockoutprice) :
    """Calculate the price of long options """
    return np.maximum((indexvalue - knockoutprice) * 0.01, 0)

def priceshort(indexvalue, knockoutprice) :
    """Calculate the price of short options"""
    return np.maximum((knockoutprice - indexvalue) * 0.01, 0)

def prepare_intra_day_data(date):
    """
    Prepares intra-day data for a given date.
    """
    # Load the data for the given date
    open_day = df[df["Date"] == date]["Open"]
    knockout_long = 0.97 * open_day
    knockout_short = 1.03 * open_day
    open_price_long = pricelong(open_day,knockout_long) 
    open_price_short = priceshort(open_day,knockout_short)
    df_intra_day = intradaydax_agg[intradaydax_agg["Date"]==date]
    temp_list = []
    for element in df_intra_day.iloc[0:17].iterrows():
        high_long = pricelong(element[1]["High"],knockout_long)
        low_long = pricelong(element[1]["Low"],knockout_long)
        close_long = pricelong(element[1]["Close"],knockout_long)

        high_short = priceshort(element[1]["High"],knockout_short)
        low_short = priceshort(element[1]["Low"],knockout_short)
        close_short = priceshort(element[1]["Close"],knockout_short)
        perc_high_long = (high_long - open_price_long) / open_price_long
        perc_low_long = (low_long - open_price_long) / open_price_long
        perc_close_long = (close_long - open_price_long) / open_price_long
        perc_high_short = (high_short - open_price_short) / open_price_short
        perc_low_short = (low_short - open_price_short) / open_price_short
        perc_close_short = (close_short - open_price_short) / open_price_short

        temp_list.append([perc_high_long.iloc[0],perc_low_long.iloc[0],perc_close_long.iloc[0],perc_high_short.iloc[0],perc_low_short.iloc[0],perc_close_short.iloc[0]])
   
    
    
    return temp_list

def findtarget(date):
    close_day = df[df["Date"] == date]["Close"].values[0]
    high_day = df[df["Date"] == date]["Close"].values[0]
    low_day = df[df["Date"] == date]["Close"].values[0]
    df_intra_day = intradaydax_agg[intradaydax_agg["Date"]==date]

    index_value = df_intra_day.iloc[19]["Close"]
    

    knockout_long = 0.97 * index_value
    knockout_short = 1.03 * index_value

    open_price_long = pricelong(index_value,knockout_long) 
     
    close_price_long = pricelong(close_day,knockout_long)
     

    if close_price_long > open_price_long:
        return 1
    else:
        return 0

def upload_file_to_s3(bucket, key, file):
    s3 = boto3.client("s3")
    s3.upload_file(file, bucket, key)
    
def write_data_to_s3(bucket, key, content):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=content)
     
def evaluate_strategy(date, direction):
    close_day = df[df["Date"] == date]["Close"].values[0]
    df_intra_day = intradaydax_agg[intradaydax_agg["Date"]==date]

    index_value = df_intra_day.iloc[19]["Close"]
    

    knockout_long = 0.97 * index_value
    knockout_short = 1.03 * index_value

    open_price_long = pricelong(index_value,knockout_long)
    open_price_short = priceshort(index_value,knockout_short) 
     
    close_price_long = pricelong(close_day,knockout_long)
    close_price_short = priceshort(close_day,knockout_short)

    if close_price_long < 0.71 *open_price_long:
        close_price_long = 0.71 *open_price_long
    if close_price_short < 0.71 *open_price_short:
        close_price_short = 0.71 *open_price_short 
    
    if direction == 1:
        del df_intra_day
        return 1+ (close_price_long - open_price_long) / open_price_long
    else:
        del df_intra_day
        return 1+(close_price_short - open_price_short) / open_price_short
     
class TimeLimitExceededException(Exception):
    pass

def current_german_time():
    utc_now = datetime.now(pytz.utc)
    berlin_tz = pytz.timezone('Europe/Berlin')
    return utc_now.astimezone(berlin_tz)

def should_terminate(termination_time):
    german_time = current_german_time()
    # Ensure termination_time is offset-aware and in the same timezone
    if termination_time.tzinfo is None or termination_time.tzinfo.utcoffset(termination_time) is None:
        berlin_tz = pytz.timezone('Europe/Berlin')
        termination_time = berlin_tz.localize(termination_time)
    return german_time >= termination_time


bucket = "REDACTED_BUCKET"
dax_key = "dax.csv"
df = read_data_from_s3(bucket, dax_key)

df = df.dropna()
df["Open_200_MA"] = df["Open"].rolling(window=200).mean()
df["High_200_MA"] = df["High"].rolling(window=200).mean()
df["Low_200_MA"] = df["Low"].rolling(window=200).mean()
df["Close_200_MA"] = df["Close"].rolling(window=200).mean()
df["Open_100_MA"] = df["Open"].rolling(window=100).mean()
df["High_100_MA"] = df["High"].rolling(window=100).mean()
df["Low_100_MA"] = df["Low"].rolling(window=100).mean()
df["Close_100_MA"] = df["Close"].rolling(window=100).mean()
df["Open_10_MA"] = df["Open"].rolling(window=10).mean()
df["High_10_MA"] = df["High"].rolling(window=10).mean()
df["Low_10_MA"] = df["Low"].rolling(window=10).mean()
df["Close_10_MA"] = df["Close"].rolling(window=10).mean()
df = df.dropna()
df['RSI_open'] = ta.momentum.rsi(df['Open'])
df['RSI_open'] = (df['RSI_open'] - 50) / 50
df['RSI_diff_open'] = df['RSI_open'].diff()

df['Close_lag1'] = df['Close'].shift(1)
df['Open_lag1'] = df['Open'].shift(1)
df['Low_lag1'] = df['Low'].shift(1)
df['High_lag1'] = df['High'].shift(1)

df['Close_lag1_200_MA'] = df['Close_200_MA'].shift(1)
df['Open_lag1_200_MA'] = df['Open_200_MA'].shift(1)
df['Low_lag1_200_MA'] = df['Low_200_MA'].shift(1)
df['High_lag1_200_MA'] = df['High_200_MA'].shift(1)

df['Close_lag1_100_MA'] = df['Close_100_MA'].shift(1)
df['Open_lag1_100_MA'] = df['Open_100_MA'].shift(1)
df['Low_lag1_100_MA'] = df['Low_100_MA'].shift(1)
df['High_lag1_100_MA'] = df['High_100_MA'].shift(1)

df['Close_lag1_10_MA'] = df['Close_10_MA'].shift(1)
df['Open_lag1_10_MA'] = df['Open_10_MA'].shift(1)
df['Low_lag1_10_MA'] = df['Low_10_MA'].shift(1)
df['High_lag1_10_MA'] = df['High_10_MA'].shift(1)

df["Open_Close"] = (df["Open"]- df["Close_lag1"]) / df["Open"]
df["Open_Open"] = (df["Open"]- df["Open_lag1"]) / df["Open"]
df["Open_Low"] = (df["Open"]- df["Low_lag1"]) / df["Open"]
df["Open_High"] = (df["Open"]- df["High_lag1"]) / df["Open"]

df["Open_Close_200_MA"] = (df["Open"]- df["Close_lag1_200_MA"]) / df["Open"]
df["Open_Open_200_MA"] = (df["Open"]- df["Open_lag1_200_MA"]) / df["Open"]
df["Open_Low_200_MA"] = (df["Open"]- df["Low_lag1_200_MA"]) / df["Open"]
df["Open_High_200_MA"] = (df["Open"]- df["High_lag1_200_MA"]) / df["Open"]

df["Open_Close_100_MA"] = (df["Open"]- df["Close_lag1_100_MA"]) / df["Open"]
df["Open_Open_100_MA"] = (df["Open"]- df["Open_lag1_100_MA"]) / df["Open"]
df["Open_Low_100_MA"] = (df["Open"]- df["Low_lag1_100_MA"]) / df["Open"]
df["Open_High_100_MA"] = (df["Open"]- df["High_lag1_100_MA"]) / df["Open"]

df["Open_Close_10_MA"] = (df["Open"]- df["Close_lag1_10_MA"]) / df["Open"]
df["Open_Open_10_MA"] = (df["Open"]- df["Open_lag1_10_MA"]) / df["Open"]
df["Open_Low_10_MA"] = (df["Open"]- df["Low_lag1_10_MA"]) / df["Open"]
df["Open_High_10_MA"] = (df["Open"]- df["High_lag1_10_MA"]) / df["Open"]

df['EMA_Close'] = ta.trend.ema_indicator(df['Close'])
df['EMA_Ratio_Close'] = (df['Close'] - df['EMA_Close']) / df['Close']
bollinger = ta.volatility.BollingerBands(close=df['Close'])
df['BB_Bandwidth'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / bollinger.bollinger_mavg()
df['BB_Percent'] = (df['Close'] - bollinger.bollinger_lband()) / (bollinger.bollinger_hband() - bollinger.bollinger_lband())/2
stochastic = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
df['Stochastic_Scaled'] = stochastic / 100.0
df['EMA_Ratio_Close'] = df['EMA_Ratio_Close'].shift(1)
df['BB_Bandwidth'] = df['BB_Bandwidth'].shift(1)
df['BB_Percent'] = df['BB_Percent'].shift(1)
df['Stochastic_Scaled']  = df['Stochastic_Scaled'].shift(1)

df['SMA_5'] = ta.trend.SMAIndicator(close=df['Close'], window=5).sma_indicator()
df['SMA_10'] = ta.trend.SMAIndicator(close=df['Close'], window=10).sma_indicator()
df['MA_Crossover_Signal'] = 0
df.loc[df['SMA_5'] > df['SMA_10'], 'MA_Crossover_Signal'] = 1  # Bullish crossover
df.loc[df['SMA_5'] < df['SMA_10'], 'MA_Crossover_Signal'] = -1  # Bearish crossover
df['MA_Crossover'] = (df['SMA_5']-df['SMA_10'])/df['SMA_10']
df['MA_Crossover'] = df['MA_Crossover'].shift(1)
df['MA_Crossover_Signal'] = df['MA_Crossover_Signal'].shift(1)
df = df.drop(columns=['EMA_Close','SMA_5','SMA_10',"MA_Crossover"])

logger.info("Initial Prep of daily candles done, downloading intradayfiles")


api_key = "REDACTED_API_KEY"
data_frame = fetch_and_concat_data(api_key)

logger.info("Intraday download done. Building proper candles")

intradaydax = data_frame[["datetime","open","high","low","close"]]
intradaydax['datetime'] = pd.to_datetime(intradaydax['datetime'])
intradaydax.set_index('datetime', inplace=True)

# Calculate the updated daily candles
intradaydax_agg = calculate_updated_daily_candles(intradaydax)

logger.info("Intraday candles done. Finalizing data preparation")

intradaydax_agg = intradaydax_agg.reset_index()

intradaydax_agg['Date'] = intradaydax_agg['datetime'].dt.date
intradaydax_agg['Date'] = pd.to_datetime(intradaydax_agg['Date'], errors='coerce')
intradaydax_agg =intradaydax_agg.rename(columns={"high": "High", "low": "Low","open": "Open","close": "Close"})

all_daily_data = []
for elem in intradaydax_agg["Date"].unique():
    all_daily_data.append(prepare_intra_day_data(elem))
all_daily_data = np.array(all_daily_data)
 

all_targets = []
for elem in intradaydax_agg["Date"].unique():
    all_targets.append(findtarget(elem))
all_targets = np.array(all_targets)


counts = np.bincount(all_targets)
num_zeros = counts[0]
num_ones = counts[1]

# Calculate proportions
total_elements = all_targets.size
proportion_zeros = num_zeros / total_elements
proportion_ones = num_ones / total_elements

# Print the results
logger.info(f"Number of 0s: {num_zeros}" )
logger.info(f"Number of 1s:  {num_ones}" )
logger.info(f"Proportion of 0s:  { proportion_zeros}")
logger.info(f"Proportion of 1s:  {proportion_ones}")

df['weekday'] = df["Date"].dt.weekday.astype(np.int8) 
df['month'] = df["Date"].dt.month.astype(np.int8) - 1
df['monthday'] = df["Date"].dt.day.astype(np.int8) -1
df['month_of_quarter'] = ((df["Date"].dt.month - 1) % 3) 
df = one_hot_encode_column(df,'weekday')
df = one_hot_encode_column(df,'month') 
df = one_hot_encode_column(df,'monthday')
df = one_hot_encode_column(df,'month_of_quarter')

columns = ['Open', 'High', 'Low',"Close",'Open_lag1', 'High_lag1', 'Low_lag1',"Close_lag1"]
columns_delete = []

for elem in columns:
    columns_delete.append(f"{elem}")
    columns_delete.append(f"{elem}_200_MA")
    columns_delete.append(f"{elem}_100_MA")
    columns_delete.append(f"{elem}_10_MA")

dataset= df.drop(columns_delete,axis=1)
dataset = dataset.iloc[:-1]
dataset = dataset.dropna()

filtered_dataset = dataset[dataset['Date'].isin(intradaydax_agg['Date'])]
data = filtered_dataset.values[:,1:]
X1 = data
X2 = all_daily_data[:,:]
y = all_targets[:]
if len(X2) > len(X1):
    X2 = X2[:len(X1)]
    y = y[:len(X1)]



# Split each array individually while keeping the same random_state to ensure matching indices
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
X2_train, X2_test, _ , _ = train_test_split(X2, y, test_size=0.2, random_state=42)  # y is just to keep the split consistent
data_train, data_test, _ , _ = train_test_split(filtered_dataset.values, y, test_size=0.2, random_state=42)  # y is just to keep the split consistent

X1_train = X1_train.astype(np.float32)
X2_train = X2_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X1_test = X1_test.astype(np.float32)
X2_test = X2_test.astype(np.float32)
y_test = y_test.astype(np.float32)
logger.info("Data preparation done. Training ML model")

np.savez('train_test_data_V2.npz', X1_train=X1_train, X2_train=X2_train, y_train=y_train, X1_test=X1_test, X2_test=X2_test, y_test=y_test,data_train=data_train,data_test=data_test)  
df.to_csv("df_V2.csv",index=False)
intradaydax_agg.to_csv("intradaydax_agg_V2.csv",index=False)

bucketmodels = "REDACTED_BUCKET"

TOTAL_ITERATIONS = 10000
LOG_INTERVAL = 100
start_time = time.time()

termination_time = df.iloc[-1]['Date'].replace(hour=8,minute=55)

if manual:
    termination_time = termination_time + timedelta(days=1)
def optimizeml(**kwargs):
    if should_terminate(termination_time):
        #pass
        raise TimeLimitExceededException("Approaching cut-off time. Terminating tuning...")
    global performance, counter

    params = kwargs
    params["performance"] = performance
    with open('params_V2.json', 'w') as json_file:
        json.dump(params, json_file)

    result = subprocess.run(['python', 'trainmodel.py'], text=False, capture_output=True)

  
    counter += 1
    if counter % LOG_INTERVAL == 0:
        elapsed_time = time.time() - start_time
        avg_speed = counter / elapsed_time * 60  # iterations per minute     
        
        logger.info(f"Completed {counter} of the iterations.")
        logger.info(f"Average speed: {avg_speed:.2f} iterations/minute.")
        
   
    with open('params_V2.json', 'r') as json_file:
        params = json.load(json_file)
    strat_performance = params["performance_new"]
    if strat_performance > performance:
        logger.info(f"New performance is {strat_performance} and it is better than the previous one {performance} ")  
        performance = strat_performance


    
    return strat_performance

optimizer = BayesianOptimization(
    f=optimizeml,
     pbounds={'x1': (1, 128),'x2': (1, 128),'x3': (1, 128),'x4': (1, 128),
 'drop1': (0, 0.5),'drop2': (0, 0.5),'drop3': (0, 0.5),'drop4': (0, 0.5),
 'leaky1': (0, 1),'leaky2': (0, 1),'leaky3': (0, 1),'leaky4': (0, 1),
 "lstm1": (5, 48),'lstm2': (5, 48),
 'epoch': (10, 200),
 'batch_size': (2, 256)},
    verbose=0,
    random_state=42,
    #bounds_transformer=bounds_transformer
)
performance = 0
counter = 0


try:
    optimizer.maximize(
        init_points=15,
        n_iter=TOTAL_ITERATIONS-15,
    )
except TimeLimitExceededException as e:
    logger.info(e)
except Exception as e:
    traceback.print_exc()