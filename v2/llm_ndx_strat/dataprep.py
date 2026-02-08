import pandas as pd
import numpy as np
 
import boto3

import requests
from io import StringIO
from datetime import datetime, timedelta
import time
import sys
import logging
import ta
from pandas.errors import SettingWithCopyWarning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
from tqdm.auto import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



def get_with_retries(base_url, params, max_retries=5, backoff_factor=1, timeout=10):
    """
    Makes a GET request with automatic retries, including handling ReadTimeout.

    Args:
        base_url (str): The base URL for the GET request.
        params (dict): The query parameters for the request.
        max_retries (int): Maximum number of retries.
        backoff_factor (float): Factor for waiting between retries (exponential backoff).
        timeout (int): Timeout for each request in seconds.

    Returns:
        Response object if successful, raises an exception otherwise.
    """
    # Configure retries
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,  # Exponential backoff: 1s, 2s, 4s, etc.
        status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET"],  # Retry only for GET requests
        raise_on_status=False  # Handle HTTP errors manually
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Create a session and mount the adapter
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Make the GET request
    try:
        response = session.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response
    except requests.exceptions.RequestException as e:
        print(f"Request ultimately failed after {max_retries} retries. Error: {e}")
        raise



def fetch_and_concat_data(api_key,start_date,end_date):
    base_url = "https://financialmodelingprep.com/api/v3/historical-chart/5min/^NDX"
    

    all_data_frames = []
    request_count = 0
    requests_per_minute = 200
    sleep_time = 60 / requests_per_minute  # Adjust sleep time to maintain rate limit
    total_days = (end_date - start_date).days

    with tqdm(total=total_days, desc="Fetching data") as pbar:
        while start_date <= end_date:
            params = {
                "apikey": api_key,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": start_date.strftime("%Y-%m-%d")
            }

            response = requests.get(base_url, params=params)
            request_count += 1

            if response.status_code == 200:
                data = response.json()
                if data:  # Check if data is not empty
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])  # Ensure the 'date' column is in datetime format
                    df = df.iloc[::-1]  # Reverse the dataframe to maintain chronological order
                    all_data_frames.append(df)
            else:
                print(f"Failed to fetch data for {start_date.strftime('%Y-%m-%d')}")

            # Sleep to maintain the rate limit
            time.sleep(sleep_time)

            start_date += timedelta(days=1)
            pbar.update(1)

    # Concatenate all data frames
    final_df = pd.concat(all_data_frames, ignore_index=True)

    # Remove duplicates based on the datetime column
    final_df.drop_duplicates(subset='date', keep='first', inplace=True)

    return final_df
# Function to calculate daily candles updated at every 5-minute interval
def calculate_updated_daily_candles(df):
    # Create an empty DataFrame for the updated daily candles
    updated_daily_candles = pd.DataFrame(columns=['open', 'high', 'low', 'close'], index=df.index)

    # Initialize tqdm progress bar
    batch_size = 100  # Update the progress bar every 100 steps
    progress_bar = tqdm(total=len(df), desc="Calculating daily candles", unit="step")

    # Iterate over the 5-minute candles
    for idx, current_time in enumerate(df.index):
        # Filter data up to the current timestamp
        current_day_data = df[df.index.date == current_time.date()]
        up_to_current_time_data = current_day_data[current_day_data.index <= current_time]

        # Calculate updated daily candle
        updated_daily_candles.loc[current_time, 'open'] = current_day_data.iloc[0]['open']
        updated_daily_candles.loc[current_time, 'high'] = up_to_current_time_data['high'].max()
        updated_daily_candles.loc[current_time, 'low'] = up_to_current_time_data['low'].min()
        updated_daily_candles.loc[current_time, 'close'] = up_to_current_time_data.iloc[-1]['close']

        # Update progress bar in batches
        if (idx + 1) % batch_size == 0 or idx == len(df) - 1:
            progress_bar.update(batch_size)

    progress_bar.close()
    return updated_daily_candles





def prepare_csvs(api_key,start_date,end_date):
    # Usage
    
    data_frame = fetch_and_concat_data(api_key,start_date,end_date)
    intradayndx = data_frame[["date","open","high","low","close"]]
    intradayndx = intradayndx.rename(columns={"date":"datetime"})
    intradayndx['datetime'] = pd.to_datetime(intradayndx['datetime'])
    intradayndx.set_index('datetime', inplace=True)
    # Calculate the updated daily candles
    intradayndx_agg = calculate_updated_daily_candles(intradayndx)
    
     
    intradayndx_agg = intradayndx_agg.reset_index()

    intradayndx_agg['Date'] = intradayndx_agg['datetime'].dt.date
    
    intradayndx_agg['Date'] = pd.to_datetime(intradayndx_agg['Date'], errors='coerce')
    
    intradayndx_agg =intradayndx_agg.rename(columns={"high": "High", "low": "Low","open": "Open","close": "Close"})
    
    daily_df = intradayndx.resample('D').agg({'open': 'first', 
                                 'high': 'max', 
                                 'low': 'min', 
                                 'close': 'last'})
    
    intradayndx = intradayndx.reset_index()
    intradayndx['Date'] = intradayndx['datetime'].dt.date
    intradayndx['Date'] = pd.to_datetime(intradayndx['Date'], errors='coerce')
    intradayndx =intradayndx.rename(columns={"high": "High", "low": "Low","open": "Open","close": "Close"})
    # Drop rows with NaN values (days where there might be no data)
    daily_df.dropna(inplace=True)
    daily_df = daily_df.reset_index()
    daily_df = daily_df.rename(columns={"high": "High", "low": "Low","open": "Open","close": "Close","datetime": "Date"})
    return daily_df,intradayndx_agg, intradayndx
    


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

def transform_daily_df(df : pd.DataFrame):
    df["Open_200_MA"] = df["Open"].rolling(window=30).mean()
    df["High_200_MA"] = df["High"].rolling(window=30).mean()
    df["Low_200_MA"] = df["Low"].rolling(window=30).mean()
    df["Close_200_MA"] = df["Close"].rolling(window=30).mean()
    df["Open_100_MA"] = df["Open"].rolling(window=14).mean()
    df["High_100_MA"] = df["High"].rolling(window=14).mean()
    df["Low_100_MA"] = df["Low"].rolling(window=14).mean()
    df["Close_100_MA"] = df["Close"].rolling(window=14).mean()
    df["Open_10_MA"] = df["Open"].rolling(window=7).mean()
    df["High_10_MA"] = df["High"].rolling(window=7).mean()
    df["Low_10_MA"] = df["Low"].rolling(window=7).mean()
    df["Close_10_MA"] = df["Close"].rolling(window=7).mean()
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

    df["Date"] = pd.to_datetime(df['Date'], errors='coerce')
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
    dataset = dataset.iloc[:]
    dataset = dataset.dropna()
    return dataset


def pricelong(indexvalue, knockoutprice) :
    """Calculate the price of long options """
    return np.maximum((indexvalue - knockoutprice) * 0.01, 0)

def priceshort(indexvalue, knockoutprice) :
    """Calculate the price of short options"""
    return np.maximum((knockoutprice - indexvalue) * 0.01, 0)


def prepare_intra_day_data(date,dfdaily,intradayndx_agg):
    """
    Prepares intra-day data for a given date.
    """
    # Load the data for the given date
    open_day = dfdaily[dfdaily["Date"] == date]["Open"]
     
    knockout_long = 0.96 * open_day
    knockout_short = 1.04 * open_day
    open_price_long = pricelong(open_day,knockout_long) 
    open_price_short = priceshort(open_day,knockout_short)
    df_intra_day = intradayndx_agg[intradayndx_agg["Date"]==date]
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



def findtarget(date,dfdaily,intradayndx_agg):
    close_day = dfdaily[dfdaily["Date"] == date]["Close"].values[0]
    df_intra_day = intradayndx_agg[intradayndx_agg["Date"] == date]

    index_value = df_intra_day.iloc[17]["Close"]

    knockout_long = 0.93 * index_value
    open_price_long = pricelong(index_value, knockout_long)
    close_price_long = pricelong(close_day, knockout_long)

    percentage_diff = ((close_price_long - open_price_long) / open_price_long) * 100

    if percentage_diff <= 0:
        return 0   
    else:
        return 1   
    
