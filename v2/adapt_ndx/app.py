 
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
 
 
   
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

bucket = "REDACTED_BUCKET"
dax_key = "ndx.csv"
def read_data_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'], parse_dates=['Date'])

def fetch_and_concat_data(api_key,date):
    base_url = "https://api.twelvedata.com/time_series"
    #start_date = datetime(year=2023, month=12, day=28)
    start_date = date
      

    all_data_frames = []
     

     
    params = {
        "symbol": "NDX",
        "interval": "5min",
        "format": "CSV",
        "apikey": api_key,
        "start_date": start_date.strftime("%m/%d/%Y 9:00"),
        "end_date": (start_date ).strftime("%m/%d/%Y 17:31")
        }

    response = requests.get(base_url, params=params)
        


    if response.status_code == 200:
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, delimiter=';')
        df = df.iloc[::-1]
        all_data_frames.append(df)
    else:
        print(f"Failed to fetch data for {start_date.strftime('%Y-%m-%d')}")

       

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

def prepare_intra_day_data(date,df , intradayndx_agg):
    """
    Prepares intra-day data for a given date.
    """
    # Load the data for the given date
    open_day = df[df["Date"] == date]["Open"]
    knockout_long = 0.96 * open_day
    knockout_short = 1.04 * open_day
    open_price_long = pricelong(open_day,knockout_long) 
    open_price_short = priceshort(open_day,knockout_short)
    df_intra_day = intradayndx_agg[intradayndx_agg["Date"]==date]
    temp_list = []
    for element in df_intra_day.iloc[0:20].iterrows():
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



 
api_key = "REDACTED_API_KEY"
 

def find_atr(date,intradayndx_agg):
    # Convert the relevant columns to numeric, forcing any errors to be NaN
     
    df_intra_day = intradayndx_agg[intradayndx_agg["Date"]==date]
     
    df_intra_day['Previous Close'] = df_intra_day['Close'].shift(1)
    df_intra_day['High'] = pd.to_numeric(df_intra_day['High'], errors='coerce')
    df_intra_day['Low'] = pd.to_numeric(df_intra_day['Low'], errors='coerce')
    df_intra_day['Close'] = pd.to_numeric(df_intra_day['Close'], errors='coerce')
    df_intra_day['Previous Close'] = pd.to_numeric(df_intra_day['Previous Close'], errors='coerce')


    df_intra_day['TR'] = df_intra_day[['High', 'Low', 'Previous Close']].apply(
        lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['Previous Close']), abs(x['Low'] - x['Previous Close'])), axis=1
    )

    # Calculate ATR for the first 17 values
    atr_period = 17
    df_intra_day['ATR'] = df_intra_day['TR'].rolling(window=atr_period).mean()

    # Extract the first 17 rows
    df_intra_day_17 = df_intra_day.iloc[0:17][['Date', 'High', 'Low', 'Close', 'TR', 'ATR']]
    
    return df_intra_day_17.iloc[16]["ATR"]

def handler(event, context):
    

    df = read_data_from_s3(bucket, dax_key)
    current_date = datetime.now().date() 
    s3_client = boto3.client('s3')
     
    bucket_name = 'REDACTED_BUCKET'
    file_name = 'performance_by_atr_bin_direction.pkl'
    s3_key = f'{file_name}'
    s3_client.download_file(bucket_name, s3_key,"/tmp/"+ file_name)

# Load the DataFrame from the downloaded pickle file
    performance_by_atr_bin_direction = pd.read_pickle("/tmp/"+ file_name)
    
    data_frame = fetch_and_concat_data(api_key,current_date)

    
    last_row = df.iloc[-1]
    open_row = data_frame.values[0,1]
    high_value = last_row['High']
    low_value = last_row['Low']
    close_value = last_row['Close']
    new_row = {'Date': current_date, 'Open': open_row, 'High': high_value, 'Low': low_value, 'Close': close_value}
   
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True) 
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') 
    df = df.dropna()

    df["Open_30_MA"] = df["Open"].rolling(window=30).mean()
    df["High_30_MA"] = df["High"].rolling(window=30).mean()  
    df["Low_30_MA"] = df["Low"].rolling(window=30).mean()
    df["Close_30_MA"] = df["Close"].rolling(window=30).mean()
    df["Open_14_MA"] = df["Open"].rolling(window=14).mean()
    df["High_14_MA"] = df["High"].rolling(window=14).mean()
    df["Low_14_MA"] = df["Low"].rolling(window=14).mean()
    df["Close_14_MA"] = df["Close"].rolling(window=14).mean()
    df["Open_7_MA"] = df["Open"].rolling(window=7).mean()
    df["High_7_MA"] = df["High"].rolling(window=7).mean()
    df["Low_7_MA"] = df["Low"].rolling(window=7).mean()
    df["Close_7_MA"] = df["Close"].rolling(window=7).mean()
    df = df.dropna()
    df['RSI_open'] = ta.momentum.rsi(df['Open'])
    df['RSI_open'] = (df['RSI_open'] - 50) / 50
    df['RSI_diff_open'] = df['RSI_open'].diff()

    df['Close_lag1'] = df['Close'].shift(1)
    df['Open_lag1'] = df['Open'].shift(1)
    df['Low_lag1'] = df['Low'].shift(1)
    df['High_lag1'] = df['High'].shift(1)

    df['Close_lag1_30_MA'] = df['Close_30_MA'].shift(1)
    df['Open_lag1_30_MA'] = df['Open_30_MA'].shift(1)
    df['Low_lag1_30_MA'] = df['Low_30_MA'].shift(1)
    df['High_lag1_30_MA'] = df['High_30_MA'].shift(1)

    df['Close_lag1_14_MA'] = df['Close_14_MA'].shift(1)
    df['Open_lag1_14_MA'] = df['Open_14_MA'].shift(1)
    df['Low_lag1_14_MA'] = df['Low_14_MA'].shift(1)
    df['High_lag1_14_MA'] = df['High_14_MA'].shift(1)

    df['Close_lag1_7_MA'] = df['Close_7_MA'].shift(1)
    df['Open_lag1_7_MA'] = df['Open_7_MA'].shift(1)
    df['Low_lag1_7_MA'] = df['Low_7_MA'].shift(1)
    df['High_lag1_7_MA'] = df['High_7_MA'].shift(1)

    df["Open_Close"] = (df["Open"]- df["Close_lag1"]) / df["Open"]
    df["Open_Open"] = (df["Open"]- df["Open_lag1"]) / df["Open"]
    df["Open_Low"] = (df["Open"]- df["Low_lag1"]) / df["Open"]
    df["Open_High"] = (df["Open"]- df["High_lag1"]) / df["Open"]

    df["Open_Close_30_MA"] = (df["Open"]- df["Close_lag1_30_MA"]) / df["Open"]
    df["Open_Open_30_MA"] = (df["Open"]- df["Open_lag1_30_MA"]) / df["Open"]
    df["Open_Low_30_MA"] = (df["Open"]- df["Low_lag1_30_MA"]) / df["Open"]
    df["Open_High_30_MA"] = (df["Open"]- df["High_lag1_30_MA"]) / df["Open"]

    df["Open_Close_14_MA"] = (df["Open"]- df["Close_lag1_14_MA"]) / df["Open"]
    df["Open_Open_14_MA"] = (df["Open"]- df["Open_lag1_14_MA"]) / df["Open"]
    df["Open_Low_14_MA"] = (df["Open"]- df["Low_lag1_14_MA"]) / df["Open"]
    df["Open_High_14_MA"] = (df["Open"]- df["High_lag1_14_MA"]) / df["Open"]

    df["Open_Close_7_MA"] = (df["Open"]- df["Close_lag1_7_MA"]) / df["Open"]
    df["Open_Open_7_MA"] = (df["Open"]- df["Open_lag1_7_MA"]) / df["Open"]
    df["Open_Low_7_MA"] = (df["Open"]- df["Low_lag1_7_MA"]) / df["Open"]
    df["Open_High_7_MA"] = (df["Open"]- df["High_lag1_7_MA"]) / df["Open"]
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
        columns_delete.append(f"{elem}_30_MA")
        columns_delete.append(f"{elem}_14_MA")
        columns_delete.append(f"{elem}_7_MA")

    dataset= df.drop(columns_delete,axis=1)
    dataset = dataset.dropna()
    last_row_array = dataset.iloc[-1].values[1:]
    last_row_array = last_row_array.astype(np.float32)



    intradayndx = data_frame[["datetime","open","high","low","close"]]
    intradayndx['datetime'] = pd.to_datetime(intradayndx['datetime'])
    intradayndx.set_index('datetime', inplace=True)
    intradayndx_agg = calculate_updated_daily_candles(intradayndx)
    intradayndx_agg = intradayndx_agg.reset_index()
    intradayndx_agg['Date'] = intradayndx_agg['datetime'].dt.date
    intradayndx_agg['Date'] = pd.to_datetime(intradayndx_agg['Date'], errors='coerce')
    intradayndx_agg =intradayndx_agg.rename(columns={"high": "High", "low": "Low","open": "Open","close": "Close"})
    direction = int(event["position"][0] == "Long")
    atr = find_atr(current_date.strftime("%Y-%m-%d"),intradayndx_agg)
     
        
    index_value = intradayndx_agg.iloc[17]["Close"]
    print(atr)
    print(performance_by_atr_bin_direction[atr][0])
    knockout_long = 0.96 * index_value
    knockout_short = 1.04 * index_value
    long_move =  {
  "action": "BUY",
  "position": [
    "Long",
    knockout_long
  ],
  "original_direction" : direction,
  "atr" : atr,
  "account": "Turbo24"
}
    short_move = {
  "action": "BUY",
  "position": [
    "Short",
    knockout_short
  ],
  "original_direction" : direction,
  "atr" : atr,
  "account": "Turbo24"
}
    zero_move = {
    "action": "NOTHING",
  "position": [
     
  ],
  "original_direction" : direction,
  "atr" : atr,
  "account": "Turbo24"

    }
    try:
        if direction == 1 and not np.isnan(performance_by_atr_bin_direction[atr][1]) and performance_by_atr_bin_direction[atr][1] > 1 or (direction ==0 and np.isnan(performance_by_atr_bin_direction[atr][0])) or (direction == 0 and performance_by_atr_bin_direction[atr][0] < 1) :
            print(f"{atr},Long")
            return long_move
        elif (direction == 0 and not np.isnan(performance_by_atr_bin_direction[atr][0]) and performance_by_atr_bin_direction[atr][0] > 1) or (direction == 1 and np.isnan(performance_by_atr_bin_direction[atr][1])) or (direction == 1 and performance_by_atr_bin_direction[atr][1] < 1):
            print(f"{atr},Short")
            return short_move
        else:
            return zero_move
    except:
        return zero_move




    
    return  {"fail" :1}



    


   

    


  


   





 