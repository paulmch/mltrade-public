import boto3
import json
import pandas as pd
s3 = boto3.resource('s3')
bucket_name = 'REDACTED_BUCKET'
bucket = s3.Bucket(bucket_name)
client = boto3.client("s3")
import pandas as pd
import numpy as np
import ta
import datetime
import requests
from io import StringIO
from datetime import datetime, timedelta
import time
import  os


from decimal import Decimal

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def fetch_and_concat_data(api_key):
    base_url = "https://api.twelvedata.com/time_series"
    start_date = datetime(year=2022, month=7, day=10)
    #start_date = datetime(year=2022, month=6, day=19)
    end_date = datetime.now()  # Or any other end date you want

    all_data_frames = []
    request_count = 0

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
        if request_count >= 55:
            time.sleep(60)  # Sleep for 60 seconds
            request_count = 0  # Reset request count

        start_date += timedelta(days=1)

    # Concatenate all data frames
    final_df = pd.concat(all_data_frames, ignore_index=True)

    # Remove duplicates based on the datetime column
    final_df.drop_duplicates(subset='datetime', keep='first', inplace=True)

    return final_df

# Usage
api_key = "REDACTED_API_KEY"
data_frame = fetch_and_concat_data(api_key)
logger.info(f"Got all data from API")
intradaydax = data_frame[["datetime","open","high","low","close"]]
intradaydax['datetime'] = pd.to_datetime(intradaydax['datetime'])
intradaydax.set_index('datetime', inplace=True)

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

# Calculate the updated daily candles
intradaydax_agg = calculate_updated_daily_candles(intradaydax)

daily_df = intradaydax.resample('D').agg({'open': 'first', 
                                 'high': 'max', 
                                 'low': 'min', 
                                 'close': 'last'})

# Drop rows with NaN values (days where there might be no data)
daily_df.dropna(inplace=True)

bucket = "REDACTED_BUCKET"
bucketmodels = "REDACTED_BUCKET"
dayactions_key = "dayactions.csv"
dax_key = "dax.csv"
def read_data_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'], parse_dates=['Date'])
actions = read_data_from_s3(bucket, dayactions_key)

daily_df =daily_df.rename(columns={"open": "Open","high": "High_final", "low": "Low_final", "close":"Close"})
intradaydax_agg = intradaydax_agg.rename(columns={"open": "Open","high": "High", "low": "Low", "close":"Last"})
daily_df = daily_df.reset_index()
intradaydax_agg = intradaydax_agg.reset_index()


actions['Date'] = pd.to_datetime(actions['Date'], errors='coerce')


intradaydax_agg['Date'] = intradaydax_agg['datetime'].dt.date
daily_df['Date'] = daily_df['datetime'].dt.date
# Convert 'Date' column in intradaydax to datetime64[ns]
intradaydax_agg['Date'] = pd.to_datetime(intradaydax_agg['Date'], errors='coerce')
daily_df['Date'] = pd.to_datetime(daily_df['Date'], errors='coerce')
# Merge the data again
merged_data = pd.merge(intradaydax_agg, daily_df[['Date', 'Close',"High_final","Low_final"]], on='Date', how='left')
merged_data = pd.merge(merged_data, actions, on='Date', how='left')

def pricelong(indexvalues: pd.Series, knockoutprices: pd.Series) -> pd.Series:
    """Calculate the price of long options for a Series of index values and knockout prices."""
    return np.maximum((indexvalues - knockoutprices) * 0.01, 0)

def priceshort(indexvalues: pd.Series, knockoutprices: pd.Series) -> pd.Series:
    """Calculate the price of short options for a Series of index values and knockout prices."""
    return np.maximum((knockoutprices - indexvalues) * 0.01, 0)

merged_data =merged_data.rename(columns={"high": "High", "low": "Low","open": "Low"})


merged_data['Knockout_Long'] = 0.97 * merged_data['Open']
merged_data['Knockout_Short'] = 1.03 * merged_data['Open']
 

merged_data['Option_Short_Open'] = priceshort(merged_data['Open'], merged_data['Knockout_Short'])
merged_data['Option_Long_Open'] = pricelong(merged_data['Open'], merged_data['Knockout_Long'])



merged_data['Option_Long_High'] = pricelong(merged_data['High'], merged_data['Knockout_Long'])
merged_data['Option_Short_High'] = priceshort(merged_data['High'], merged_data['Knockout_Short'])

merged_data['Option_Long_Low'] = pricelong(merged_data['Low'], merged_data['Knockout_Long'])
merged_data['Option_Short_Low'] = priceshort(merged_data['Low'], merged_data['Knockout_Short'])

merged_data['Option_Long_Last'] = pricelong(merged_data['Last'], merged_data['Knockout_Long'])
merged_data['Option_Short_Last'] = priceshort(merged_data['Last'], merged_data['Knockout_Short'])

merged_data['Option_Long_Close'] = pricelong(merged_data['Close'], merged_data['Knockout_Long'])
merged_data['Option_Short_Close'] = priceshort(merged_data['Close'], merged_data['Knockout_Short'])

data = merged_data

data.to_csv('merged_data.csv', index=False)

def upload_file_to_s3(bucket, key, file):
    s3 = boto3.client("s3")
    s3.upload_file(file, bucket, key)
    logger.info(f"Uploaded {key} to s3://{bucket}/{key}")

bucketmodels = "REDACTED_BUCKET"
upload_file_to_s3(bucketmodels, "intraday/merged_data.csv", 'merged_data.csv')