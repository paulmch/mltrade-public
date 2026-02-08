
import json

import os
print(os.environ)


import numpy as np
import pandas as pd
import boto3
import ta

 
from decimal import Decimal
import datetime
import pytz
import logging
import sys

def current_german_time():
    utc_now = datetime.datetime.now(pytz.utc)
    berlin_tz = pytz.timezone('Europe/Berlin')
    return utc_now.astimezone(berlin_tz)
 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Function to read data from S3
def read_data_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'], parse_dates=['Date'])

# Function to write data to S3
def write_data_to_s3(bucket, key, content):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=content)
    logger.info(f"Uploaded {key} to s3://{bucket}/{key}")
def upload_file_to_s3(bucket, key, file):
    s3 = boto3.client("s3")
    s3.upload_file(file, bucket, key)
    logger.info(f"Uploaded {key} to s3://{bucket}/{key}")
def one_hot_encode_column(data: pd.DataFrame, col):
    df = data.copy()
    max_value = df[col].max()
    for i in range(0, max_value+1):
         df[col+str(i)] = np.where(df[col]==i, 1, 0)
    df = df.drop(columns=[col])
    return df

def one_hot_encode_column_with_fixed_columns(data: pd.DataFrame, col, max_classes):
    df = data.copy()
    for i in range(0, max_classes+1):
         df[col+str(i)] = np.where(df[col]==i, 1, 0)
    return df
def copy_files_between_buckets(src_bucket, dst_bucket, file_keys, new_prefix):
    s3 = boto3.client('s3')

    for key in file_keys:
        # Get the file name without any folder structure
        file_name = key.split('/')[-1]

        # Combine the new prefix with the file name
        new_key = f"{new_prefix}{file_name}"

        # Copy the file to the new location
        s3.copy_object(Bucket=dst_bucket, CopySource={'Bucket': src_bucket, 'Key': key}, Key=new_key)
        print(f"Copied {key} from {src_bucket} to {new_key} in {dst_bucket}")


# Load data from S3
bucket = "REDACTED_BUCKET"
bucketmodels = "REDACTED_BUCKET"
dayactions_key = "dayactions.csv"
dax_key = "dax.csv"

df1 = read_data_from_s3(bucket, dayactions_key)
df = read_data_from_s3(bucket, dax_key)
df['weekday'] = df["Date"].dt.weekday.astype(np.int8) 
df['monthday'] = df["Date"].dt.day.astype(np.int8) -1
df['month_of_quarter'] = ((df["Date"].dt.month - 1) % 3) 
df = one_hot_encode_column(df,'weekday')
df = one_hot_encode_column(df,'monthday')
df = one_hot_encode_column(df,'month_of_quarter')
df1 =one_hot_encode_column_with_fixed_columns(df1,"action",19)
full_df =  pd.merge(df1, df, on="Date")
def price(indexvalue : float, knockoutprice : float  ):
    return max((indexvalue - knockoutprice)* 0.01 , 0 )

prevclose = 13015.230469
prevopen = 12814.099609
prevhigh = 13019.129883
prevlow = 12766.799805

def buy(row):
     
    action = row["action"]
    if action <= 8:
        knockoutvalue = (0.89 + 0.01* action) * row["Open"]
        buyprice = price(row["Open"], knockoutvalue  )
        lowprice = price(row["Low"], knockoutvalue  )
        closeprice = price(row["Close"], knockoutvalue  )
        if lowprice <= 0.71 * buyprice:
            return 0
        return int(buyprice < closeprice)
    if action >= 10:
        knockoutvalue = (1.03 + 0.01* (   action-10)) * row["Open"]
        buyprice = price(knockoutvalue,row["Open"])
        highprice = price(knockoutvalue,row["High"])
        closeprice = price(knockoutvalue,row["Close"])
        if highprice <= 0.71 * buyprice:
            return 0
        return int(buyprice < closeprice)
def selloposite(row):
    stop = 0.71
    action = row["action"]
     
    if action <= 8:
        action = 11 + (8 - action)
         
    
    elif action >=10:
        action = 7 + (10 -action )
    
     
    if action <= 8:
        knockoutvalue = (0.89 + 0.01* action) * row["Open"]
        buyprice = price(row["Open"], knockoutvalue  )
        lowprice = price(row["Low"], knockoutvalue  )
        closeprice = price(row["Close"], knockoutvalue  )
        if lowprice <= stop * buyprice:
            return  stop
        return     1+ (closeprice-buyprice) / buyprice
    if action >= 10:
        knockoutvalue = (1.03 + 0.01* (   action-10)) * row["Open"]
        buyprice = price(knockoutvalue,row["Open"])
        highprice = price(knockoutvalue,row["High"])
        closeprice = price(knockoutvalue,row["Close"])
        if highprice <= stop * buyprice:
            return stop
        return     1+ (closeprice-buyprice) / buyprice        
        
def closeperc(row):
    global prevclose
    tmp = row["Open"]
    tmp = (tmp -prevclose)/prevclose
    prevclose = row["Close"]
    return tmp
def sell(row):
    stop = 0.71 
    action = row["action"]
    if action <= 8:
        knockoutvalue = (0.89 + 0.01* action) * row["Open"]
        buyprice = price(row["Open"], knockoutvalue  )
        lowprice = price(row["Low"], knockoutvalue  )
        closeprice = price(row["Close"], knockoutvalue  )
        if lowprice <= stop * buyprice:
            return  stop
        return     1+ (closeprice-buyprice) / buyprice
    if action >= 10:
        knockoutvalue = (1.03 + 0.01* (   action-10)) * row["Open"]
        buyprice = price(knockoutvalue,row["Open"])
        highprice = price(knockoutvalue,row["High"])
        closeprice = price(knockoutvalue,row["Close"])
        if highprice <= stop * buyprice:
            return stop
        return     1+ (closeprice-buyprice) / buyprice 
full_df["prevclose"]= full_df.apply(closeperc, axis=1)
full_df["result"]= full_df.apply(buy, axis=1)
full_df["performance"]= full_df.apply(sell, axis=1)
full_df["performance_oppossite"]= full_df.apply(selloposite, axis=1)
full_df["performancemean"] =full_df.performance.rolling(5).mean()
full_df["performance_oppossitemean"] =full_df.performance_oppossite.rolling(5).mean()
full_df['performancemean'] = full_df['performancemean'].shift(1)
full_df['performance_oppossitemean'] = full_df['performance_oppossitemean'].shift(1)
full_df["performancemean_diff"] =full_df["performancemean"].diff()
full_df["performance_oppossitemean_diff"] = full_df["performance_oppossitemean"].diff()
full_df['RSI_Close']  = ta.momentum.rsi(full_df['Close'])
full_df['RSI_Open']  =   ta.momentum.rsi(full_df['Open'])
full_df['RSI_Close'] = (full_df['RSI_Close'] - 50) / 50
full_df['RSI_Open'] = (full_df['RSI_Open'] - 50) / 50
full_df['RSI_Close_14d']  =   ta.momentum.rsi(full_df.Close.rolling(14).mean())
full_df['RSI_Close_14d'] = (full_df['RSI_Close_14d'] - 50) / 50

full_df['EMA_Close'] = ta.trend.ema_indicator(full_df['Close'])
full_df['EMA_Ratio_Close'] = (full_df['Close'] - full_df['EMA_Close']) / full_df['Close']
bollinger = ta.volatility.BollingerBands(close=full_df['Close'])
full_df['BB_Bandwidth'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / bollinger.bollinger_mavg()
full_df['BB_Percent'] = (full_df['Close'] - bollinger.bollinger_lband()) / (bollinger.bollinger_hband() - bollinger.bollinger_lband())/2
stochastic = ta.momentum.stoch(full_df['High'], full_df['Low'], full_df['Close'])
full_df['Stochastic_Scaled'] = stochastic / 100.0
full_df = full_df.drop(columns=['EMA_Close'])
full_df['RSI_Close_diff'] = full_df['RSI_Close'].diff()
full_df['RSI_Open_diff'] = full_df['RSI_Open'].diff()



last_rsi_close = full_df['RSI_Close'].iloc[-1]
last_rsi_open = full_df['RSI_Open'].iloc[-1]
last_rsi_close_14d = full_df['RSI_Close_14d'].iloc[-1]
last_rsi_close_diff = full_df['RSI_Close_diff'].iloc[-1]
last_rsi_open_diff = full_df['RSI_Open_diff'].iloc[-1]
# new ta indicators
last_ema_ratio_close = full_df['EMA_Ratio_Close'].iloc[-1]
last_bb_bandwidth = full_df['BB_Bandwidth'].iloc[-1]
last_bb_percent = full_df['BB_Percent'].iloc[-1]
last_stochastic_scaled = full_df['Stochastic_Scaled'].iloc[-1]

full_df['RSI_Open'] = full_df['RSI_Open'].shift(1)
full_df['RSI_Close'] = full_df['RSI_Close'].shift(1)
full_df['RSI_Close_14d'] = full_df['RSI_Close_14d'].shift(1)
full_df['RSI_Open_diff'] = full_df['RSI_Open_diff'].shift(1)
full_df['RSI_Close_diff'] = full_df['RSI_Close_diff'].shift(1)

full_df['EMA_Ratio_Close'] = full_df['EMA_Ratio_Close'].shift(1)
full_df['BB_Bandwidth'] = full_df['BB_Bandwidth'].shift(1)
full_df['BB_Percent'] = full_df['BB_Percent'].shift(1)
full_df['Stochastic_Scaled']  = full_df['Stochastic_Scaled'].shift(1)

full_df['SMA_5'] = ta.trend.SMAIndicator(close=full_df['Close'], window=5).sma_indicator()
full_df['SMA_10'] = ta.trend.SMAIndicator(close=full_df['Close'], window=10).sma_indicator()

# Moving Average Crossover Signal
full_df['MA_Crossover_Signal'] = 0
full_df.loc[full_df['SMA_5'] > full_df['SMA_10'], 'MA_Crossover_Signal'] = 1  # Bullish crossover
full_df.loc[full_df['SMA_5'] < full_df['SMA_10'], 'MA_Crossover_Signal'] = -1  # Bearish crossover


last_ma_crossover_signal = full_df['MA_Crossover_Signal'].iloc[-1]

full_df['MA_Crossover_Signal'] = full_df['MA_Crossover_Signal'].shift(1)

up = 1
previous = 0
def observe_market_variation(row):
    global up,previous
    if up == 1 and row["performance"] >= 1:
        previous += 1
        return previous 
    elif up == -1 and row["performance"] <= 1:
        previous += 1
        return previous  
    elif up == 1 and row["performance"] <= 1:
        up = -1
        previous -= 1
        return previous 
    elif up == -1 and row["performance"] >= 1:
        up = 1
        previous -= 1
        return previous 

full_df["market_variation"] = full_df.apply(observe_market_variation, axis=1)
full_df["market_variation_tanh"] = (full_df["market_variation"]/10).apply(np.tanh)
full_df = full_df.drop(columns=["market_variation"])

full_df["market_variation_tanh_diff"] = full_df["market_variation_tanh"].diff()



last_market_variation_tanh_diff = full_df['market_variation_tanh_diff'].iloc[-1]
last_market_variation_tanh = full_df['market_variation_tanh'].iloc[-1]

full_df["market_variation_tanh"] = full_df["market_variation_tanh"].shift(1)
full_df["market_variation_tanh_diff"] =  full_df["market_variation_tanh_diff"].shift(1)
full_df = full_df.drop(columns=['SMA_5','SMA_10'])


full_df = full_df.dropna()
full_df["performancemean"] = full_df["performancemean"].apply(lambda x : x-1)
full_df["performance_oppossitemean"] = full_df["performance_oppossitemean"].apply(lambda x : x-1)
 


dataset = full_df.drop(columns=['Date','Open',"High","Low","Close"])
dataset["direction"] = (dataset["action"] <= 8).astype(int) 
value_data = full_df.iloc[49:][['Date',"action","performance","performance_oppossite"]]
dataset =dataset.drop(columns=['action'])
data = dataset.values
data = np.delete(data, 179, axis=0) ## at least 5 sigma deviation day, will not yield anything to be used in the training
X_indices =  np.r_[0:62, 65:79+2]
X = data[:, X_indices]
y_indices = np.r_[62:65, 79:80+2]
y = data[:, y_indices]
new_df= pd.DataFrame({"index" : np.array([i for i in range(0,len(y))]),"one": y[:,1],"two": y[:,2]})
new_df["onemean"] =new_df.one.rolling(5).mean()
new_df["twomean"] =new_df.two.rolling(5).mean()
new_df = new_df.dropna()
new_df["onemean"] = new_df["onemean"].apply(lambda x : x-1)
new_df["twomean"] = new_df["twomean"].apply(lambda x : x-1)
new_df["onediff"] = new_df["onemean"].diff()
new_df["twodiff"] = new_df["twomean"].diff()
last_mean_values = new_df.values[len(new_df.values)-1][3:7]

a = 60
dataset = full_df.iloc[a:].drop(columns=['Date','Open',"High","Low","Close","action","performance","performance_oppossite"])
value_data = full_df.iloc[a:][['Date',"performance","performance_oppossite"]]
data = dataset.values
 


def put_value_in_dynamodb(table_name, primary_key_name, primary_key_value, value):
    # Create a DynamoDB client
    dynamodb = boto3.resource('dynamodb')
    decimal_value = Decimal(str(value))
    # Get the table
    table = dynamodb.Table(table_name)

    # Create the item to put in the table
    item = {
        primary_key_name: primary_key_value,
        'value': decimal_value
    }

    # Put the item in the table
    response = table.put_item(Item=item)

    return response

sns_client = boto3.client("sns", region_name="eu-west-1")
sns_topic_arn = "arn:aws:sns:eu-west-1:REDACTED_AWS_ACCOUNT_ID:ig-messages"
def publish_sns_message(message, subject="Hyperparametertuning"):
    try:
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Message=json.dumps(message, ensure_ascii=False),
            Subject=subject
        )
    except Exception as e:
        logger.error(f"Failed to publish SNS message: {str(e)}")


# Example usage
table_name = 'turnarount'
primary_key_name = 'val'
primary_key_value = 'turnaroundday'



val1 =last_mean_values[0]
val2 =last_mean_values[1]
val1shift =last_mean_values[2]
val2shift =last_mean_values[3]
response = put_value_in_dynamodb(table_name, primary_key_name,"performancemean", val1)
response = put_value_in_dynamodb(table_name, primary_key_name, "performance_oppossitemean", val2)
response = put_value_in_dynamodb(table_name, primary_key_name,"performancemean_diff", val1shift)
response = put_value_in_dynamodb(table_name, primary_key_name, "performance_oppossitemean_diff", val2shift)
response = put_value_in_dynamodb(table_name, primary_key_name,"RSI_Close", last_rsi_close)
response = put_value_in_dynamodb(table_name, primary_key_name, "RSI_Open", last_rsi_open)
response = put_value_in_dynamodb(table_name, primary_key_name,"RSI_Close_diff", last_rsi_close_diff)
response = put_value_in_dynamodb(table_name, primary_key_name, "RSI_Open_diff", last_rsi_open_diff)



response = put_value_in_dynamodb(table_name, primary_key_name, "EMA_Ratio_Close", last_ema_ratio_close)
response = put_value_in_dynamodb(table_name, primary_key_name,"BB_Bandwidth", last_bb_bandwidth)
response = put_value_in_dynamodb(table_name, primary_key_name, "BB_Percent", last_bb_percent)
response = put_value_in_dynamodb(table_name, primary_key_name, "Stochastic_Scaled", last_stochastic_scaled)
response = put_value_in_dynamodb(table_name, primary_key_name, "MA_Crossover_Signal", last_ma_crossover_signal)
 
response = put_value_in_dynamodb(table_name, primary_key_name, "Market_Variation_Tanh", last_market_variation_tanh)
response = put_value_in_dynamodb(table_name, primary_key_name, "Market_Variation_Tanh_Diff", last_market_variation_tanh_diff)
 
 
 