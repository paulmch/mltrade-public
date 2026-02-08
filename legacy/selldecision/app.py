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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow import keras
import tensorflow as tf
from bayes_opt import BayesianOptimization
from decimal import Decimal
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
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



# 1. RSI (Relative Strength Index)
data['RSI'] = ta.momentum.RSIIndicator(close=data['Last'], window=5).rsi()

# 2. Moving Averages
data['SMA_5'] = ta.trend.SMAIndicator(close=data['Last'], window=5).sma_indicator()
data['SMA_10'] = ta.trend.SMAIndicator(close=data['Last'], window=10).sma_indicator()

# Moving Average Crossover Signal
data['MA_Crossover_Signal'] = 0
data.loc[data['SMA_5'] > data['SMA_10'], 'MA_Crossover_Signal'] = 1  # Bullish crossover
data.loc[data['SMA_5'] < data['SMA_10'], 'MA_Crossover_Signal'] = -1  # Bearish crossover

# 3. MACD (Moving Average Convergence Divergence)
macd = ta.trend.MACD(close=data['Last'], window_slow=10, window_fast=5)
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()
data['MACD_Histogram'] = macd.macd_diff()

# 4. Bollinger Bands
bollinger = ta.volatility.BollingerBands(close=data['Last'], window=10, window_dev=5)
data['Bollinger_HBand'] = bollinger.bollinger_hband()
data['Bollinger_LBand'] = bollinger.bollinger_lband()
data['Bollinger_MBand'] = bollinger.bollinger_mavg()
data['Bollinger_PctB'] = bollinger.bollinger_pband()
data['Bollinger_Width'] = bollinger.bollinger_wband()

def sell_long(row):
    # Convert to int by casting instead of using astype, since we are dealing with individual elements of the row
    good_path = (row["Last"]> row["Open"]) & (row["Open"] < row["Close"]) & (int(row["High"]) == int(row["High_final"]))
 
    bad_path = (row["Last"] > row["Open"]) & (row["Open"] > row["Close"]) & (int(row["High"]) == int(row["High_final"]))

    sell_asap = (row["Last"] < row["Open"]) & (row["Open"] > row["Close"]) & (int(row["High"]) == int(row["High_final"]))
    #print(f"{datetime}:{good_path} or {bad_path} and {not sell_asap }")
    # The logical conditions need to be carefully structured to evaluate as you expect
    if (good_path or bad_path) and not sell_asap:
        return 1
    elif not good_path and not bad_path and not sell_asap:
        return 0
    else: 
        return 1
    
def sell_short(row):
    # Convert to int by casting instead of using astype, since we are dealing with individual elements of the row
    good_path = (row["Last"] < row["Open"]) & (row["Open"] > row["Close"]) & (int(row["Low"]) == int(row["Low_final"]))
    bad_path = (row["Last"] < row["Open"]) & (row["Open"] < row["Close"]) & (int(row["Low"]) == int(row["Low_final"]))
    sell_asap = (row["Last"] > row["Open"]) & (row["Open"] < row["Close"]) & (int(row["Low"]) == int(row["Low_final"]))
    #print(f"{row['datetime']}:{good_path} or {bad_path} and {not sell_asap }")
    # The logical conditions need to be carefully structured to evaluate as you expect
    if (good_path or bad_path) and not sell_asap:
        return 1
    elif not good_path and not bad_path and not sell_asap:
        return 0
    else: 
        return 1
    
data['sell_Short'] = data.apply(sell_short, axis=1)
data['sell_Long'] = data.apply(sell_long, axis=1)
merged_data['sell_Short'] = merged_data.groupby(merged_data['Date'].dt.date)['sell_Short'].cummax()
merged_data['sell_Long'] = merged_data.groupby(merged_data['Date'].dt.date)['sell_Long'].cummax()
# Dropping rows with missing values
cleaned_data = data.dropna()

def time_series_train_test_split(features, target, date_column, test_size=0.2, random_state=None):
    # Ensure the date column is in datetime format
  

    # Get unique dates and shuffle them
    unique_dates = features[date_column].unique()
    np.random.seed(random_state)  # Set the random state for reproducibility
    np.random.shuffle(unique_dates)

    # Split dates for training and testing
    split_index = int(len(unique_dates) * (1 - test_size))
    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    # Create train and test sets
    train_mask = features[date_column].isin(train_dates)
    test_mask = features[date_column].isin(test_dates)
 
    X_train = features[train_mask]#.drop(columns=[date_column])
    y_train = target[train_mask].drop(columns=[date_column])
    X_test = features[test_mask]#.drop(columns=[date_column])
    y_test = target[test_mask].drop(columns=[date_column])

    return X_train, X_test, y_train, y_test


direction = os.environ["direction"]

features = cleaned_data[['datetime','Date','RSI', 'MACD', 'MACD_Signal', 'Bollinger_PctB', 'Bollinger_Width',"upval","downval",f"Option_{direction}_Open",f"Option_{direction}_Last"]]
features["pl"] = (features[f"Option_{direction}_Last"] -features[f"Option_{direction}_Open"])/features[f"Option_{direction}_Open"]  
features["MACD"] =features["MACD"] / 50
features["RSI"] =features["RSI"] / 100
features["Bollinger_PctB"] =features["Bollinger_PctB"]  
features["Bollinger_Width"] =features["Bollinger_Width"] / 5


features =features.drop(columns=[f"Option_{direction}_Open",f"Option_{direction}_Last"])
target = cleaned_data[['Date',f'sell_{direction}']]
# Splitting the data into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = time_series_train_test_split(features, target,'Date', test_size=0.2, random_state=42)
X_test.to_csv(f'test{direction}.csv', index=False)
start_time = time.time()
TOTAL_ITERATIONS = 1500
LOG_INTERVAL = 100

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
def upload_file_to_s3(bucket, key, file):
    s3 = boto3.client("s3")
    s3.upload_file(file, bucket, key)
    logger.info(f"Uploaded {key} to s3://{bucket}/{key}")

def write_data_to_s3(bucket, key, content):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=content)
    logger.info(f"Uploaded {key} to s3://{bucket}/{key}")
table_name = 'turnarount'
primary_key_name = 'val'
primary_key_value = 'turnaroundday'

logger.info(f"Dataprep ready starting ML training")


bucketmodels = "REDACTED_BUCKET"
upload_file_to_s3(bucketmodels, f"intraday/test{direction}.csv", f'test{direction}.csv')

def evaluate_day(day, proba,test_model):
    testcase = X_test[X_test["Date"] == day]
    features = testcase
    
     
    
    X =features.values[:,2:].astype(np.float32)
    y_pred_class = (test_model.predict(X,verbose =0) > 0.5).astype("int32")
    last = 1
    for elem in features["pl"].values:
        if elem < 0.71-1:
            last = 1 +  0.71-1
            break
        else:
            last = 1 + elem
    counter = 0
    pred = 1
    for i, elem in enumerate(y_pred_class[:,0]):
        if elem ==0:
            counter = 0
        else:
            counter += 1
            if (1-proba)**counter < 0.05:
                try:
                    pred =1+ features["pl"].values[i+3]
                except:
                    pred =1+ features["pl"].values[len(features["pl"].values)-1]
                    
                break
    
    if (1-proba)**counter > 0.05:
        pred =1+ features["pl"].values[len(features["pl"].values)-1]
                
        
        
            
    return last,pred


def optimizeml(**kwargs):
    
    tf.keras.backend.clear_session()
    global performance, prediction, counter, bestmodel
    p1 = int(kwargs["x1"])
    drop1 = kwargs["drop1"]
    l1 = kwargs["leaky1"]
   
    p2 = int(kwargs["x2"])
    l2 = kwargs["leaky2"]
    drop2 = kwargs["drop2"]
    
    p3 = int(kwargs["x3"])
    l3 = kwargs["leaky3"]
    drop3 = kwargs["drop3"]
    
    p4 = int(kwargs["x4"])
    l4 = kwargs["leaky4"]
    drop4 = kwargs["drop4"]
    
    
    epoch =int(kwargs["epoch"])
    batch_size =int(kwargs["batch_size"])
    
    input_model = Input(shape= X_train.values[:,2:].shape[1:3])
    denseone = Dense(p1,)(input_model)
    denseone = LeakyReLU(alpha=l1)(denseone)
    denseone = Dropout(rate=drop1)(denseone)
    denseone= BatchNormalization()(denseone)
    
    densetwo = Dense(p2,)(denseone)
    densetwo = LeakyReLU(alpha=l2)(densetwo)
    densetwo = Dropout(rate=drop2)(densetwo)
    densetwo= BatchNormalization()(densetwo)

    densethree = Dense(p3,)(densetwo)
    densethree = LeakyReLU(alpha=l3)(densethree)
    densethree = Dropout(rate=drop3)(densethree)
    densethree= BatchNormalization()(densethree)

    densefour = Dense(p4,)(densethree)
    densefour = LeakyReLU(alpha=l4)(densefour)
    densefour = Dropout(rate=drop4)(densefour)
    densefour= BatchNormalization()(densefour)
    final = Dense(1, name='Dense_final', activation='sigmoid')(densefour)
    model = Model(inputs=[input_model], outputs=final)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=Adam())
    
    
    history =model.fit(X_train.values[:,2:].astype(np.float32),y_train.values[:,0].astype(np.float32),epochs=epoch,validation_data=(X_test.values[:,2:].astype(np.float32), y_test.values[:,0].astype(np.float32)),batch_size=batch_size,verbose=0)
    loss = history.history["val_loss"]
    last_val_loss = loss[len(loss)-1]
    y_pred = model.predict(X_test.values[:,2:].astype(np.float32),verbose=0)
    y_pred_class = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = float(auc(fpr, tpr))
    report = classification_report(y_test, y_pred_class, output_dict=True)
    proba = report["1"]["precision"]
    counter += 1
    strategy_eval = []
    for elem in X_test["Date"].unique():
        last , pred = evaluate_day(elem,proba,model)
        strategy_eval.append([last,pred])
    strategy_eval = np.array(strategy_eval)

    last = 1
    strat = 1
    for elem in strategy_eval:
        last *= elem[0]
        strat *= elem[1]
    
    if strat > performance:
        performance = strat
        model.save_weights(f"daymodel{direction}.h5f", overwrite=True)
        upload_file_to_s3(bucketmodels,f"confidencemodelday/daymodel{direction}.h5f.data-00000-of-00001", f"daymodel{direction}.h5f.data-00000-of-00001")
        upload_file_to_s3(bucketmodels,f"confidencemodelday/daymodel{direction}.h5f.index",f"daymodel{direction}.h5f.index")
        write_data_to_s3(bucketmodels, f"confidencemodelday/daymodel{direction}.csv", json.dumps(kwargs ))
        
        response = put_value_in_dynamodb(table_name, primary_key_name, f"proba_{direction}", proba)
        logger.info("{}:{}||{},{},{},{}".format(counter,performance,p1,p2,p3,p4))
    if counter % LOG_INTERVAL == 0:
        elapsed_time = time.time() - start_time
        avg_speed = counter / elapsed_time * 60  # iterations per minute
        
        remaining_iterations = TOTAL_ITERATIONS - counter
        estimated_completion_time = remaining_iterations / avg_speed  /60
        
        logger.info(f"Completed {counter/TOTAL_ITERATIONS*100}% of the iterations.")
        logger.info(f"Average speed: {avg_speed:.2f} iterations/minute.")
        logger.info(f"Estimated time to completion: {estimated_completion_time:.2f} hours.")
    
    return strat

 
optimizer = BayesianOptimization(
    f=optimizeml,
     pbounds={'x1': (1, 128),'x2': (1, 128),'x3': (1, 128),'x4': (1, 128),
 'drop1': (0, 0.5),'drop2': (0, 0.5),'drop3': (0, 0.5),'drop4': (0, 0.5),
 'leaky1': (0, 1),'leaky2': (0, 1),'leaky3': (0, 1),'leaky4': (0, 1),
 'epoch': (10, 200),
 'batch_size': (2, 256)},
    verbose=0,
    random_state=42,
    #bounds_transformer=bounds_transformer
)
performance = 0
counter = 0
optimizer.maximize(
    init_points=5,
    n_iter=TOTAL_ITERATIONS-5,
)
 
    
 
 