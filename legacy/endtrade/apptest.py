import pandas as pd
import numpy as np
import requests
import json
import boto3
import ta
from datetime import datetime, timedelta
from io import StringIO
import time
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger() 
db = boto3.resource("dynamodb")
table = db.Table("turnarount")
bucket = "REDACTED_BUCKET"
merged_data = "intraday/merged_data.csv"

def read_data_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'], parse_dates=['Date'])

def fetch_and_concat_data(api_key):
    base_url = "https://api.twelvedata.com/time_series"
    start_date = datetime.now()-timedelta(days=1)
     
    end_date = datetime.now()  # Or any other end date you want

    all_data_frames = []
    request_count = 0

    while start_date < end_date:
        params = {
            "symbol": "GDAXI",
            "interval": "5min",
            "format": "CSV",
            "apikey": api_key,
            "start_date": start_date.strftime("%m/%d/%Y 8:00"),
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
            logger.info(f"Failed to fetch data for {start_date.strftime('%Y-%m-%d')}")

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
api_key = "REDACTED_API_KEY"

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
s3 = boto3.client("s3")

def pricelong(indexvalues: pd.Series, knockoutprices: pd.Series) -> pd.Series:
    """Calculate the price of long options for a Series of index values and knockout prices."""
    return np.maximum((indexvalues - knockoutprices) * 0.01, 0)

def priceshort(indexvalues: pd.Series, knockoutprices: pd.Series) -> pd.Series:
    """Calculate the price of short options for a Series of index values and knockout prices."""
    return np.maximum((knockoutprices - indexvalues) * 0.01, 0)


def load_model(kwargs):
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
    
    
    input_model = Input(shape= X.shape[1:3])
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
    model.load_weights(f"daymodel{direction}.h5f")
    return model
def handler(event, context):
    invest =  int(table.get_item( Key={'val': "invest"})["Item"]["value"])
    if invest == 0:
        logger.info("Already sold. terminating")
        return {
            "result" : 0
        }
    previous_day = read_data_from_s3(bucket, merged_data)
    data_frame = fetch_and_concat_data(api_key)
    if data_frame.empty:
        logger.info("Data frame is empty. Terminating the program.")
        return {
            "result" : 0
        }
    intradaydax = data_frame[["datetime","open","high","low","close"]]
    intradaydax['datetime'] = pd.to_datetime(intradaydax['datetime'])
    intradaydax.set_index('datetime', inplace=True)
    intradaydax_agg = calculate_updated_daily_candles(intradaydax)

    daily_df = intradaydax.resample('D').agg({'open': 'first', 
                                 'high': 'max', 
                                 'low': 'min', 
                                 'close': 'last'})

    # Drop rows with NaN values (days where there might be no data)
    daily_df.dropna(inplace=True)
    bucket = "REDACTED_BUCKET"
    date_prefix = datetime.now().strftime("confidence/%Y-%m-%d")  # Adjust the date format as needed
    response = s3.list_objects_v2(Bucket=bucket, Prefix=date_prefix)

    # Filter to find the file for the specific day
    files = response.get('Contents', [])

    key = files[0]["Key"]
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    jsonfile = obj["Body"].read()
    jsonfile =json.loads(jsonfile.decode("utf-8"))
    downval = jsonfile['confidence']['Points down until Strategy change']/ 250
    upval = jsonfile['confidence']['Points up until Strategy change']/ 250
    date = datetime.strptime(jsonfile["date"], '%Y-%m-%d %H:%M:%S.%f')
    action = jsonfile["action"] 
    data = {"Date": np.array([date]), "upval": np.array([upval]), "downval" : np.array([downval]), "action" : np.array([action]) }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    actions = df
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

    data = pd.concat([previous_day, merged_data])
    data.drop_duplicates()
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
    db = boto3.resource("dynamodb")
    table = db.Table("turnarount")
    direction =  str(table.get_item( Key={'val': "Direction"})["Item"]["value"])
    specific_date =  datetime.now().strftime("%Y-%m-%d")
    filtered_data = data[data['Date'] == specific_date]
    features = filtered_data[['datetime','Date','RSI',  'MACD', 'MACD_Signal', 'Bollinger_PctB', 'Bollinger_Width',"upval","downval",f"Option_{direction}_Open",f"Option_{direction}_Last"]]
    features["pl"] = (features[f"Option_{direction}_Last"] -features[f"Option_{direction}_Open"])/features[f"Option_{direction}_Open"]  
    features["MACD"] =features["MACD"] / 50
    features["RSI"] =features["RSI"] / 100
    features["Bollinger_PctB"] =features["Bollinger_PctB"]  
    features["Bollinger_Width"] =features["Bollinger_Width"] / 5
    features =features.drop(columns=[f"Option_{direction}_Open",f"Option_{direction}_Last"])
    X =features.values[:,2:].astype(np.float32)

    s3 = boto3.resource('s3')
    dax = (s3.Object("REDACTED_BUCKET", f"confidencemodelday/daymodel{direction}.csv").get()['Body'].read().decode('utf-8') )
    analysisfile = open(f"daymodel{direction}.csv", "w+")
    analysisfile.write(dax)
    analysisfile.close()
    dax = (s3.Object("REDACTED_BUCKET", f"confidencemodelday/daymodel{direction}.h5f.index").get()['Body'].read() )
    analysisfile = open(f"daymodel{direction}.h5f.index", 'wb')
    analysisfile.write(dax)
    analysisfile.close()
    dax = (s3.Object("REDACTED_BUCKET", f"confidencemodelday/daymodel{direction}.h5f.data-00000-of-00001").get()['Body'].read() )
    analysisfile = open(f"daymodel{direction}.h5f.data-00000-of-00001", 'wb')
    analysisfile.write(dax)
    analysisfile.close()
    schema = json.loads(open(f"daymodel{direction}.csv", 'r').read())
    model = load_model(schema)
    y_pred_class = (model.predict(X) > 0.5).astype("int32")  
    target = filtered_data[['datetime','Date',"Last"]]
    data = {"datetime": target["datetime"].values,"pred": y_pred_class.reshape(-1),"Last":target["Last"].values}
    counter = 0
    pred = 1
    df = pd.DataFrame(data=data)
    proba =  float(table.get_item( Key={'val': f"proba_{direction}"})["Item"]["value"])
    logger.info(proba)

     
    client_step = boto3.client('stepfunctions')
    state_machine_arn = 'arn:aws:states:eu-west-1:REDACTED_AWS_ACCOUNT_ID:stateMachine:sellmarket'
    invoke_event = {
            "action": "SELL",
            "account": "Stonks",
        }
    for elem in df.values:
     
        if elem[1] ==0:
            counter = 0
        else:
            counter += 1
            if (1-proba)**counter < 0.05:
                logger.info("verkaufen")
                logger.info(elem[0])
                return {
                "result" :1,
                "date" : elem[0]
                }
                #response = client_step.start_execution(
                ##    stateMachineArn=state_machine_arn,
                #    input=json.dumps(invoke_event)
   # )
    return {
            "result" :0,
            }


   





 