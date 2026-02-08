import os
import json
import shutil
import numpy as np
import pandas as pd
import boto3
import ta
import tensorflow as tf
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, LeakyReLU, BatchNormalization, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from bayes_opt import BayesianOptimization
from tensorflow import keras
from decimal import Decimal
import traceback
import datetime
import pytz
import logging
import sys

class TimeLimitExceededException(Exception):
    pass

def current_german_time():
    utc_now = datetime.datetime.now(pytz.utc)
    berlin_tz = pytz.timezone('Europe/Berlin')
    return utc_now.astimezone(berlin_tz)

def should_terminate():
    german_time = current_german_time()
    return 8 <= german_time.hour < 9



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
last_rsi_close_14d = full_df['RSI_Close_14d'].iloc[-1]
last_rsi_open = full_df['RSI_Open'].iloc[-1]
last_rsi_close_diff = full_df['RSI_Close_diff'].iloc[-1]
last_rsi_open_diff = full_df['RSI_Open_diff'].iloc[-1]
# new ta indicators
last_ema_ratio_close = full_df['EMA_Ratio_Close'].iloc[-1]
last_bb_bandwidth = full_df['BB_Bandwidth'].iloc[-1]
last_bb_percent = full_df['BB_Percent'].iloc[-1]
last_stochastic_scaled = full_df['Stochastic_Scaled'].iloc[-1]

full_df['RSI_Open'] = full_df['RSI_Open'].shift(1)
full_df['RSI_Close'] = full_df['RSI_Close'].shift(1)
full_df['RSI_Open_diff'] = full_df['RSI_Open_diff'].shift(1)
full_df['RSI_Close_diff'] = full_df['RSI_Close_diff'].shift(1)
full_df['RSI_Close_14d'] = full_df['RSI_Close_14d'].shift(1)
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
full_df["market_variation_tanh"] = full_df["market_variation_tanh"].shift(1)
full_df["market_variation_tanh_diff"] = full_df["market_variation_tanh"].diff()



full_df = full_df.drop(columns=['SMA_5','SMA_10'])


full_df = full_df.dropna()
full_df["performancemean"] = full_df["performancemean"].apply(lambda x : x-1)
full_df["performance_oppossitemean"] = full_df["performance_oppossitemean"].apply(lambda x : x-1)
 


dataset = full_df.drop(columns=['Date','Open',"High","Low","Close"])
dataset["direction"] = (dataset["performance"] >= 1).astype(int) 
value_data = full_df.iloc[49:][['Date',"action","performance","performance_oppossite"]]
dataset =dataset.drop(columns=['action'])
data = dataset.values
data = np.delete(data, 179, axis=0) ## at least 5 sigma deviation day, will not yield anything to be used in the training
X_indices =  np.r_[0:62, 65:79+2]
X = data[:, X_indices]
y_indices = np.r_[62:65, 79+2:80+2]
y = data[:, y_indices]
value_data_df =value_data["action"].value_counts() / (value_data["action"].value_counts().sum() )

new_df= pd.DataFrame({"index" : np.array([i for i in range(0,len(y))]),"one": y[:,1],"two": y[:,2]})
new_df["onemean"] =new_df.one.rolling(5).mean()
new_df["twomean"] =new_df.two.rolling(5).mean()
new_df = new_df.dropna()
new_df["onemean"] = new_df["onemean"].apply(lambda x : x-1)
new_df["twomean"] = new_df["twomean"].apply(lambda x : x-1)
new_df["onediff"] = new_df["onemean"].diff()
new_df["twodiff"] = new_df["twomean"].diff()
last_mean_values = new_df.values[len(new_df.values)-1][3:7]

val_stat = value_data_df.values[0]
bestabprox =[0,10,40]
for i in range(0,10000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    val = sum(y_test[:,0]) / len(y_test)
    val2 = sum(y_test[:,3])/len(y_test)
    if np.abs(val -0.5) < bestabprox[1] and np.abs(val2 -val_stat) < bestabprox[2]:
        bestabprox = [i,np.abs(val -0.5),np.abs(val2 -val_stat)  ]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=bestabprox[0])

prediction = []

def create_model(trial):
    input_model = Input(shape=X_train.shape[1:3])

    p1 = trial.suggest_int("x1", 16, 256)
    p2 = trial.suggest_float("drop", 0.1, 0.5)
    p3 = trial.suggest_float("leaky", 0.01, 0.99)
    denseone = Dense(p1)(input_model)
    denseone = LeakyReLU(alpha=p3)(denseone)
    denseone = Dropout(rate=p2)(denseone)
    denseone = BatchNormalization()(denseone)

    x2 = trial.suggest_int("x2", 16, 256)
    l2 = trial.suggest_float("leaky2", 0.01, 0.99)
    drop2 = trial.suggest_float("drop2", 0.1, 0.5)
    densetwo = Dense(x2)(denseone)
    densetwo = LeakyReLU(alpha=l2)(densetwo)
    densetwo = Dropout(rate=drop2)(densetwo)
    densetwo = BatchNormalization()(densetwo)

    final = Dense(1, name='Dense_final', activation='sigmoid')(densetwo)
    model = Model(inputs=[input_model], outputs=final)

    return model

def objective(trial):
    tf.keras.backend.clear_session()

 
  

    p4 = trial.suggest_int("epoch", 10, 1500)
    p5 = trial.suggest_int("batch_size", 8, 128)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = create_model(trial)
    model.compile(loss=losses.BinaryCrossentropy(), optimizer=Adam(learning_rate=learning_rate))

    # Save    

    history = model.fit(X_train, y_train[:, 0],
                        epochs=p4,
                        validation_data=(X_test, y_test[:, 0]),
                        batch_size=p5,
                        verbose=0,
                        )
    trial_dir = f"dayconf/trial_{trial.number}"
    model.save_weights("model.h5f", overwrite=True)

  

    upload_file_to_s3(bucketmodels,f"trialmodels/{trial.number}/model.h5f.data-00000-of-00001", "model.h5f.data-00000-of-00001")
    upload_file_to_s3(bucketmodels,f"trialmodels/{trial.number}/model.h5f.index","model.h5f.index")
    prediction_test =[elem for elem in model.predict(X_test).reshape(-1)]
    prediction_train =[elem for elem in model.predict(X_train).reshape(-1)]
    max_performance = 0
    max_train = 0
    max_performance = 0
    max_train = 0
    for q in range(0,99):
        #print(q)
        alpha = q*0.01+0.01
        prediction =[int(elem >alpha) for elem in prediction_test]
        #print(prediction_test)
        
    
        real_performance = 1
    #print(prediction)
        for i, elem in enumerate(prediction):
            if elem == 1:
                real_performance *= y_test[i,1]
        
            if elem == 0:
                real_performance *= y_test[i,2]
        performance_train = 1
        prediction =[int(elem >alpha) for elem in prediction_train]
        for i, elem in enumerate(prediction):
            if elem == 1:
                performance_train *= y_train[i,1]
        
            if elem == 0:
                performance_train *= y_train[i,2]
    #print(performance_train * 0.0001/1000)
        real_performance += performance_train * 0.0001/10000/2
        
        if real_performance > max_performance:
            #print(real_performance)
            max_train = performance_train * 0.0001/10000/2
            max_performance = real_performance
        

              
    print(f"{ performance_train * 0.0001/10000/2}")    
    return -real_performance

counter = 0
performance = 0
start_time = time.time()
TOTAL_ITERATIONS = 1500
LOG_INTERVAL = 100
train_test_diff = 0
alpha_opt = 0
beta_opt = 0
 

def optimizeml(**kwargs):
    tf.keras.backend.clear_session()
    global performance,  counter, train_test_diff,alpha_opt 
    # Check if we should terminate based on the time
    if should_terminate():
        #pass
        raise TimeLimitExceededException("Approaching cut-off time. Terminating tuning...")
    global performance, prediction, counter
    p1 = int(kwargs["x1"])
    p2 = kwargs["drop"]
    p3 = kwargs["leaky"]
    
    x2 = int(kwargs["x2"])
    l2 = kwargs["leaky2"]
    drop2 = kwargs["drop2"]

    x3 = int(kwargs["x3"])
    l3 = kwargs["leaky3"]
    drop3 = kwargs["drop3"]

    p4 =int(kwargs["epoch"])
    p5 =int(kwargs["batch_size"])

    input_model = Input(shape= X_train.shape[1:3])
    
    denseone = Dense(p1,)(input_model)
    denseone = LeakyReLU(alpha=p3)(denseone)
    denseone = Dropout(rate=p2)(denseone)
    denseone= BatchNormalization()(denseone)
    
    densetwo = Dense(x2,)(denseone)
    densetwo = LeakyReLU(alpha=l2)(densetwo)
    densetwo = Dropout(rate=drop2)(densetwo)
    densetwo= BatchNormalization()(densetwo)

    densetree = Dense(x3,)(densetwo)
    densetree = LeakyReLU(alpha=l3)(densetree)
    densetree = Dropout(rate=drop3)(densetree)
    densetree= BatchNormalization()(densetree)
    final = Dense(1, name='Dense_final', activation='sigmoid')(densetree)
    model = Model(inputs=[input_model], outputs=final)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=Adam())
    history =model.fit(X_train,y_train[:,0],epochs=p4,validation_data=(X_test, y_test[:,0]),batch_size=p5,verbose=0)
    loss = history.history["val_loss"]
    last_val_loss = loss[len(loss)-1]
    prediction_test =[elem for elem in model.predict(X_test,verbose=0).reshape(-1)]
    prediction_train =[elem for elem in model.predict(X_train,verbose=0).reshape(-1)]
    max_performance = 0
    max_train = 0
    max_alpha = 0
     
    start, end = 0.5, 0.52

    # Step size
    step = 0.01
    for alpha in np.arange(start, end, step):
     
        #print(q)
        
        prediction =[0 if elem < alpha else 1 for elem in prediction_test]
        #print(prediction_test)
        
    
        real_performance = 1
    #print(prediction)
        for i, elem in enumerate(prediction):
            if elem == 1:
                real_performance *= y_test[i,1]
        
            if elem == 0:
                real_performance *= y_test[i,2]
        performance_train = 1
        prediction =[0 if elem <= alpha else 1 for elem in prediction_train]
        for i, elem in enumerate(prediction):
            if elem == 1:
                performance_train *= y_train[i,1]
        
            if elem == 0:
                performance_train *= y_train[i,2]
    #print(performance_train * 0.0001/1000)
        real_performance += performance_train * 0.0001/10000/8
        
        if real_performance > max_performance:
            #print(real_performance)
            max_train = performance_train * 0.0001/10000/8
            max_performance = real_performance
            max_alpha = alpha
          
              
        
    
    
    
 
    
    if max_performance>performance:
        #print("dfhdjfhdj")
        model.save_weights("model.h5f", overwrite=True)
        upload_file_to_s3(bucketmodels,f"confidencemodelday/model.h5f.data-00000-of-00001", "model.h5f.data-00000-of-00001")
        upload_file_to_s3(bucketmodels,f"confidencemodelday/model.h5f.index","model.h5f.index")
        write_data_to_s3(bucketmodels, "confidencemodelday/model.csv", json.dumps(kwargs ))
        performance = max_performance
        logger.info("{}:{}||{}|{}".format(counter,performance,max_train,max_alpha))
        alpha_opt = max_alpha
        train_test_diff = performance - max_train
    counter += 1    
    
    kwargs["counter"]= counter
    # log the amount of time needed
    if counter % LOG_INTERVAL == 0:
        elapsed_time = time.time() - start_time
        avg_speed = counter / elapsed_time * 60  # iterations per minute
        
        remaining_iterations = TOTAL_ITERATIONS - counter
        estimated_completion_time = remaining_iterations / avg_speed  /60
        
        logger.info(f"Completed {counter/TOTAL_ITERATIONS*100}% of the iterations.")
        logger.info(f"Average speed: {avg_speed:.2f} iterations/minute.")
        logger.info(f"Estimated time to completion: {estimated_completion_time:.2f} hours.")


    #print(real_performance)
    return real_performance


optimizer = BayesianOptimization(
    f=optimizeml,
     pbounds={'x1': (1, 128),'x2': (1,128),'x3': (1,128),
 'drop': (0, 0.5),'drop2': (0, 0.5),'drop3': (0, 0.5),
 'leaky': (0, 1),'leaky2': (0, 1),'leaky3': (0, 1),
 'epoch': (10, 1700),
 'batch_size': (2, 300)},
    verbose=0,
    random_state=42,
    #bounds_transformer=bounds_transformer
)
performance = 0
try:
    optimizer.maximize(
        init_points=15,
        n_iter=TOTAL_ITERATIONS-15,
    )
except TimeLimitExceededException as e:
    logger.info(e)
except Exception as e:
    traceback.print_exc()

 

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
 



 
response = put_value_in_dynamodb(table_name, primary_key_name, "alpha_opt", alpha_opt)
response = put_value_in_dynamodb(table_name, primary_key_name, "train_test_diff_max", train_test_diff)


 

 
 
 