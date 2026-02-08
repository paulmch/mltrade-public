import pandas as pd
import numpy as np
import json
import boto3 
s3 = boto3.resource('s3')
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

startval = int(os.environ["startval"])
def generate_data_x(action,date,startval,endval,upval,downval):
    data = [upval,downval] + [int(elem == action) for elem in range(0,20)] +[int(date.weekday()== elem) for elem in range(0,5)]   + [int(date.day-1== elem) for elem in range(0,31)] + [(startval-endval)/endval]
    return np.array([data])
def load_model(kwargs):
    p1 = int(kwargs["x1"])
    p2 = kwargs["drop"]
    p3 = kwargs["leaky"]
    p4 =int(kwargs["epoch"])
    p5 =int(kwargs["batch_size"])
    x2 = int(kwargs["x2"])
    l2 = kwargs["leaky2"]
    drop2 = kwargs["drop2"]
    input_model = Input(shape=(59,))
    denseone = Dense(p1,)(input_model)
    denseone = LeakyReLU(alpha=p3)(denseone)
    denseone = Dropout(rate=p2)(denseone)
    denseone= BatchNormalization()(denseone)
    densetwo = Dense(x2,)(denseone)
    densetwo = LeakyReLU(alpha=l2)(densetwo)
    densetwo = Dropout(rate=drop2)(densetwo)
    densetwo= BatchNormalization()(densetwo)
    final = Dense(1, name='Dense_final', activation='sigmoid')(densetwo)
    model = Model(inputs=[input_model], outputs=final)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=Adam())
    model.load_weights("/tmp/model.h5f")
    return model
    
dax = (s3.Object("REDACTED_BUCKET", "dax.csv").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/dax.csv", "w+")
analysisfile.write(dax)
analysisfile.close()

dax = (s3.Object("REDACTED_BUCKET", "dayactions.csv").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/dayactions.csv", "w+")
analysisfile.write(dax)
analysisfile.close()

df1 = pd.read_csv("/tmp/dayactions.csv", parse_dates=['Date'])
path = "/tmp/dax.csv"

df = pd.read_csv(path,usecols=['Date','Open',"High","Low","Close"],  parse_dates=['Date'])
def one_hot_encode_column(data: pd.DataFrame , col):
    df = data.copy()
    max = df[col].max()
    for i in range(0,max+1):
         df[col+str(i)] = np.where(df[col]==i, 1, 0)
    df = df.drop(columns=[col])
    return df
def one_hot_encode_column_with_fixed_columns(data: pd.DataFrame , col,max_classes):
    df = data.copy()
    
    for i in range(0,max_classes+1):
         df[col+str(i)] = np.where(df[col]==i, 1, 0)
     
    return df
df['weekday'] = df["Date"].dt.weekday.astype(np.int8) 
 
df['monthday'] = df["Date"].dt.day.astype(np.int8) -1
df = one_hot_encode_column(df,'weekday')
 
df = one_hot_encode_column(df,'monthday')
df1 =one_hot_encode_column_with_fixed_columns(df1,"action",19)

full_df =  pd.merge(df1, df, on="Date")
def price(indexvalue : float, knockoutprice : float  ):
    return max((indexvalue - knockoutprice)* 0.01 , 0 )

prevclose = 13015.230469
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
 
        
full_df["prevclose"]= full_df.apply(closeperc, axis=1)
full_df["result"]= full_df.apply(buy, axis=1)
full_df["performance"]= full_df.apply(sell, axis=1)
full_df["performance_oppossite"]= full_df.apply(selloposite, axis=1)
dataset = full_df.iloc[startval:].drop(columns=['Date','Open',"High","Low","Close","action","performance","performance_oppossite"])
value_data = full_df.iloc[startval:][['Date',"performance","performance_oppossite"]]
data = dataset.values
X = data[:,:59]
y = data[:,59]



def handler(event, context):
    dax = (s3.Object("REDACTED_BUCKET", "confidencemodelday/model.csv").get()['Body'].read().decode('utf-8') )
    analysisfile = open("/tmp/model.csv", "w+")
    analysisfile.write(dax)
    analysisfile.close()
    dax = (s3.Object("REDACTED_BUCKET", "confidencemodelday/model.h5f.index").get()['Body'].read() )
    analysisfile = open("/tmp/model.h5f.index", 'wb')
    analysisfile.write(dax)
    analysisfile.close()
    dax = (s3.Object("REDACTED_BUCKET", "confidencemodelday/model.h5f.data-00000-of-00001").get()['Body'].read() )
    analysisfile = open("/tmp/model.h5f.data-00000-of-00001", 'wb')
    analysisfile.write(dax)
    analysisfile.close()
    schema = json.loads(open("/tmp/model.csv", 'r').read())
    model = load_model(schema)
    prediction =[int(elem > 0.5) for elem in model.predict(X).reshape(-1)]
    actual = [int(elem > 0.5) for elem in y]
    correct = 0
    for i, elem in enumerate(prediction):
        if elem == actual[i]:
            correct += 1
    number_of_wins =  0
    number_of_loss_pred = 0
    for i, elem in enumerate(prediction):
        if elem == 0:
            number_of_loss_pred += 1
            if value_data.iloc[i]["performance_oppossite"] > 1:
                number_of_wins += 1
    return_json = {"confidence_prediction" : correct / len(prediction),
                "opposite_action_performance" : number_of_wins / number_of_loss_pred }
    

     

    
     

    # Load Model
    

    return return_json  

