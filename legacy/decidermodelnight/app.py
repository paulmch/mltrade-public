import pandas as pd
import numpy as np
import os
import json
import boto3
client = boto3.client('lambda')
s3 = boto3.resource('s3')
dax = (s3.Object("REDACTED_BUCKET", "dax.csv").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/dax.csv", "w+")
analysisfile.write(dax)
analysisfile.close()
dax = (s3.Object("REDACTED_BUCKET", "deviation.csv").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/deviation.csv", "w+")
analysisfile.write(dax)
analysisfile.close()

tmp = (s3.Object("REDACTED_BUCKET", "modelnight/bestnight.json").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/bestnight.json", "w+")
analysisfile.write(tmp)
analysisfile.close()
    
    
tmp = (s3.Object("REDACTED_BUCKET", "modelnight/bestnight.index").get()['Body'].read()  )
analysisfile = open("/tmp/bestnight.index", 'wb')
analysisfile.write(tmp)
analysisfile.close()

tmp = (s3.Object("REDACTED_BUCKET", "modelnight/bestnight.data-00000-of-00001").get()['Body'].read()  )
analysisfile = open("/tmp/bestnight.data-00000-of-00001", 'wb')
analysisfile.write(tmp)
analysisfile.close()




path = "/tmp/deviation.csv"

df = pd.read_csv(path,  parse_dates=['Date'])
df = df.rename(columns={"Open" : "Open_conf", "High" : "High_conf", "Low" : "Low_conf", "Close" : "Close_conf"})

path = "/tmp/dax.csv"

dax = pd.read_csv(path,usecols=['Date','Open',"High","Low","Close"],  parse_dates=['Date'])

dax = dax.dropna()


full_df =  pd.merge(df, dax, on="Date")
full_df["Open_dev"] = (full_df["Open"]- full_df["Open_conf"] ) / full_df["Open_sigma"]
full_df["High_dev"] = (full_df["High"]- full_df["High_conf"] ) / full_df["High_sigma"]
full_df["Low_dev"] = (full_df["Low"]- full_df["Low_conf"] ) / full_df["Low_sigma"]
full_df["Close_dev"] = (full_df["Close"]- full_df["Close_conf"] ) / full_df["Close_Sigma"]
full_df['weekday'] = full_df["Date"].dt.weekday.astype(np.int8) 
full_df['monthday'] = full_df["Date"].dt.day.astype(np.int8) -1
del dax, df

def aux_data(values,i):
    val = []
    val = [(-values[i][9]+values[i][12])/2000]
    val += [10* (-values[i][9]+values[i][12])/values[i][9]]
    return val
def optimalnightstrategy(values,i):
    if values[i][12] <= values[i+1][9]:
        return 0
    if values[i][12] > values[i+1][9]:
        return 1
    

def one_hot_encode(number,classes):
    return [int( number == i   ) for i in range(0,classes)]
def one_hot_encode_all(row,classes):
    list_columns = []
    for i in range(0,16):
        list_columns += one_hot_encode(row[i],classes)
    return list_columns

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
import tensorflow as tf

def prediction(model,row,aux_row):
    row =np.array([row])
    aux_row =np.array([aux_row])
    predict = model.predict([row,aux_row])
    prediction = [int(elem > 0.5) for elem in predict.reshape(-1)]

    return prediction[0]
            
            
    

def load_model(kwargs):
    global maximum_win
    tf.keras.backend.clear_session()
    p1 = int(kwargs["x1"])
    p2 = int(kwargs["x2"])
    p3 = int(kwargs["dense_aux"])
    input_model = Input(shape= (11, 19))
    denseone = LSTM(p1,dropout =kwargs["drop1"],recurrent_dropout = kwargs["recdrop1"])(input_model)
    denseone = LeakyReLU(alpha=kwargs["relu1"])(denseone)
    denseone= BatchNormalization()(denseone)
    input_aux = Input(shape= (2,))
    dense_aux = Dense(p3,)(input_aux)

    dense_aux = LeakyReLU(alpha=kwargs["relu1_aux"])(dense_aux)
    dense_aux = Dropout(rate=kwargs["drop_aux"])(dense_aux)
    dense_aux= BatchNormalization()(dense_aux)
    concate_layers = tf.keras.layers.concatenate([denseone,dense_aux])
    densetwo =Dense(p2,)(concate_layers)
    densetwo = LeakyReLU(alpha=kwargs["relu2"])(densetwo)
    densetwo = Dropout(rate=kwargs["drop2"])(densetwo)
    densetwo= BatchNormalization()(densetwo)
    final = Dense(1, name='Dense_final', activation='sigmoid')(densetwo)
    model = Model(inputs=[input_model,input_aux], outputs=final)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=kwargs["learningrate"]))
    model.load_weights("/tmp/{}".format(kwargs["key"]))
    return model




 


def handler(event, context):
  
    values = full_df.values
    x_data_aux = np.array([aux_data(values,i) for i in range(1,len(values)-1)])
    help = open("/tmp/{}.json".format("bestnight"), 'r').read()
    model_parameter = eval(help)
    params = model_parameter["hyperparameters"]
    params["key"] =  "bestnight"
    model = load_model(params)

    key = event["key"]
    value_id = event["value_id"]

    actions  = []
    for i in range(0,11):
        tmp = s3.Object('REDACTED_BUCKET',"night/{}/{}/{}/{}_{}.txt".format(key[0],key[1],key[2],value_id,i)).get()['Body'].read().decode('utf-8')
        actions .append(int(tmp))
    actions = np.array(actions)
    #actions = np.array([one_hot_encode(elem,19)for elem in actions ])
    action =prediction(model,actions,x_data_aux[len(x_data_aux)-1   ])

    if action == 0:
        knockoutprice = full_df.values[len(full_df.values) -2 ][12] -600
        pos = ["long",knockoutprice]
    if action == 1:
        knockoutprice = full_df.values[len(full_df.values) -2 ][12] +600
        pos = ["short",knockoutprice]
    

    

    content = {"title" : "MLStratnight" , "investment" : pos}
   
    client.invoke(
    FunctionName='sendMail',
    InvocationType='Event',
    Payload=json.dumps(content))
   


   





 