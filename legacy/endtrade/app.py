 
import numpy as np
 
import json
import boto3 
from datetime import datetime
s3 = boto3.resource('s3')
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

def load_model(kwargs):
    p1 = int(kwargs["x1"])
    p2 = kwargs["drop"]
    p3 = kwargs["leaky"]
    
    x2 = int(kwargs["x2"])
    l2 = kwargs["leaky2"]
    drop2 = kwargs["drop2"]

    x3 = int(kwargs["x3"])
    l3 = kwargs["leaky3"]
    drop3 = kwargs["drop3"]

    input_model = Input(shape=(78,))
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
    model.load_weights("/tmp/model.h5f")
    return model


def generate_data_x(event):
    action = event["action"]
    date = datetime.now()
    startval = event["startval"]
    endval = event["endval"]
    upval = event["upval"]
    downval = event["downval"]
    performancemean = event["performancemean"]
    performance_oppossitemean = event["performance_oppossitemean"]
    performancemean_diff = event["performancemean_diff"]
    performance_oppossitemean_diff = event["performance_oppossitemean_diff"]
    last_rsi_close = event["RSI_Close"]
    last_rsi_close_14d = event["RSI_Close_14d"]
    last_rsi_open = event["RSI_Open"]
    last_rsi_close_diff = event["RSI_Close_diff"]
    last_rsi_open_diff = event["RSI_Open_diff"]

    last_ema_ratio_close = event["EMA_Ratio_Close"]
    last_bb_bandwidth = event["BB_Bandwidth"]
    last_bb_percent = event["BB_Percent"]
    last_stochastic_scaled = event["Stochastic_Scaled"]
    
    last_ma_crossover_signal = event['MA_Crossover_Signal']
    last_market_variation_tanh = event['Market_Variation_Tanh']
    last_market_variation_tanh_diff = event['Market_Variation_Tanh_diff']
    action_data = [int(elem == action) for elem in range(0, 20)]
    weekday_data = [int(date.weekday() == elem) for elem in range(0, 5)]
    day_data = [int(date.day-1 == elem) for elem in range(0, 31)]
    month_data = [int(((date).month-1) % 3 == elem) for elem in range(0, 3)]
    
    data = [upval, downval] + action_data + weekday_data + day_data + month_data
    data += [(startval - endval) / endval]
    data += [performancemean, performance_oppossitemean, performancemean_diff, performance_oppossitemean_diff]
    data += [last_rsi_close, last_rsi_open,last_rsi_close_14d]
    data += [last_ema_ratio_close, last_bb_bandwidth,last_bb_percent,last_stochastic_scaled]
    data +=  [last_rsi_close_diff, last_rsi_open_diff]
    data += [last_ma_crossover_signal]
    data += [last_market_variation_tanh, last_market_variation_tanh_diff]
    return np.array([data])


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
    prediction = load_model(schema).predict(generate_data_x(event))

    # Load Model
    

    return {"loss_prop" : float(1- prediction[0]),        "win_prop" : float(prediction[0])}
   


   





 