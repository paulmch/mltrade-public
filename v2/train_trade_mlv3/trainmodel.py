import tensorflow as tf
import boto3
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout,LeakyReLU,BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import logging
import sys
from ncps import wirings
from ncps.tf import LTC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def pricelong(indexvalue, knockoutprice) :
    """Calculate the price of long options """
    return np.maximum((indexvalue - knockoutprice) * 0.01, 0)

def priceshort(indexvalue, knockoutprice) :
    """Calculate the price of short options"""
    return np.maximum((knockoutprice - indexvalue) * 0.01, 0)


def upload_file_to_s3(bucket, key, file):
    s3 = boto3.client("s3")
    s3.upload_file(file, bucket, key)
    
def write_data_to_s3(bucket, key, content):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=content)
     
def evaluate_strategy(date, direction):
    close_day = df[df["Date"] == date]["Close"].values[0]
    df_intra_day = intradaydax_agg[intradaydax_agg["Date"]==date]

    index_value = df_intra_day.iloc[19]["Close"]
    

    knockout_long = 0.97 * index_value
    knockout_short = 1.03 * index_value

    open_price_long = pricelong(index_value,knockout_long)
    open_price_short = priceshort(index_value,knockout_short) 
     
    close_price_long = pricelong(close_day,knockout_long)
    close_price_short = priceshort(close_day,knockout_short)

    if close_price_long < 0.71 *open_price_long:
        close_price_long = 0.71 *open_price_long
    if close_price_short < 0.71 *open_price_short:
        close_price_short = 0.71 *open_price_short 
    
    if direction == 1:
        del df_intra_day
        return 1+ (close_price_long - open_price_long) / open_price_long
    else:
        del df_intra_day
        return 1+(close_price_short - open_price_short) / open_price_short
    
intradaydax_agg = pd.read_csv("intradaydax_agg_V2.csv")
intradaydax_agg['Date'] = pd.to_datetime(intradaydax_agg['Date'], errors='coerce')

df = pd.read_csv("df_V2.csv")
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

numpy_arrays = np.load('train_test_data_V2.npz', allow_pickle=True)

X1_train, X2_train,X1_test, X2_test, y_train, y_test, data_train,data_test = numpy_arrays['X1_train'], numpy_arrays['X2_train'], numpy_arrays['X1_test'], numpy_arrays['X2_test'], numpy_arrays['y_train'], numpy_arrays['y_test'], numpy_arrays['data_train'], numpy_arrays['data_test'] 
with open('params_V2.json', 'r') as json_file:
    kwargs = json.load(json_file)
performance = kwargs["performance"]
p1 = int(kwargs["x1"])
drop1 = kwargs["drop1"]
l1 = kwargs["leaky1"]
   
p2 = int(kwargs["x2"])
l2 = kwargs["leaky2"]
drop2 = kwargs["drop2"]
    
p3 = int(kwargs["x3"])
l3 = kwargs["leaky3"]
drop3 = kwargs["drop3"]
    
lstm1p1 = int(kwargs["lstm1"])
 
    
lstm2p2 = int(kwargs["lstm2"])
 
    
epoch =int(kwargs["epoch"])
batch_size =int(kwargs["batch_size"])
    
input_model_X1 = Input(shape= X1_train.shape[1:3])
denseone = Dense(p1,)(input_model_X1)
denseone = LeakyReLU(alpha=l1)(denseone)
denseone = Dropout(rate=drop1)(denseone)
denseone= BatchNormalization()(denseone)
    
densetwo = Dense(p2,)(denseone)
densetwo = LeakyReLU(alpha=l2)(densetwo)
densetwo = Dropout(rate=drop2)(densetwo)
densetwo= BatchNormalization()(densetwo)
final = Dense(1, name='Dense_final', activation='sigmoid')(densetwo)

input_model_X2 = Input(shape= X2_train.shape[1:3])
fc_wiring = wirings.AutoNCP(lstm1p1,lstm1p1-3)
fc_wiring2 = wirings.AutoNCP(lstm2p2, lstm2p2-3)
lstm1 = LTC(fc_wiring, return_sequences=True)(input_model_X2)
lstm2 = LTC(fc_wiring2, return_sequences=False)(lstm1)
final2 = Dense(1, name='Dense_final2', activation='sigmoid')(lstm2)

mergelayer = concatenate([final, final2])

densemerge = Dense(p3)(mergelayer)
densemerge = LeakyReLU(alpha=l3)(densemerge)
densemerge = Dropout(rate=drop3)(densemerge)
densemerge= BatchNormalization()(densemerge)
finalmerge = Dense(1, name='Dense_finalmerge', activation='sigmoid')(densemerge)

model = Model(inputs=[input_model_X1, input_model_X2], outputs=finalmerge)
model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=Adam())
    
    
history = model.fit([X1_train,X2_train],y_train,epochs=epoch,validation_data=([X1_test,X2_test], y_test),batch_size=batch_size,verbose=0)
y_test_predict =model.predict([X1_test,X2_test],verbose=0)
y_pred_class = (y_test_predict > 0.5).astype("int32")  # Convert probabilities to class labels

strat_performance = 1
for i, elem in enumerate(data_test[:,0]):
    direction = y_pred_class[i,0]
    temp_performance = evaluate_strategy(elem, direction)
    strat_performance *= temp_performance
bucketmodels = "REDACTED_BUCKET"
if strat_performance > performance:
        logger.info(f"New performance is {strat_performance} and it is better than the previous one {performance} ")  
        performance = strat_performance
        model.save_weights(f"interdaymodel_V3.h5f", overwrite=True)
        upload_file_to_s3(bucketmodels,f"confidencemodelday/interdaymodel_V3.h5f.data-00000-of-00001", f"interdaymodel_V3.h5f.data-00000-of-00001")
        upload_file_to_s3(bucketmodels,f"confidencemodelday/interdaymodel_V3.h5f.index",f"interdaymodel_V3.h5f.index")
        write_data_to_s3(bucketmodels, f"confidencemodelday/interdaymodel_V3.csv", json.dumps(kwargs ))


kwargs["performance_new"] = strat_performance
with open('params_V2.json', 'w') as json_file:
    json.dump(kwargs, json_file)
