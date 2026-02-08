from tabnanny import verbose
import pandas as pd
import numpy as np
import os
import json
import boto3
from datetime import datetime
from bayes_opt import BayesianOptimization
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
values = full_df.values
del dax, df

def aux_data(values,i):
    val = []
    val = [10* (values[i+1][9]-values[i][12])/values[i][12]]
    val += [10* (-values[i][9]+values[i][12])/values[i][9]]
    return val
    
x_data_aux = np.array([aux_data(values,i) for i in range(1,len(values)-1)])
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
def buy_long(values,budget, x12,x13,step,stop ):
    i = step
    
    
       
    knockoutprice = x12 * values[i+1][9]
        
    price_per_share = price(values[i+1][9], knockoutprice  )
    
    allocated_money = min(budget,x13 )
    budget = budget - allocated_money
   
    number_of_shares = int( (allocated_money-1)/ price_per_share   )
    allocated_money = allocated_money - number_of_shares * price_per_share -1
    budget = budget + allocated_money
    return budget, ["long",knockoutprice,number_of_shares,stop]

def buy_short(values,budget, x12,x13,step,stop ):
    i = step
    
    
        
    knockoutprice = x12 *values[i+1][9]
        
    price_per_share = price( knockoutprice, values[i+1][9]  )
    allocated_money = min(budget,x13 )
    budget = budget - allocated_money
    number_of_shares = int( (allocated_money-1)/ price_per_share   )
    allocated_money = allocated_money - number_of_shares * price_per_share -1
    budget = budget + allocated_money
    return budget, ["short",knockoutprice,number_of_shares,stop]
    
def sell(values, budget, pos,step):
    i = step+1
    
    if(len(pos) == 0 ):
        
        return budget,[]
    if pos[0] == "long":
        price_per_share =  price(values[i][12], pos[1]  )
        if price(values[i][11],pos[1] ) < pos[3] * price(values[i][9], pos[1]  ):
            price_per_share = pos[3] * price(values[i][9], pos[1]  )
            
            
        if price_per_share == 0:
            return budget, []
        price_per_share = price_per_share - 0.01
        sell_money =  price_per_share * pos[2]
        
        if sell_money == 0:
            return budget, []
        budget = budget + sell_money -1
        return budget, []
    if pos[0] == "short":
        price_per_share = price( pos[1],values[i][12]  )
        if price( pos[1],values[i][10]  )  <=  pos[3] * price( pos[1],values[i][9]  ):
            price_per_share = pos[3] * price( pos[1],values[i][9]  )
        if price_per_share == 0:
            return budget, []
        price_per_share = price_per_share - 0.01
            
        sell_money =  price_per_share * pos[2]
        
        if sell_money == 0:
            return budget, []
        budget = budget + sell_money -1
        return budget, []
    
    
def take_action(budget,step, action,values,stop,x6,x7):
    pos = []
    
    previous_budget = budget
    direction = 0
    if action > 8 and action < 8 + x6:
        action = 8 
    if action > 10 - x7 and action < 10:
        action = 10 
    if action <= 8:
        x12 = 0.89 + 0.01* action
        budget, pos = buy_long(values,budget, x12,Maximum_invest, step,stop )
        direction =1
            
    if action  >= 10:
        x12 = 1.03 + 0.01* (   action-10)
        direction = -1
        budget, pos = buy_short(values, budget, x12,Maximum_invest,step,stop)
    pos_tmp = pos
    #print([pos,values[step,0]])
   
    budget, pos = sell(values,budget, pos , step)
    
        
    return budget, direction  


def notstoppedout(direction,values, leverage , i ,stop = 0.71  ):
    if direction:
        knockoutprice = leverage * values[i+1][9]
        price_per_share = price(values[i+1][9], knockoutprice  )
        if price(values[i+1][11], knockoutprice  ) < price_per_share * stop:
            return False
        return True
    else:
        knockoutprice = leverage *values[i+1][9]
        price_per_share = price( knockoutprice, values[i+1][9]  )
        if price(  knockoutprice ,values[i+1][10]  )  <  price_per_share * stop:
            return False
        return True
            
Maximum_invest = 750000.
def test_action(stop,action):
    a = 613
  
    val = full_df.values
    
    
    budget_list = [] 
    direction_list = []
    good_trades = 0
    budget =   1000
    prev_budget = budget
    budget_list.append(budget)
    for i , elem in enumerate(action[:]):
    
        budget, direction = take_action(budget, i+a+1 , elem , val,stop,0,0)
        direction_list.append(direction)
        budget_list.append(budget)
       
    return budget_list, direction_list


def calculateoptimalleverage(values,i,stop= 0.71):
    direction = values[i+1][19]
    if direction:
         
        knockouts = [notstoppedout(direction,values,0.89 + 0.01* action,i,stop) for action in range(0,9)]
        
        counter = -1
        for elem in knockouts:
            if elem:
                counter += 1
        if counter == -1:
            counter = 9
        return counter
    else:
         
        knockouts = [notstoppedout(direction,values, 1.03 + 0.01* (   action),i,stop) for action in range(0,9)]
        counter = -1
        for elem in knockouts:
            if not elem:
                counter += 1
        if counter == 8:
            counter = 9
            return counter
        return counter + 11



def prediction(model,row,aux_row):
    row =np.array([row])
    aux_row =np.array([aux_row])
    predict = model.predict([row,aux_row])
    i = -1
    j = -1
    max_value = -30
    for elem in predict[0]:
        i += 1
        if elem > max_value:
            j = i
            max_value = elem
    return j
            
            
    

def load_model(kwargs):
    global maximum_win
    tf.keras.backend.clear_session()
    p1 = int(kwargs["x1"])
    p2 = int(kwargs["x2"])
    p3 = int(kwargs["dense_aux"])
    input_model = Input(shape= (16, 19))
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
    final = Dense(19, name='Dense_final', activation='softmax')(densetwo)
    model = Model(inputs=[input_model,input_aux], outputs=final)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=kwargs["learningrate"]))
    model.load_weights("/tmp/{}".format(kwargs["key"]))
    return model





def price(indexvalue : float, knockoutprice : float  ):
    return max((indexvalue - knockoutprice)* 0.01 , 0 )
action_list = []         
for i in range(0,16):
    agent = (s3.Object("REDACTED_BUCKET", "intradayaction/agent{}.csv".format(i)).get()['Body'].read().decode('utf-8') )
    analysisfile = open("/tmp/agent{}.csv".format(i), "w+")
    analysisfile.write(agent)
    analysisfile.close()
    df = pd.read_csv("/tmp/agent{}.csv".format(i),parse_dates=['Date'])
    action_list.append(df)
x_data = np.array( [elem.values[:len(elem.values),1] for elem in action_list ]).transpose()

x_data_transformed = np.array([[one_hot_encode(elem,19)for elem in row] for row in x_data])
def test_stop(stop):
    global predict
    budget_list, direction_list = test_action(stop,predict)
    return budget_list[len(budget_list)-2]
values = full_df.values
x_data_aux = np.array([aux_data(values,i) for i in range(1,len(values)-1)])
payload = {"fill123":"fill345"}
response=client.invoke(
        FunctionName='findbestmodel',
        InvocationType='RequestResponse',
        Payload=json.dumps(payload))
modelname = json.loads(response["Payload"].read().decode('utf-8'))["best_model"]
tmp = (s3.Object("REDACTED_BUCKET", "rawmodelday/modelstructure/{}".format(modelname)).get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/bestday.json", "w+")
analysisfile.write(tmp)
analysisfile.close()
    
    
tmp = (s3.Object("REDACTED_BUCKET", "rawmodelday/models/{}.index".format(modelname)).get()['Body'].read()  )
analysisfile = open("/tmp/bestday.index", 'wb')
analysisfile.write(tmp)
analysisfile.close()

tmp = (s3.Object("REDACTED_BUCKET", "rawmodelday/models/{}.data-00000-of-00001".format(modelname)).get()['Body'].read()  )
analysisfile = open("/tmp/bestday.data-00000-of-00001", 'wb')
analysisfile.write(tmp)
analysisfile.close()



help = open("/tmp/{}.json".format("bestday"), 'r').read()
model_parameter = json.loads(help)
params = model_parameter["hyperparameters"]
params["key"] =  "bestday"
model = load_model(params )
predict =[ prediction(model,x_data_transformed[i],x_data_aux[i]) for i in range(613,len(x_data_transformed)) ]

def handler(event, context):
    bounds = {"stop" : (0,1)  }
    optimizer = BayesianOptimization( f=test_stop,
        pbounds=bounds
    ,verbose=2,
    random_state=42,
)
    optimizer.maximize(init_points=5, n_iter=100)
    optimalparams =optimizer.max
    optimalstoploss = optimalparams["params"]["stop"]
    jsonstructure = {
        
        "model" : modelname.replace(".json",""),
        "recommended stop loss": "{:.2f} %".format(optimalstoploss*100)

    
    
        }
    s3.Object('REDACTED_BUCKET',"rawmodelday/modelday/recommendations.json").put(Body=json.dumps(jsonstructure))

 
 
     
    

    
 
   


   





 