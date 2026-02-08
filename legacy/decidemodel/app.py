import pandas as pd
import numpy as np
import os
import json
import boto3
from datetime import datetime
import traceback
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


turnaround = float(os.environ["turnaround"])



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
    val = [10* (values[i+1][9]-values[i][12])/values[i][12]]
    val += [10* (-values[i][9]+values[i][12])/values[i][9]]
    return val
    

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

def addintradayaction(iteration : int,date : str,action : int):
    tmp = s3.Object('REDACTED_BUCKET',"intradayaction/agent{}.csv".format(iteration)).get()['Body'].read().decode('utf-8')
    tmp = tmp+"{},{}\n".format(date,action)
    s3.Object('REDACTED_BUCKET', "intradayaction/agent{}.csv".format(iteration)).put(Body=tmp)




def handler(event, context):

    # Load Model
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

    full_df.at[len(full_df.values)-1, "Open" ] = event["start_value"]
    
    startvalue  = event["start_value"]
    values = full_df.values
    intradayactiondate = values[len(values)-2,0].strftime("%Y-%m-%d")
    x_data_aux = np.array([aux_data(values,i) for i in range(1,len(values)-1)])
    help = open("/tmp/{}.json".format("bestday"), 'r').read()
    model_parameter = eval(help)
    params = model_parameter["hyperparameters"]
    params["key"] =  "bestday"
    model = load_model(params )

    key = event["key"]
    value_id = event["value_id"]

    actions  = []
    for i in range(0,16):
        tmp = s3.Object('REDACTED_BUCKET',"day/{}/{}/{}/{}_{}.txt".format(key[0],key[1],key[2],value_id,i)).get()['Body'].read().decode('utf-8')
        addintradayaction(i,intradayactiondate,int(tmp))
        actions .append(int(tmp))
    actions = np.array(actions)
    actions = np.array([one_hot_encode(elem,19)for elem in actions ])
    action =prediction(model,actions,x_data_aux[len(x_data_aux)-1   ])
    pos = []
    if action <= 8:
        
        x12 = 0.89 + 0.01* action
        knockoutprice = x12 *  full_df.values[len(full_df.values) -1 ][9]

        price_per_share =  price(full_df.values[len(full_df.values) -1 ][9], knockoutprice  )
        pos = ["Long",knockoutprice, 0.71 *price_per_share  ]

    if action >= 10:
        
            
        x12 = 1.03 + 0.01* (   action-10)
        knockoutprice = x12 *full_df.values[len(full_df.values) -1 ][9]  
        price_per_share =  price(knockoutprice,full_df.values[len(full_df.values) -1 ][9]  )
        pos = ["Short",knockoutprice, 0.71 *price_per_share  ]
    stored_action = {}
    min_max_value = {}
    if len(pos) > 1:
        direction = pos[0].lower()
        results = json.loads(s3.Object("REDACTED_BUCKET", "rangepred/actual.txt").get()['Body'].read().decode('utf-8'))
        calculated_values = [[float(key),(results[key].replace("[","").replace("]","").replace("'","")).split(",")[0]] for key in results.keys()]
        found = False
        interval = 0
        min_value = 250
        max_value = 250
        for i in range(0,len(calculated_values)-1):
            if startvalue >= calculated_values[i][0] and  startvalue <= calculated_values[i+1][0]:
        
                found = True
                interval = i
        if found:
            for i in range(0,interval+1):
                if calculated_values[interval - i][1] != direction:
                    min_value = startvalue- calculated_values[interval - i][0]
                    break
 

            for i in range(interval+1,len(calculated_values)-1):
                if calculated_values[i][1] != direction:
                    max_value = -startvalue+ calculated_values[ i][0]
                    break
        min_max_value = {
            "Points down until Strategy change" :  min_value,
            "Points up until Strategy change" : max_value
                        }
        time_now = datetime.now()

        s3.Object('REDACTED_BUCKET',"confidence/{}.json".format(time_now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        
         ).put(Body=json.dumps({"date" : time_now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "confidence" : min_max_value,
         "startvalue" : startvalue,
         "action" :  action 
        
        }  ))
        tmp = json.loads(s3.Object('REDACTED_BUCKET',"rawmodelday/modelday/recommendations.json").get()['Body'].read().decode('utf-8'))

        try:
            content = {
            "action" : action ,
            "startval" : startvalue,
            "endval" : full_df.iloc[len(full_df.values)-1]["Close"],
            "upval" : max_value/250,
            "downval" : min_value/250
            }
   
            responce = client.invoke(
            FunctionName='dayconfidenceml',
            InvocationType='RequestResponse',
            Payload=json.dumps(content))
            confidence= json.loads(responce["Payload"].read().decode('utf-8'))
            min_max_value.update(confidence)
        except:
            traceback.print_exc()
            print("O dear a bug? How could that happen? EXPLAIN!EXPLAIN!EXPLAIN")


        stored_action = {
            "Index" : "DAX40",
            "Option type" : "Knockout without fixed stopp loss",
            "Direction"   : pos[0],
            "Knockout" : pos[1]

        }
        stored_action.update(tmp)
        if min_max_value["win_prop"] <turnaround:
            #responce = client.invoke(
            #FunctionName='oppositestrategy',
            #InvocationType='RequestResponse',
            #Payload=json.dumps(content))
            #oppositestrategystats= json.loads(responce["Payload"].read().decode('utf-8'))["opposite_action_performance"]
            if action <= 8:
                action = 11 + (8 - action)
                x12 = 1.03 + 0.01* (   action-10)
                knockoutprice = x12 *full_df.values[len(full_df.values) -1 ][9]  
                price_per_share =  price(knockoutprice,full_df.values[len(full_df.values) -1 ][9]  )
                pos = ["Short",knockoutprice, 0.71 *price_per_share  ]
         
    
            elif action >=10:
                action = 7 + (10 -action )
                x12 = 0.89 + 0.01* action
                knockoutprice = x12 *  full_df.values[len(full_df.values) -1 ][9]

                price_per_share =  price(full_df.values[len(full_df.values) -1 ][9], knockoutprice  )
                pos = ["Long",knockoutprice, 0.71 *price_per_share  ]
            stored_action = {
            "Index" : "DAX40",
            "Option type" : "Knockout without fixed stopp loss",
            "Direction"   : pos[0],
            "Knockout" : pos[1]

        }

        
    else:
        stored_action = {
            "Index" : "DAX40",
            "Option type" : "Undefined",
           
        }
    

    

    content = {"title" : "MLStrat day" , "investment" : stored_action}
   
    client.invoke(
    FunctionName='sendMail',
    InvocationType='Event',
    Payload=json.dumps(content))

    return {
    "value" : content
   }
   


   





 