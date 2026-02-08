MAX_ACCOUNT_BALANCE = 2147483647
INITIAL_ACCOUNT_BALANCE = 1000
Maximum_invest = 750000.
Max_obsv = 500
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
 
import boto3
s3 = boto3.resource('s3')
dax = (s3.Object("REDACTED_BUCKET", "dax.csv").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/dax.csv", "w+")
analysisfile.write(dax)
analysisfile.close()
dax = (s3.Object("REDACTED_BUCKET", "deviation.csv").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/deviation.csv", "w+")
analysisfile.write(dax)
analysisfile.close()

def load_models(structure, model):
    tmp = (s3.Object("modelsreinf", "night/{}".format(structure)).get()['Body'].read().decode('utf-8') )
    analysisfile = open("/tmp/"+structure, "w+")
    analysisfile.write(tmp)
    analysisfile.close()
    
    
    tmp = (s3.Object("modelsreinf", "night/{}.index".format(model)).get()['Body'].read()  )
    analysisfile = open("/tmp/{}.index".format(model), 'wb')
    analysisfile.write(tmp)
    analysisfile.close()
    
    tmp = (s3.Object("modelsreinf", "night/{}.data-00000-of-00002".format(model)).get()['Body'].read()  )
    analysisfile = open("/tmp/{}.data-00000-of-00002".format(model), 'wb')
    analysisfile.write(tmp)
    analysisfile.close()
    
    tmp = (s3.Object("modelsreinf", "night/{}.data-00001-of-00002".format(model)).get()['Body'].read()  )
    analysisfile = open("/tmp/{}.data-00001-of-00002".format(model), 'wb')
    analysisfile.write(tmp)
    analysisfile.close()

class stock(gym.Env):
    
    metadata = {'render.modes': ['human']}
    

    def __init__(self, df,train=True, stop = 0.5):
        super(stock, self).__init__()
        
        if train:
            self.df = df.iloc[0:548].values
        else:
            self.df = df.values
        self.tmp = []
        self.stop = stop
        self.timestep = 1
        self.reward_range = (-MAX_ACCOUNT_BALANCE, MAX_ACCOUNT_BALANCE)
        self.budget = INITIAL_ACCOUNT_BALANCE
        
        n_actions = 19
        self.action_space = spaces.Discrete(n_actions)
        
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(1,49), dtype=np.float32)
        
    def collect_data(self):
        x = [] 
        index = self.timestep
        values = self.df
        for i in range(index,index+1):
            open_close_dev = (values[i+1][1] - values[i][12]) /values[i+1][1]
            high_close_dev = (values[i+1][2] - values[i][12]) /values[i+1][2]
            low_close_dev = (values[i+1][3] - values[i][12]) /values[i+1][3]
            close_close_dev = (values[i+1][4] - values[i][12]) /values[i+1][4]
            open_conf = (values[i-1][13] - values[i][13]) / values[i-1][13]
        
            open_act = (values[i-1][9] - values[i][9]) / values[i-1][9]
            high_act = (values[i-1][10] - values[i][10]) / values[i-1][10]
            low_act = (values[i-1][11] - values[i][11]) / values[i-1][11]
            close_act = (values[i-1][12] - values[i][12]) / values[i-1][12]
    
            open_pred = (values[i-1][1] - values[i][1]) / values[i-1][1]
            high_pred = (values[i-1][2] - values[i][2]) / values[i-1][2]
            low_pred = (values[i-1][3] - values[i][3]) / values[i-1][3]
            close_pred = (values[i-1][4] - values[i][4]) / values[i-1][4]
            dayinfo = [float( i == values[i,17]) for i in range(0,5) ] +  [float( i == values[i,18]) for i in range(0,31)]
            tmp = [open_pred, high_pred,low_pred, close_pred, open_act, high_act, low_act, close_act, open_close_dev
                 , high_close_dev, low_close_dev, close_close_dev, open_conf/Max_obsv]
            tmp = tmp + dayinfo
            x.append(tmp)
       
       
        return np.array(x) 

    def price(self,indexvalue : float, knockoutprice : float  ):
        return max((indexvalue - knockoutprice)* 0.01 , 0 )
        
    def buy_long(self,budget, x12,x13 ):
        i = self.timestep
        values = self.df
       
        knockoutprice = x12 * min(values[i+1][1], values[i][12]  )
        
        price_per_share = self.price(values[i][12], knockoutprice  )
    
        allocated_money = min(budget,x13 )
        budget = budget - allocated_money
   
        number_of_shares = int( (allocated_money-1)/ price_per_share   )
        allocated_money = allocated_money - number_of_shares * price_per_share -1
        budget = budget + allocated_money
        return budget, ["long",knockoutprice,number_of_shares,price_per_share]

    def buy_short(self,budget, x12,x13 ):
        i = self.timestep
        values = self.df
        
        knockoutprice = x12 *max(values[i+1][1], values[i][12]  )
        
        price_per_share = self.price( knockoutprice, values[i][12]  )
        allocated_money = min(budget,x13 )
        budget = budget - allocated_money
        number_of_shares = int( (allocated_money-1)/ price_per_share   )
        allocated_money = allocated_money - number_of_shares * price_per_share -1
        budget = budget + allocated_money
        return budget, ["short",knockoutprice,number_of_shares,price_per_share]
    
    def sell(self,budget, pos):
        i = self.timestep+1
        stop = self.stop
        values = self.df
        if(len(pos) == 0 ):
            return budget,[]
        if pos[0] == "long":
            price_per_share = self.price(values[i][9], pos[1]  )
            if self.price(values[i][9], pos[1]  ) < pos[3] * stop:
                price_per_share = pos[3] * stop
            if price_per_share == 0:
                return budget, []
            price_per_share = price_per_share - 0.01
            sell_money =  price_per_share * pos[2]
        
            if sell_money == 0:
                return budget, []
            budget = budget + sell_money -1
            return budget, []
        if pos[0] == "short":
            price_per_share = self.price( pos[1],values[i][9]  )
            if  self.price( pos[1],values[i][9]  ) < pos[3] * stop:
                price_per_share = pos[3] * stop
            if price_per_share == 0:
                return budget, []
            price_per_share = price_per_share - 0.01
            sell_money =  price_per_share * pos[2]
        
            if sell_money == 0:
                return budget, []
            budget = budget + sell_money -1
            return budget, []
    
    
    def take_action(self,step, action):
        pos = []
        previous_budget = self.budget
        if action < 9:
            x12 = 0.90 + 0.01* action
            self.budget, pos = self.buy_long(self.budget, x12,Maximum_invest )
            
        if action > 9:
            x12 = 1.02 + 0.01* (   action-10)
            self.budget, pos = self.buy_short(self.budget, x12,Maximum_invest )
        pos_tmp = pos
        self.budget, pos = self.sell(self.budget, pos )
        self.tmp.append([action,self.budget,self.df[self.timestep,0],pos_tmp])
        return (self.budget - previous_budget) / previous_budget*100
            
            
        
    
    
    def step(self, action):
        
        reward = self.take_action(self,action)
        self.timestep += 1
        
        
        done = self.budget <= 25 or self.timestep == len(self.df) -1
        if self.timestep == len(self.df) -1:
            self.timestep = 1
        
        obs = self.collect_data()
        return obs, reward, done, {}
        
        
        
    def reset(self):
        self.budget = INITIAL_ACCOUNT_BALANCE
        self.timestep = 1
        self.tmp = []
        return self.collect_data()
    
    
        
         
        
    
    def render(self, mode='human'):
        profit = self.budget - INITIAL_ACCOUNT_BALANCE
        
        print(f'Step: {self.timestep}')
        
        print(self.tmp[len(self.tmp)-1])
        print(f'Balance: {self.budget}')
        print(f'Profit: {profit}')
        
 
    def close(self):
        pass
  
  

path = "/tmp/deviation.csv"

df = pd.read_csv(path,  parse_dates=['Date'])
df = df.rename(columns={"Open" : "Open_conf", "High" : "High_conf", "Low" : "Low_conf", "Close" : "Close_conf"})

path = "/tmp/dax.csv"

dax = pd.read_csv(path,usecols=['Date','Open',"High","Low","Close"],  parse_dates=['Date'])

#dax = dax.dropna()


full_df =  pd.merge(df, dax, on="Date")
full_df["Open_dev"] = (full_df["Open"]- full_df["Open_conf"] ) / full_df["Open_sigma"]
full_df["High_dev"] = (full_df["High"]- full_df["High_conf"] ) / full_df["High_sigma"]
full_df["Low_dev"] = (full_df["Low"]- full_df["Low_conf"] ) / full_df["Low_sigma"]
full_df["Close_dev"] = (full_df["Close"]- full_df["Close_conf"] ) / full_df["Close_Sigma"]
full_df['weekday'] = full_df["Date"].dt.weekday.astype(np.int8) 
full_df['monthday'] = full_df["Date"].dt.day.astype(np.int8) -1


# In[6]:

def lstmblock(input_channel,units,drop, leaky,dropout_recurrent,return_sequences = False ):
    lstm = LSTM(int(units), return_sequences=return_sequences,dropout = drop, recurrent_dropout=dropout_recurrent)(input_channel)
    lstm = LeakyReLU(alpha=leaky)(lstm)
    lstm = BatchNormalization()(lstm)
    return lstm
    
    
def denseblock(input_channel, units,drop,leaky):
    dense = Dense(int(units))(input_channel)
    dense = LeakyReLU(alpha=leaky)(dense)
    dense = Dropout(rate=drop)(dense)
    dense = BatchNormalization()(dense)
    return dense
    
def lstmparameters(number : int):
    return { "x"+str(number) : (2,256) , "leaky" + str(number) : (0.1, 0.9), "drop" +  str(number): (0,0.5) , "drop_recurrent" +  str(number): (0,0.5)  }

def denseparameter(number : int):
    return {
       "x"+str(number) : (2,256) ,
         "leaky" + str(number) : (0.1, 0.9),
        "drop" +  str(number): (0,0.5)
        
        
    }
def modelstructure(number_of_layers, number_heads, number_of_head_layers, number_of_lstm_layers ):
    structure = {"layers": {}}
    if number_of_head_layers >= number_of_layers:
        number_of_head_layers = number_of_layers
    layer = 1
    head_layer = 1
    lstm_layer = 1 
    counter_of_blocks = 1
    for i in range(1,number_of_layers +1):
        if number_of_head_layers -head_layer >= 0:
            if number_of_lstm_layers - lstm_layer >= 0: 
                structure["layers"]["layer_"+ str(layer)] = ["lstm",[ lstmparameters( i + counter_of_blocks) for i in range(0,number_heads)    ]]
                lstm_layer += 1
                layer += 1
                counter_of_blocks +=  number_heads
                head_layer += 1
                continue
            else:
                structure["layers"]["layer_"+ str(layer)] = ["dense",[ denseparameter( i + counter_of_blocks) for i in range(0,number_heads)    ]]
                layer += 1
                counter_of_blocks +=  number_heads
                head_layer += 1
                continue
        if number_of_lstm_layers - lstm_layer >= 0: 
            structure["layers"]["layer_"+ str(layer)] = ["lstm",[ lstmparameters(  counter_of_blocks)   ]]
            lstm_layer += 1
            layer += 1
            counter_of_blocks += 1
            continue
        else:
            structure["layers"]["layer_"+ str(layer)] = ["dense",[ denseparameter( counter_of_blocks)    ]]
            layer += 1
             
            counter_of_blocks +=  1
            continue
             
            
        
            
        
        
    return structure
def parameters(structure):
    params = {}
    for elem in structure["layers"]:
        for elem2 in structure["layers"][elem][1]:
           
        
        
        
        
            params.update(elem2)
    return params
def build_model(parameters, structure,number_of_layers ,number_heads, number_of_head_layers,number_of_lstm_layers):
    layer = 1
    p = parameters
    counter_of_blocks = 1
    lstm_layer = 1
    head_layer = 1
    env = stock(full_df,True)
    num_actions = env.action_space.n
    input_layer = Input(shape= (1,) + env.observation_space.shape)
    if number_of_head_layers >= number_of_layers:
        number_of_head_layers = number_of_layers
    if number_of_lstm_layers >number_of_layers:
        number_of_lstm_layers = number_of_layers
        
    if number_of_head_layers == 0:
        number_heads = 1
    reshape = tf.keras.layers.Reshape((1,49), input_shape=(1,) + env.observation_space.shape)(input_layer)
    model_layers = [[reshape for elem in range(0, number_heads)]]
    for i, elem in enumerate(structure["layers"]):
        
        if number_of_head_layers -head_layer >= 0:
            if number_of_lstm_layers - lstm_layer >= 0:
                new_layers = []
                
                for  j, elem in enumerate(model_layers[i]):
                    
                    new_layers.append(lstmblock(model_layers[i][j],p["x"+str(j + counter_of_blocks)],p["drop"+str(j + counter_of_blocks)], p["leaky"+str(j + counter_of_blocks)],p["drop_recurrent"+str(j + counter_of_blocks)],return_sequences = number_of_lstm_layers - lstm_layer!=0 )   )
                model_layers.append(new_layers)
                lstm_layer += 1
                layer += 1
                counter_of_blocks +=  number_heads
                head_layer += 1
               
                if number_of_head_layers -head_layer < 0:
                    
                   
                    model_layers.append(  tf.keras.layers.concatenate(model_layers[i+1])  )
  
                continue
            else:
                new_layers = []
                for  j, elem in enumerate(model_layers[i]):
                    new_layers.append(denseblock(model_layers[i][j],p["x"+str(j + counter_of_blocks)], p["drop"+str(j + counter_of_blocks)] ,p["leaky"+str(j + counter_of_blocks)])   )
                model_layers.append(new_layers)
                layer += 1
                counter_of_blocks +=  number_heads
                head_layer += 1
                if number_of_head_layers -head_layer < 0:
                   
                    model_layers.append(  tf.keras.layers.concatenate(model_layers[i+1])  )
            
                continue
        else:
            if number_of_lstm_layers - lstm_layer >= 0: 
                model_layers.append(lstmblock(model_layers[i+1],p["x"+str( counter_of_blocks)],p["drop"+str( counter_of_blocks)], p["leaky"+str(counter_of_blocks)],p["drop_recurrent"+str(counter_of_blocks)],return_sequences = number_of_lstm_layers - lstm_layer!=0 )   )
                lstm_layer += 1
                layer += 1
                counter_of_blocks += 1
                continue
            else:
                #print(model_layers[i+1])
                model_layers.append(denseblock(model_layers[i+1],p["x"+str(  counter_of_blocks)], p["drop"+str( counter_of_blocks)],p["leaky"+str( counter_of_blocks)] )   )
                layer += 1
                counter_of_blocks += 1 
                continue
                
            
                
            
            
         
            
            
            
        
    
    
    final = Dense(num_actions, name='output')(model_layers[ len( model_layers )-1])
    
    model = Model(inputs=input_layer, outputs=final)
    return model
def load_model(csvname, weightname ):
 
    
    env = stock(full_df,True)
    num_actions = env.action_space.n
    np.random.seed(123)
    tf.keras.backend.clear_session()
    help = open("/tmp/"+csvname, 'r').read()
    model_parameter = eval(help)
    ms = model_parameter["structure"]
    params  = model_parameter["parameters"]
    
    
    
    neuralnetwork =modelstructure(ms[0],ms[1],ms[2],ms[3])
    model = build_model(params, neuralnetwork,ms[0],ms[1],ms[2],ms[3])


    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=100,
                 gamma=.99, target_model_update=10000, 
               train_interval=4,  policy=policy,delta_clip=1.,
               enable_dueling_network=True,
               dueling_type='avg')
    dqn.compile(Adam(lr=.00025))
    dqn.load_weights("/tmp/"+weightname)
    return dqn
   
def collect_data(index,df):
    x = [] 
  
    values = df.values
    for i in range(index,index+1):
        open_close_dev = (values[i+1][1] - values[i][12]) /values[i+1][1]
        high_close_dev = (values[i+1][2] - values[i][12]) /values[i+1][2]
        low_close_dev = (values[i+1][3] - values[i][12]) /values[i+1][3]
        close_close_dev = (values[i+1][4] - values[i][12]) /values[i+1][4]
        open_conf = (values[i-1][13] - values[i][13]) / values[i-1][13]
        
        open_act = (values[i-1][9] - values[i][9]) / values[i-1][9]
        high_act = (values[i-1][10] - values[i][10]) / values[i-1][10]
        low_act = (values[i-1][11] - values[i][11]) / values[i-1][11]
        close_act = (values[i-1][12] - values[i][12]) / values[i-1][12]
    
        open_pred = (values[i-1][1] - values[i][1]) / values[i-1][1]
        high_pred = (values[i-1][2] - values[i][2]) / values[i-1][2]
        low_pred = (values[i-1][3] - values[i][3]) / values[i-1][3]
        close_pred = (values[i-1][4] - values[i][4]) / values[i-1][4]
        dayinfo = [float( i == values[i,17]) for i in range(0,5) ] +  [float( i == values[i,18]) for i in range(0,31)]
        tmp = [open_pred, high_pred,low_pred, close_pred, open_act, high_act, low_act, close_act, open_close_dev
                 , high_close_dev, low_close_dev, close_close_dev, open_conf/Max_obsv]
        tmp = tmp + dayinfo
        x.append(tmp)
       
       
    return np.array(x) 
    















def handler(event, context):

    param1 = event["structure"]
    param2 = event["weights"]

    
    key = event["key"]
    load_models(param1,param2)
    dqn = load_model(param1,param2)
    tmp =collect_data(len(full_df.values)-2,full_df)
    prediction =dqn.model.predict(tmp.reshape(1,1,1,49))[0]
    maxval = -1000
    j = 0
    for i, elem in enumerate(prediction):
     
        if elem > maxval:
            j = i
            maxval = elem
    
    s3.Object('REDACTED_BUCKET', 'night/{}/{}/{}/{}.txt'.format(
        key[0],key[1],key[2],event["modelnumber"]

    )
    
    ).put(Body=str(j))

    return {
        "test" : j

    }