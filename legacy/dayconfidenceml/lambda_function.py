import json
import boto3
from decimal import Decimal
db = boto3.resource("dynamodb")
table = db.Table("turnarount")
client = boto3.client('lambda')

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

table_name = 'turnarount'
primary_key_name = 'val'
primary_key_value = 'turnaroundday'
def lambda_handler(event, context):
    

    alpha_opt =  table.get_item( Key={'val': "alpha_opt"})["Item"]["value"]
    performancemean =  table.get_item( Key={'val': "performancemean"})["Item"]["value"]
    performance_oppossitemean =  table.get_item( Key={'val': "performance_oppossitemean"})["Item"]["value"]
    performancemean_diff =  table.get_item( Key={'val': "performancemean_diff"})["Item"]["value"]
    performance_oppossitemean_diff =  table.get_item( Key={'val': "performance_oppossitemean_diff"})["Item"]["value"]
    last_rsi_close =  table.get_item( Key={'val': "RSI_Close"})["Item"]["value"]
    last_rsi_open =  table.get_item( Key={'val': "RSI_Open"})["Item"]["value"]
    last_rsi_close_diff =  table.get_item( Key={'val': "RSI_Close_diff"})["Item"]["value"]
    last_rsi_open_diff =  table.get_item( Key={'val': "RSI_Open_diff"})["Item"]["value"]
    last_rsi_close_14d =  table.get_item( Key={'val': "RSI_Close_14d"})["Item"]["value"]
    last_ema_ratio_close =  table.get_item( Key={'val': "EMA_Ratio_Close"})["Item"]["value"] 
    last_bb_bandwidth = table.get_item( Key={'val': "BB_Bandwidth"})["Item"]["value"] 
    last_bb_percent =  table.get_item( Key={'val': "BB_Percent"})["Item"]["value"] 
    last_stochastic_scaled =  table.get_item( Key={'val': "Stochastic_Scaled"})["Item"]["value"] 
    last_ma_crossover_signal =  table.get_item( Key={'val': "MA_Crossover_Signal"})["Item"]["value"]
    last_market_variation_tanh =  table.get_item( Key={'val': "Market_Variation_Tanh"})["Item"]["value"]
    last_market_variation_tanh_diff =  table.get_item( Key={'val': "Market_Variation_Tanh_Diff"})["Item"]["value"]
    print(event)
    event["performancemean"] = float(performancemean)
    event["performance_oppossitemean"] = float(performance_oppossitemean)
    event["RSI_Close"] = float(last_rsi_close)
    event["RSI_Close_14d"] = float(last_rsi_close_14d)
    event["RSI_Open"] = float(last_rsi_open)
    
    
    event["RSI_Close_diff"] = float(last_rsi_close_diff)
    event["RSI_Open_diff"] = float(last_rsi_open_diff)
    event["performancemean_diff"] = float(performancemean_diff)
    event["performance_oppossitemean_diff"] = float(performance_oppossitemean_diff)
    
    event["EMA_Ratio_Close"] = float (last_ema_ratio_close)
    event["BB_Bandwidth"] =float (last_bb_bandwidth )
    event["BB_Percent"] = float(last_bb_percent )
    event["Stochastic_Scaled"] = float(last_stochastic_scaled)
    event["MA_Crossover_Signal"] = float(last_ma_crossover_signal)
    
    event['Market_Variation_Tanh'] = float(last_market_variation_tanh)
    event['Market_Variation_Tanh_diff'] = float(last_market_variation_tanh_diff)

    responce = client.invoke(
            FunctionName='decisiondayconfidence',
            InvocationType='RequestResponse',
            Payload=json.dumps(event))
    confidence= json.loads(responce["Payload"].read().decode('utf-8'))
    print(confidence)
    
    response = put_value_in_dynamodb(table_name, primary_key_name, "invest",1)
    if confidence["win_prop"] < alpha_opt:
         return { "loss_prop" : 0.95,        "win_prop" : 0.05 }
    else:
        return { "loss_prop" : 0.05,        "win_prop" : 0.95 }
        
   
