import json
import boto3

def put_value_in_dynamodb(table_name, primary_key_name, primary_key_value, value):
    # Create a DynamoDB client
    dynamodb = boto3.resource('dynamodb')
    
    # Get the table
    table = dynamodb.Table(table_name)

    # Create the item to put in the table
    item = {
        primary_key_name: primary_key_value,
        'value': value
    }

    # Put the item in the table
    response = table.put_item(Item=item)

    return response



sns = boto3.client('sns')
db = boto3.resource("dynamodb")
table = db.Table("turnarount")
table_name = 'turnarount'
primary_key_name = 'val'
primary_key_value = 'turnaroundday'
table = db.Table("turnarount")
AUTO_TRADER_LAMBDA_NAME = "autotraderig"
lambda_client = boto3.client("lambda")
def lambda_handler(event, context):
    shouldbuy =  table.get_item( Key={'val': "invest"})["Item"]["value"]
    invest = event["investment"]
    title = event["title"]
    print(event)
    if title == "MLStrat day" and int(shouldbuy) == 1:
        response = put_value_in_dynamodb(table_name, primary_key_name, "Direction", event["investment"]["Direction"])
        invoke_event = {
        "action": "BUY",
        "account": "Stonks",
        "position": [event["investment"]["Direction"],event["investment"]["Knockout"]]
                }
        
        lambda_client.invoke(
            FunctionName=AUTO_TRADER_LAMBDA_NAME,
            InvocationType="Event",
            Payload=json.dumps(invoke_event)
        )
     
        
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
