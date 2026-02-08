import json
import boto3
s3 = boto3.resource('s3')
client = boto3.client('lambda')
from datetime import datetime
import numpy as np
import pandas as pd

 
 
 
   



def handler(event, context):
    bucket = s3.Bucket('REDACTED_BUCKET')
    objs = bucket.objects.filter(Prefix= "confidence/"  )
    print("filesfound")
    listoffiles = [elem.key for elem in objs]
    filecontent = []
    for elem in listoffiles:
        filecontent.append(json.loads(s3.Object("REDACTED_BUCKET", elem).get()['Body'].read().decode('utf-8')))
    print("filesread")
    up = "Points up until Strategy change"
    down = "Points down until Strategy change"
    date = [datetime.strptime(elem["date"], '%Y-%m-%d %H:%M:%S.%f') for elem in filecontent]
    upval= [elem["confidence"][up] / 250 for elem in filecontent]
    downval = [elem["confidence"][down] /250 for elem in filecontent]
    action = [elem["action"] for elem in filecontent]
    data = {"Date": np.array(date), "upval": np.array(upval), "downval" : np.array(downval), "action" : np.array(action) }
    df = pd.DataFrame(data)
    print("dataframebuild")
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.to_csv("/tmp/confidence.csv",index=False)
    analysisfile = open("/tmp/confidence.csv", "r")
    data = analysisfile.read()
    analysisfile.close()
    print("dataframesaved")
    s3.Object('REDACTED_BUCKET', 'dayactions.csv').put(Body=data)    
    return {
        "state" : "day performance updated"

    }

    


  


   





 