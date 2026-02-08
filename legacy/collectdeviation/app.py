import pandas as pd
import numpy as np
import json
import boto3
from datetime import datetime
import time
client = boto3.client('lambda')
s3 = boto3.resource('s3')
dax = (s3.Object("REDACTED_BUCKET", "dax.csv").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/dax.csv", "w+")
analysisfile.write(dax)
analysisfile.close()

dax = (s3.Object("REDACTED_BUCKET", "feiertage.csv").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/feiertage.csv", "w+")
analysisfile.write(dax)
analysisfile.close()

dax = (s3.Object("REDACTED_BUCKET", "deviation.csv").get()['Body'].read().decode('utf-8') )
analysisfile = open("/tmp/deviation.csv", "w+")
analysisfile.write(dax)
analysisfile.close()

def append_deviation(prediction):
    path = "/tmp/dax.csv"
    dax = pd.read_csv(path,usecols=['Date','Open',"High","Low","Close"],  parse_dates=['Date'])
    last_row = dax.iloc[len(dax.values)-1]
    date_last = last_row["Date"]
    new_date = str(date_last.strftime("%Y-%m-%d"))
    
    deviation = [ np.mean( prediction[:,i]  ) for i in range(0,4)   ]
    deviation_sigma = [ np.sqrt( np.var(prediction[:,i])  ) / np.sqrt(10) for i in range(0,4)   ]
    analysisfile = open("/tmp/deviation.csv", "r")
    file = analysisfile.read()
    analysisfile.close()
    analysisfile = open("/tmp/deviation.csv", "w+")
    file = file + new_date + "," + str(deviation[0]) + "," + str(deviation[1]) + "," + str(deviation[2])+ "," + str(deviation[3])+ "," + str(deviation_sigma[0]) + "," + str(deviation_sigma[1]) + "," + str(deviation_sigma[2])+ "," + str(deviation_sigma[3])+  "\n"
    analysisfile.write(file)
    analysisfile.close()

    analysisfile = open("/tmp/deviation.csv", "r")
    data = analysisfile.read()
    analysisfile.close()
    s3.Object('REDACTED_BUCKET', 'deviation.csv').put(Body=data)

    

 
   



def handler(event, context):

    bk = s3.Bucket("modelpredictions")
    while True:
        time_now = datetime.now()
        last_modified = [(time_now - obj.last_modified.replace(tzinfo=None) ).days*24. +(time_now - obj.last_modified.replace(tzinfo=None) ).seconds/3600.   for obj in bk.objects.all()]
        training_time = [ elem < 2 for elem in last_modified   ]
  
        var = True
        var = var and len(training_time)==10
        for elem in training_time:
            var = var and elem
        if var and len(training_time)>0:
            break
        time.sleep(1)

    

    list1 = []
    for obj in bk.objects.all():
         
        list1.append(json.loads(s3.Object("modelpredictions", obj.key).get()['Body'].read().decode('utf-8')))
    

    append_deviation(np.array(list1))

    content = {"title" : "MLStratnight" }
    client = boto3.client('lambda')
    client.invoke(
    FunctionName='start_night_prediction',
    InvocationType='Event',
    Payload=json.dumps(content))
    return {
        "state" : "deviation written, reinf started"

    }

    


  