import pandas as pd
import numpy as np
import os
import json
import boto3
import urllib.request
from datetime import datetime
from datetime import date
from datetime import timedelta
import time
import random
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


def download():
    start_date = "2020-11-05"
    datetime_start = datetime.strptime(start_date,"%Y-%m-%d")
    today = date.today()


    datetime_end = datetime.strptime(today.strftime("%d-%m-%Y"),"%d-%m-%Y")

    difference = (datetime_end - datetime_start)
    period = 1604620800 + difference.days * 86400 
    
    url = "https://query1.finance.yahoo.com/v7/finance/download/%5EGDAXI?period1=567820800&period2={periodend}&interval=1d&events=history&includeAdjustedClose=true".format(periodend = period) 
    urllib.request.urlretrieve(url, '/tmp/tmp.csv')

def ready_csv():
  
    path = "/tmp/tmp.csv"
    dax_tmp = pd.read_csv(path,usecols=['Date','Open',"High","Low","Close"],  parse_dates=['Date'])
    last_row = dax_tmp.iloc[len(dax_tmp.values)-1]
    path = "/tmp/dax.csv"
    dax = pd.read_csv(path,usecols=['Date','Open',"High","Low","Close"],  parse_dates=['Date'])
    dax.at[len(dax.values)-1, "Open" ] =  last_row["Open"]
    dax.at[len(dax.values)-1, "High" ] =  last_row["High"]
    dax.at[len(dax.values)-1,"Low" ] =  last_row["Low"]
    dax.at[len(dax.values)-1,"Close" ] =  last_row["Close"]
    df2 = {'Date': calculate_new_date( last_row["Date"]  ), 'Open': last_row["Open"], "High" :last_row["High"] , "Low" : last_row["Low"] ,'Close': last_row["Close"]}
    dax = pd.concat([dax, pd.DataFrame([df2])], ignore_index=True)
    dax.to_csv("/tmp/dax.csv", index = False)


  
    
def calculate_new_date(datestart):
    
    path = "/tmp/feiertage.csv"
    dateparse = lambda x: datetime.strptime(x, '%d.%m.%Y')
    feiertage = pd.read_csv(path,  parse_dates=['Date'],date_parser=dateparse)
    weekday = datestart.weekday()
    
    while True:
        datestart = datestart + timedelta(days =1)
        weekday = datestart.weekday()
        if weekday <= 4:
            break

    if len(((feiertage['Date'].where(feiertage['Date'] == datestart)).dropna()).values) == 1:
        datestart = calculate_new_date(datestart)
    return datestart



def handler(event, context):
    path = "/tmp/feiertage.csv"
    dateparse = lambda x: datetime.strptime(x, '%d.%m.%Y')
    feiertage = pd.read_csv(path,  parse_dates=['Date'],date_parser=dateparse)
    today = date.today()
    datestart = datetime.strptime(today.strftime("%d-%m-%Y"),"%d-%m-%Y")
    if len(((feiertage['Date'].where(feiertage['Date'] == datestart)).dropna()).values) == 0 and today.weekday() <=4:
        sec = random.randint(0,60)
        print("Start downloading in {} seconds".format(sec))
        time.sleep(sec)
        download()
        ready_csv()

        analysisfile = open("/tmp/dax.csv", "r")
        data = analysisfile.read()
        analysisfile.close()
        s3.Object('REDACTED_BUCKET', 'dax.csv').put(Body=data)

        
   
       
  


   





 