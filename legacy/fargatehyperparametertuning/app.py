import pandas as pd
import numpy as np
import json
df1 = pd.read_csv("s3://REDACTED_BUCKET/dayactions.csv", parse_dates=['Date'])
df1["action"].value_counts() / (df1["action"].value_counts().sum() )
path = "s3://REDACTED_BUCKET/dax.csv"

df = pd.read_csv(path,usecols=['Date','Open',"High","Low","Close"],  parse_dates=['Date'])

df = df.dropna()


def one_hot_encode_column(data: pd.DataFrame, col, max_classes=None):
    df = data.copy()
    
    if max_classes is not None:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
         
        missing_cols = set(range(max_classes + 1)) - set(df[col].unique())
        for col_num in missing_cols:
            dummies[col +"_"+ str(col_num)] = 0
        
        # Custom sort function to ensure correct column order
        dummies = dummies[sorted(dummies.columns, key=lambda x: int(x.split("_")[1]))]
    else:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)

    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[col])

    return df





 