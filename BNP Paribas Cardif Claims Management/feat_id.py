## Wasting Time on Kaggle Competitions
## ---------------------------------------------------------
## BNP Paribas Cardif Claims Management
## Exploratory Data Analysis (EDA)
## ---------------------------------------------------------
## Script to Identify useful features.
## Searches breadth first and drops features one by one
## based on how they perform.

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss

import pandas as pd

def convert_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            _col = "_%s" % (col)
            values = df[col].unique()
            _values = dict(zip(values, range(len(values))))
            df[_col] = df[col].map(_values).astype(int)
            df = df.drop(col, axis = 1)

    return df

def data():
    print "data"
    train_df = convert_to_numeric(pd.read_csv("train.csv")).fillna(-1)

    cut = int(len(train_df) * 0.8)
    validation_df = train_df[cut:]
    train_df = train_df[:cut]

    return train_df, validation_df

def model(df):
    print "model"
    C = AdaBoostClassifier()
    model = C.fit(df.values[:,2:],
                  df.values[:,1])
    return model

def predict(model, df):
    print "predict"
    result =  model.predict_proba(df.values[:,2:])
    print result[:,1]
    return result[:,1]

train_df, validation_df = data()
result = log_loss(validation_df.values[:,1].astype(int),
                  predict(model(train_df), validation_df))
print result
