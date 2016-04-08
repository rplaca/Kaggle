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
    cols = ['v8', 'v33', 'v46', 'v53', 'v54', 'v60', 'v63', 'v64', 'v76',
            'v89', 'v95', 'v96', 'v105', 'v116', 'v118', 'v121', 'v128']

    print "data"
    train_df = convert_to_numeric(pd.read_csv("train.csv")).fillna(-1)

    print "dropping %d columns" % len(cols)
    train_df = train_df.drop(cols, axis = 1)

    print "chipping out a validation set"
    cut = int(len(train_df) * 0.8)
    validation_df = train_df[cut:]
    train_df = train_df[:cut]

    print "done. let the madness begin"
    return train_df, validation_df

def model(df):
    C = AdaBoostClassifier()
    model = C.fit(df.values[:,2:],
                  df.values[:,1])
    return model

def predict(model, df):
    result =  model.predict_proba(df.values[:,2:])
    return result[:,1]

train_df, validation_df = data()

curr_score = log_loss(validation_df.values[:,1],
                      predict(model(train_df), validation_df))
base_score = curr_score
print [], base_score, "*"

best_score = 1
cols = train_df.columns[2:]
drop = []
while True:
    curr_col = None
    for col in cols:
        if col not in drop:
            _train_df = train_df.drop(drop + [col,], axis = 1)
            _validation_df = validation_df.drop(drop + [col,], axis = 1)
            result = log_loss(validation_df.values[:,1].astype(int),
                              predict(model(_train_df), _validation_df))
            print drop + [col,], result,
            if result < curr_score:
                curr_col = col
                curr_score = result
                print "*",
            print

    if curr_score < best_score:
        best_score = curr_score
        drop.append(curr_col)

    else:
        break

print "no more improvement."
print "base score = %s" % (base_score)
print "best score = %s" % (best_score)

print drop
print "stopping."
