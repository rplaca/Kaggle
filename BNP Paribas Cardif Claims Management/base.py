## Wasting Time on Kaggle Competitions
## ---------------------------------------------------------
## BNP Paribas Cardif Claims Management
## Exploratory Data Analysis (EDA)
## ---------------------------------------------------------
## A base predictor using all available features as-is
## LB Score : 0.68237

from sklearn.ensemble import AdaBoostClassifier

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
    print "loading train.csv"
    train_df = convert_to_numeric(pd.read_csv("train.csv")).fillna(-1)
    print "loading test.csv"
    test_df = convert_to_numeric(pd.read_csv("test.csv")).fillna(-1)

    return train_df, test_df

def model(train_df):
    C = AdaBoostClassifier()
    print "Training a classifier"

    return C.fit(train_df.values[:,2:],
                 train_df.values[:,1])

def predict(model, test_df):
    print "Predicting"
    return model.predict_proba(test_df.values[:,1:])

train_df, test_df = data()
# ID,

submission_df = pd.DataFrame(predict(model(train_df), test_df)[:,1],
                             columns = ['PredictedProb'])
submission_df.insert(0, 'ID', test_df.values[:,0].astype(int))

print submission_df

submission_df.to_csv("submission.csv", index = False)
print "Done"
