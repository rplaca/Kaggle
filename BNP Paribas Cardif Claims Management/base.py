## Wasting Time on Kaggle Competitions
## ---------------------------------------------------------
## BNP Paribas Cardif Claims Management
## Exploratory Data Analysis (EDA)
## ---------------------------------------------------------
## A base predictor using all available features as-is
## LB Score : 0.68237
##
## Removing 17 redundant columns (corr value > 0.95) scores
## the same so let's keep those out of the data sets.
## LB Score : 0.68237
##
## Removing 4 noisy columns scores better locally but worse
## on the Kaggle Leaderboard.
## LB Score : 0.68382

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
    cols = ['v8', 'v33', 'v46', 'v53', 'v54', 'v60', 'v63', 'v64', 'v76',
            'v89', 'v95', 'v96', 'v105', 'v116', 'v118', 'v121', 'v128']
    cols += ['_v79', '_v110', 'v34', '_v31']

    print "loading train.csv"
    train_df = convert_to_numeric(pd.read_csv("train.csv")).fillna(-1)
    print "dropping %d columns" % len(cols)
    train_df = train_df.drop(cols, axis = 1)

    print "loading test.csv"
    test_df = convert_to_numeric(pd.read_csv("test.csv")).fillna(-1)
    print "dropping %d columns" % len(cols)
    test_df = test_df.drop(cols, axis = 1)

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
submission_df = pd.DataFrame(predict(model(train_df), test_df)[:,1],
                             columns = ['PredictedProb'])
submission_df.insert(0, 'ID', test_df.values[:,0].astype(int))

print submission_df

submission_df.to_csv("submission.csv", index = False)
print "Done"
