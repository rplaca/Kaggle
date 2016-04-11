## Wasting Time on Kaggle Competitions
## ---------------------------------------------------------
## BNP Paribas Cardif Claims Management
## Exploratory Data Analysis (EDA)
## ---------------------------------------------------------
## A first attempt at selecting a classifier. I'm not using
## the findings from feat_id.py
##
## LB Score : 0.68237 (base)
## LB Score : 0.67256 (AdaBoost(n=100,l=0.25) DecisionTree(d=1))
## LB Score : 0.53567 (RandomForest(n=100,d=1)

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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
    C = RandomForestClassifier(n_estimators = 100, max_depth = 1)
#    C = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1),
#                           n_estimators = 100,
#                           learning_rate = 0.25)
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
