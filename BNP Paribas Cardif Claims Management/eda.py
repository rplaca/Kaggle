## Wasting Time on Kaggle Competitions
## ---------------------------------------------------------
## BNP Paribas Cardif Claims Management
## Exploratory Data Analysis (EDA)
## ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    return train_df, test_df

train_df, test_df = load_data()

def print_corr_matrix(df, cut = 0.95):
    result = {}
    for col in df.columns:
        for row in df[col].index:
            val = df[col][row]
            try:
                if val > cut and row != col:
                    try:
                        result[col].append(row)
                    except KeyError:
                        result[col] = [row,]
            except:
                pass

    for k in result.keys():
        print k, result[k]

print "TRAIN"
print_corr_matrix(train_df.drop(['ID', 'target'], axis = 1).corr())

print "TEST"
print_corr_matrix(test_df.drop(['ID'], axis = 1).corr())

"""
Columns with a corr > 0.95. * = to be removed
v26 ['v60'] 
v60 ['v26'] *
v92 ['v95'] 
v95 ['v92'] *
v108 ['v128'] 
v128 ['v108'] *
v53 ['v11'] *
v11 ['v53'] 
v97 ['v118'] 
v118 ['v97'] *
v43 ['v116'] 
v116 ['v43'] *
v29 ['v96'] 
v96 ['v29'] *
v33 ['v83'] *
v105 ['v25'] * 
v121 ['v83'] *
v8 ['v46', 'v63'] *
v17 ['v64', 'v76'] 
v54 ['v25', 'v89'] * 
v76 ['v17', 'v64'] *
v83 ['v33', 'v121']
v64 ['v17', 'v76'] *
v89 ['v25', 'v54'] *
v46 ['v8', 'v25', 'v63'] * 
v63 ['v8', 'v25', 'v46'] * 
v25 ['v46', 'v54', 'v63', 'v89', 'v105']  
"""
