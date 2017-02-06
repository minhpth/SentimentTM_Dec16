# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# AGE CLASSIFYING MODEL - APPLY MODEL
#------------------------------------------------------------------------------

# Functions:
# [1] ...

# Version: 2.0
# Last edited: 21 Dec 2016
# Edited by: Minh PHAN

#------------------------------------------------------------------------------
# Global variables and settings
#------------------------------------------------------------------------------

# Seting working directory
import os
os.chdir('D:\\SentimentTM\\22_age_model')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
np.random.seed(0)

#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'

import pickle

target_bins = [0,20,35,100]

#------------------------------------------------------------------------------
# Self-defined functions
#------------------------------------------------------------------------------

# Function to balance the training and testing data sets
def balanceTrainigSet(dataFull, col_target, p_balance, trainTest_balance):

    classes = np.array(dataFull[col_target].unique())
    size_clases = [dataFull[col_target].value_counts()[clas] for clas in classes]

    index_total = dataFull.index
    index_train = []

    if p_balance == -1: # 50/50
        n_split=int(np.round(np.min(size_clases) * trainTest_balance))
        for c in classes:
            index_c = dataFull[col_target][dataFull[col_target]==c].index.tolist()
            i_c_train = np.random.choice(index_c, size=n_split, replace=False).tolist()
            index_train = index_train+i_c_train
    else:
        size_min = np.min(size_clases)
        list_split = [s*trainTest_balance if s==size_min else p_balance*s*trainTest_balance for s in size_clases]
        for i in range(len(classes)):
            c = classes[i]
            n_split = int(round(list_split[i]))
            index_c = dataFull[col_target][dataFull[col_target]==c].index.tolist()
            i_c_train = np.random.choice(index_c, size=n_split, replace=False).tolist()
            index_train = index_train+i_c_train

    index_train = np.array(index_train)
    np.random.shuffle(index_train)
    index_test = np.array(list(set(index_total)-set(index_train)))
    np.random.shuffle(index_test)

    return index_train, index_test

# Function to plot features importances of a model
def feature_importance(clf,n_features):

    # Feature selection
    print(clf)
    importances = clf.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(n_features):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importance")
    plt.bar(range(n_features), importances[indices[:n_features]], align="center")
    plt.xticks(range(n_features), indices)
    plt.xlim([-1, n_features+3])
    plt.show()

# Function to apply the classification model
def apply_model(dataIn, categorical_cols, numerical_cols, user_col):

    dataOut = dataIn.copy(deep=True)

    dataIn.drop(user_col, axis=1, inplace=True)
    #dataOut['real'+target_col] = dataIn[target_col]

    # Encode catogorical part
    if len(categorical_cols) > 0:
        v = DictVectorizer(sparse=False)
        data_categorical = v.fit_transform(dataIn[categorical_cols].T.to_dict().values())

    # Normlaize numerical, no mean rescaling, not allowed for sparse
    scaler = StandardScaler()
    data_numerical = scaler.fit_transform(dataIn[numerical_cols].values)

    if len(categorical_cols) > 0:
        dataApply = np.concatenate((data_categorical, data_numerical), axis=1)
    else:
        dataApply = data_numerical

    # Load model and make prediction
    file_in = '.\\model\\output\\age_classifier_model.pkl'
    labelEncoder, xgb_clf = pickle.load(open(file_in, 'rb'))

    target_pred_xgb = xgb_clf.predict(dataApply)
    target_pred_xgb_proba = xgb_clf.predict_proba(dataApply)

    # Create target labels to inverse transform
    # Target values: [0, 20) = 1,  [20, 35) = 2, [35, 100) = 3
    target_pred_xgb_label = labelEncoder.inverse_transform(target_pred_xgb)

    dataOut['pred_age_group'] = target_pred_xgb_label
    dataOut['pred_age_group_confidence'] = target_pred_xgb_proba.max(axis=1)
    #dataOut['pred_0-20_proba'] = target_pred_xgb_proba[:,0]
    #dataOut['pred_20-35_proba'] = target_pred_xgb_proba[:,1]
    #dataOut['pred_35-100_proba'] = target_pred_xgb_proba[:,2]

    return dataOut

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == '__main__':

    fileNameIn = '.\\output\\allUsers_fullFeaturesAgeOUT_30k.tsv'
    fileNameOut = '.\\output\\allUsers_fullFeaturesAgeOUT_30k_classified.tsv'

    discretizeLexiconScore = True
    categorical_cols = ['userLexiconScore']
    numerical_cols = ['avg_isRT', 'avg_nUsers', 'avg_nLinks', 'avg_nHashtags',
                      'avg_nEmoticons', 'avg_nAllCap', 'avg_nEnlongatedW',
                      'avg_tweetLength', 'avg_wordLength', 'avg_nSingularP',
                      'ratioFollowersFriends', 'userListed_count', 'userFavourites_count']

    user_col = ['userScreen_name', 'userId']
    load_cols = categorical_cols + numerical_cols + user_col

    dataIn = pd.read_csv(fileNameIn, sep='\t', usecols=load_cols)

    # Discretize LexiconScroe and preprocess
    if discretizeLexiconScore==True:
        dataIn['userLexiconScore'][dataIn['userLexiconScore'] >= target_bins[-1]] = target_bins[-1]-1
        dataIn['userLexiconScore'][dataIn['userLexiconScore'] <= target_bins[0]] = target_bins[0]+1
        dataIn['userLexiconScore'] = pd.cut(dataIn['userLexiconScore'].astype(int), bins=target_bins, right=False)

    dataOut = apply_model(dataIn, categorical_cols, numerical_cols, user_col)
    dataOut.to_csv(fileNameOut, sep='\t', index=False, encoding='utf-8')

#------------------------------------------------------------------------------