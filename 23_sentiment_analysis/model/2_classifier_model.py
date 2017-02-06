# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# SENTIMENT MODEL - BUILD MODEL
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
os.chdir('D:\\SentimentTM\\23_sentiment_analysis\\model')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np

import time

from sklearn import preprocessing

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.dummy import DummyClassifier

from scipy import sparse
import xgboost as xgb
import pickle

#------------------------------------------------------------------------------
# Self-defined functions
#------------------------------------------------------------------------------

# Function to calculate the probability of each target class
# Can replace this function with predict_proba()
def classification_ProbClass(clf, data_test_new, labelEncoder, n_top):

    # Estimate prob of belonging to each class
    target_predProb = clf.predict_proba(data_test_new) # Matrix containing prob for each row of data_test
    idx_sort = np.argsort(-target_predProb,axis=1) # Indexes of array if sorted largest prob first
    i_top = idx_sort[:,idx_sort.shape[1]-n_top:idx_sort.shape[1]] # Index n_top predicted clases, order decreasing prob

    # Top n probabilities, decreasing order
    target_predProb_top = np.vstack([target_predProb[i,i_top[i,:]] for i in range(target_predProb.shape[0])]) # prob of top clases, order decreasing prob

    # Top n classes, convert class code to real target value
    realTop_target = labelEncoder.inverse_transform(i_top)

    # Create string output that combines pred class and prob, for easy visualization
    assign_class_prob = ([zip(realTop_target[i,:].astype(np.str), target_predProb_top[i,:].astype(np.str))
                          for i in range(target_predProb.shape[0])])

    pred_class_prob = [] # np.zeros((len(assign_class_prob),), dtype=object)
    for i in  range(len(assign_class_prob)):
        row = assign_class_prob[i]
        str_row = ''
        for j in range(len(row)):
            str_row = str_row + '='.join(row[j]) + ', '

        str_row = str_row[0:-2] # Remove last space and coma

        #pred_class_prob[i] = str_row
        pred_class_prob.append(str_row)

    return target_predProb_top, realTop_target, pred_class_prob

# Function to select best features
def feature_selection(data_train, target_train, data_test, scoring_type):

    # Fewature selection using linear svm
    tuned_parameters = [{'C': [0.05,0.1, 0.5, 1 ,5,10,50,100,500], 'penalty':["l1","l2"], 'dual': [False],'class_weight': ['auto'] }]
    clf_featureSelect = GridSearchCV(LinearSVC(), tuned_parameters, cv=10, scoring=scoring_type)
    clf_featureSelect.fit(data_train,target_train)
    data_train_new = clf_featureSelect.transform(data_train)
    data_test_new = clf_featureSelect.transform(data_test)

    return data_train_new, data_test_new, clf_featureSelect

# Function to run the classifier and save the models
def classifyTwitter(dataIn_training, dataIn_test):

    # Drop NA data
    dataIn_training = dataIn_training.dropna()
    dataIn_test = dataIn_test.dropna()
    dataIn_training = dataIn_training.reset_index(drop=True)
    dataIn_test = dataIn_test.reset_index(drop=True)

    # Enocde target variable
    # To decode: labelEncoder.inverse_transform([2, 2, 1])='neutral,..
    target_training_org = dataIn_training.polarity.values
    target_test_org = dataIn_test.polarity.values
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(target_training_org)
    target_train = labelEncoder.transform(target_training_org)
    target_test = labelEncoder.transform(target_test_org)

    # Drop target variable
    dataIn_training.drop('polarity', axis=1, inplace=True)
    dataIn_test.drop('polarity', axis=1, inplace=True)

    # Sparse data
    data_train_sparse = sparse.csr_matrix(dataIn_training.as_matrix())
    data_test_sparse = sparse.csr_matrix(dataIn_test.as_matrix())

    # Scale data
    # Need to preprocess data: norlaize--> probelm: if sparse, not possible
    # to standarize (if d-mean/std--> 0 besomes sometihng--> dense)
    # either no standarization, or standarization [0,1] o just by std
    use_scaler = False
    if use_scaler == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(data_train_sparse)
        data_train = scaler.transform(data_train_sparse)
        data_test = scaler.transform(data_test_sparse)
    else:
        data_train = data_train_sparse
        data_test = data_test_sparse
        scaler = None

    # Fewature selection
    # Need to save the original data_trin/test for the decoding of one-hot
    use_featureSelect = False
    scoring_type = 'f1_weighted'
    if use_featureSelect == True:
        start = time.time()
        data_train_new, data_test_new, clf_featureSelect = feature_selection(data_train,target_train,data_test,scoring_type)
        end = time.time()
        print('Feature Select time =', str(end-start))
        print('new shape= ', data_test_new.shape)
    else:
       data_train_new, data_test_new=data_train, data_test
       clf_featureSelect = None

    # Convert training data back to arrary
    # Sparse matrix has error when running cross validation
    data_train_new = data_train_new.toarray()
    data_test_new = data_test_new.toarray()

    # Train xgboost model with GridSearchCV
#    start = time.time()
#    xgb_model = xgb.XGBClassifier({'nthread':4, 'objective':'multi:softprob',
#                                   'num_class':3, 'random_state':0})
#    tuned_parameters = {'max_depth':[2,4,6,8],
#                        'n_estimators':[10,50,100,200,400,600]}
#    clf_xgb = GridSearchCV(xgb_model, tuned_parameters, cv=10, scoring=scoring_type,
#                           verbose=10)
#
#    clf_xgb.fit(data_train_new, target_train)
#    target_pred_xgb = clf_xgb.predict(data_test_new)
#    end = time.time()
#
#    print('xgboost classifier:')
#    print(clf_xgb.best_score_)
#    print(clf_xgb.best_params_)
#    print('clf_xgb time = ', str(end-start))
#    print(classification_report(target_test, target_pred_xgb))
#    print(confusion_matrix(target_test, target_pred_xgb))
#    acc_xgb = clf_xgb.score(data_test_new, target_test)
#    print(acc_xgb)

    # Re-train xgboost model with best params to evaluate
    xgb_best_params = xgb.XGBClassifier(nthread=4, objective='multi:softprob',
                                        seed=0, n_estimators=600, max_depth=2)
    xgb_best_params.fit(data_train_new, target_train)
    target_pred_xgb = xgb_best_params.predict(data_test_new)

    print('xgboost classifier:')
    print(classification_report(target_test, target_pred_xgb))
    print(confusion_matrix(target_test, target_pred_xgb))
    acc_xgb = xgb_best_params.score(data_test_new, target_test)
    print(acc_xgb)

    # Save xgboost model with best params
    xgb_best_params = xgb.XGBClassifier(nthread=4, objective='multi:softprob',
                                        seed=0, n_estimators=600, max_depth=2)
    train_all = np.concatenate((data_train_new, data_test_new), axis=0)
    target_all = np.concatenate((target_train, target_test), axis=0)
    xgb_best_params.fit(train_all, target_all)

    file_out = '.\\output\\sentiment_classifier_model.pkl'
    save_data = [labelEncoder, xgb_best_params]
    pickle.dump(save_data, open(file_out, 'wb'))

    # Dummy calssifier (baseline to compare)
    clf_dummy = DummyClassifier(strategy='stratified', random_state=0)
    clf_dummy.fit(data_train_new, target_train)
    target_pred_dummy = clf_dummy.predict(data_test_new)

    print('Dummy classifier:')
    print(classification_report(target_test, target_pred_dummy))
    print(confusion_matrix(target_test, target_pred_dummy))
    acc_dummy = clf_dummy.score(data_test_new, target_test)
    print(acc_dummy)

    # Create output results
    dataIn_test['polarity'] = labelEncoder.inverse_transform(target_test)
    dataOut = dataIn_test

    col_output = 'EstimatedPolarity_XGB'
    dataOut[col_output] = labelEncoder.inverse_transform(target_pred_xgb)

    n_top = 3 # chose top 3 classes (in this case only 3 in total but other cases can  bemore)
    target_predProb_top, realTop_target, pred_class_prob = classification_ProbClass(xgb_best_params, data_test_new, labelEncoder, n_top)

    col_output = 'EstimatedPolarity_XGB_prob'
    dataOut[col_output] = pred_class_prob

    return dataOut

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == '__main__':

    # Read data in and train sentiment classifier model
    fileNameIn_train = '.\\output\\tweeti-b.distFULL_featureEncoded.tsv'
    fileNameIn_test = '.\\output\\tweeti-b.dev.distFULL_featureEncoded.tsv'

    dataIn_training = pd.read_csv(fileNameIn_train, sep='\t', encoding='utf-8')
    dataIn_test = pd.read_csv(fileNameIn_test, sep='\t', encoding='utf-8')

    dataOut_test = classifyTwitter(dataIn_training, dataIn_test)
    
    # Merge and export results
    fileNameIn_test_withMsg = '.\\input\\data_challenge2013\\tweeti-b.dev.distFULL.tsv'
    fileNameOut_test_withMsg = '.\\output\\tweeti-b.dev.distFULL_classificationOut_withMSg.csv'

    dataIn_test_withMsg = pd.read_csv(fileNameIn_test_withMsg, sep='\t')
    colNames = list(dataIn_test_withMsg.columns)
    dataIn_test_withMsg['EstimatedPolarity_XGB_prob'] = dataOut_test['EstimatedPolarity_XGB_prob']
    dataIn_test_withMsg = dataIn_test_withMsg[colNames[0:3] + ['EstimatedPolarity_XGB_prob'] + [colNames[3]]]
    
    dataIn_test_withMsg.to_csv(fileNameOut_test_withMsg, index=False)

#------------------------------------------------------------------------------