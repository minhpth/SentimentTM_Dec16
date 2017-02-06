# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# AGE CLASSIFYING MODEL - BUILD MODEL - BUILD MODEL
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
os.chdir('D:\\SentimentTM\\22_age_model\\model')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
np.random.seed(0)

#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import sklearn.cross_validation as crossVal
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC

#from scipy import sparse
import xgboost as xgb
import time

import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'

import pickle

target_bins = [0,20,35,100] # The age groups

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
        n_split = int(np.round(np.min(size_clases) * trainTest_balance))
        for c in classes:
            index_c = dataFull[col_target][dataFull[col_target]==c].index.tolist()
            i_c_train = np.random.choice(index_c, size=n_split, replace=False).tolist()
            index_train = index_train + i_c_train
    else:
        size_min = np.min(size_clases)
        list_split = [s*trainTest_balance if s==size_min else p_balance*s*trainTest_balance for s in size_clases]
        for i in range(len(classes)):
            c = classes[i]
            n_split = int(round(list_split[i]))
            index_c = dataFull[col_target][dataFull[col_target]==c].index.tolist()
            i_c_train = np.random.choice(index_c, size=n_split, replace=False).tolist()
            index_train = index_train + i_c_train

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

# Function to run the classifier and save the models
def main(dataIn, categorical_cols, numerical_cols, target_col, user_col):

    #dataOut = dataIn[user_col].copy(deep=True)
    dataOut = dataIn.copy(deep=True)

    dataIn.drop(user_col, axis=1, inplace=True)
    dataOut['realAge'] = dataIn[target_col]

    # Bin target variable
    dataIn[target_col] = pd.cut(dataIn[target_col].astype(int), bins=target_bins, right=False).astype(object)

    # Separate training/test randomly
    useBalance = False
    if useBalance == False:
        index_data = dataIn.index
        index_train, index_test = crossVal.train_test_split(index_data, test_size=0.3, random_state=0)
    else:
        p_balance = -1
        trainTest_balance = 0.7
        index_train, index_test = balanceTrainigSet(dataIn, target_col, p_balance, trainTest_balance)

    target_train_original = dataIn[target_col].loc[index_train].values
    target_test_original = dataIn[target_col].loc[index_test].values

    # Enocde target
    labelEncoder = LabelEncoder()
    labelEncoder.fit(target_train_original)
    target_train = labelEncoder.transform(target_train_original)
    target_test = labelEncoder.transform(target_test_original)
    print(labelEncoder.classes_)

    # Get traing test data
    data_train_original = dataIn.loc[index_train]
    data_test_original = dataIn.loc[index_test]

    # Encode catogorical variables
    if len(categorical_cols) > 0:
        v = DictVectorizer(sparse=False)
        data_trainCategorical = v.fit_transform(data_train_original[categorical_cols].T.to_dict().values())
        data_testCategorical = v.transform(data_test_original[categorical_cols].T.to_dict().values())

    # Normlaize numerical variables
    scaler = StandardScaler()
    data_trainNumerical = scaler.fit_transform(data_train_original[numerical_cols].values)
    data_testNumerical = scaler.transform(data_test_original[numerical_cols].values)

    if len(categorical_cols)>0:
        data_train = np.concatenate((data_trainCategorical,data_trainNumerical), axis=1)
        data_test = np.concatenate((data_testCategorical,data_testNumerical), axis=1)
    else:
        data_train = data_trainNumerical
        data_test = data_testNumerical

    # Clasificacion

    # xgboost with GridSearchCV to find the best params
#    scoring_type='f1_weighted'
#
#    start = time.time()
#    n_classes= len(list(labelEncoder.classes_))
#    xgb_model = xgb.XGBClassifier({'nthread':6, 'objective':'multi:softprob',
#                                   'num_class':n_classes, 'random_state':0})
#
#    tuned_parameters =  {'max_depth':[2,4,6,8,14,20],
#                         'n_estimators':[10,50,100,200,400]}
#
#    clf_xgb = GridSearchCV(xgb_model, tuned_parameters, cv=10, scoring=scoring_type)
#    clf_xgb.fit(data_train,target_train)
#    target_pred_xgb = clf_xgb.predict(data_test)
#    end = time.time()
#    print(clf_xgb.best_score_)
#    print(clf_xgb.best_params_)
#    print('clf_xgb time=', end - start)
#    print(classification_report(target_test, target_pred_xgb))
#    print(confusion_matrix(target_test, target_pred_xgb))
#    c_matrix = confusion_matrix(target_test, target_pred_xgb)
#    acc_xgb = (c_matrix[0,0]+c_matrix[1,1])*1.0/(sum(sum(c_matrix)))
#    print(acc_xgb)

    # Train the model with best params
    start = time.time()
    xgb_best_params = xgb.XGBClassifier(nthread=6, objective='multi:softprob',
                                        seed=0, n_estimators=200, max_depth=2)
    xgb_best_params.fit(data_train, target_train)

    target_pred_xgb = xgb_best_params.predict(data_test)
    end = time.time()

    print('clf_xgb time=', end - start)
    print(classification_report(target_test, target_pred_xgb))
    print(confusion_matrix(target_test, target_pred_xgb))
    c_matrix = confusion_matrix(target_test, target_pred_xgb)
    acc_xgb = (c_matrix[0,0]+c_matrix[1,1])*1.0/(sum(sum(c_matrix)))
    print(acc_xgb)

    # Save xgboost model with best params
    xgb_best_params = xgb.XGBClassifier(nthread=6, objective='multi:softprob',
                                        seed=0, n_estimators=200, max_depth=2)
    train_all = np.concatenate((data_train, data_test), axis=0)
    target_all = np.concatenate((target_train, target_test), axis=0)
    xgb_best_params.fit(train_all, target_all)
    file_out = '.\\output\\age_classifier_model.pkl'
    pickle.dump([labelEncoder, xgb_best_params], open(file_out, 'wb'))

    # No bagging in RF
#    start = time.time()
#    tuned_parameters = [{'max_features': [int(np.sqrt(data_train.shape[1])),min(int(3*np.sqrt(data_train.shape[1])),data_train.shape[1])],'n_estimators': [100,1000,5000]}]#,10000
#    clf_RF = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10, scoring=scoring_type)#'accuracy')#'precision')
#    clf_RF.fit(data_train,target_train)
#    target_pred_RF = clf_RF.predict(data_test)
#    end = time.time()
#    print(clf_RF.best_score_)
#    print(clf_RF.best_params_)
#    print('clf_RF time='+str(end - start))
#    print(classification_report(target_test, target_pred_RF))
#    print(confusion_matrix(target_test, target_pred_RF))
#    feature_importance(clf_RF,data_train.shape[1])#solo para RF

    # SVM
#    start = time.time()
#    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3,1e-2,1e-1,1e-2,1e-3, 1e-4],
#                     'C': [1,10,100,300,500,750,1300,2500], 'class_weight':['auto']},
#                    {'kernel': ['linear'], 'C': [1,10,100,300,500,750,1300,2500], 'class_weight':['auto']}]
#
#    clf_SVM = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring=scoring_type)#'accuracy')#'precision')
#    clf_SVM.fit(data_train,target_train)
#    target_pred_SVM = clf_SVM.predict(data_test)
#    end = time.time()
#    print(clf_SVM.best_score_)
#    print(clf_SVM.best_params_)
#    print('clf_SVM time='+str(end - start))
#    print(classification_report(target_test, target_pred_SVM))
#    print(confusion_matrix(target_test, target_pred_SVM))

    # Dummy calssifier
    clf_dummy = DummyClassifier(strategy='stratified', random_state=0)
    clf_dummy.fit(data_train, target_train)
    target_pred_dummy = clf_dummy.predict(data_test)
    print(classification_report(target_test, target_pred_dummy))
    print(confusion_matrix(target_test, target_pred_dummy))
    acc_dummy = clf_dummy.score(data_test, target_test)
    print(acc_dummy)

    # Decode predicitons
#    target_train_pred_xgb = labelEncoder.inverse_transform(clf_xgb.predict(data_train))
#    target_test_pred_xgb = labelEncoder.inverse_transform(target_pred_xgb)

    # Build output
#    dataOut['predictedAge'] = 0
#    dataOut['predictedAge'].loc[index_train] = target_train_pred_xgb
#    dataOut['predictedAge'].loc[index_test] = target_test_pred_xgb

#    return dataOut

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == '__main__':

    fileNameIn = '.\\output\\allUsers_fullFeaturesOUT.tsv'
    #fileNameOut = '.\\output\\allUsers_fullFeaturesOUT_classified.tsv'

    discretizeLexiconScore = True
    categorical_cols = ['userLexiconScore']
    numerical_cols = ['avg_isRT', 'avg_nUsers', 'avg_nLinks', 'avg_nHashtags', 'avg_nEmoticons',
                      'avg_nAllCap', 'avg_nEnlongatedW', 'avg_tweetLength', 'avg_wordLength',
                      'avg_nSingularP', 'ratioFollowersFriends', 'userListed_count', 'userFavourites_count']

    target_col = 'age'
    user_col = 'userScreen_name'
    load_cols = categorical_cols + numerical_cols + [target_col] + [user_col]

    dataIn = pd.read_csv(fileNameIn, sep='\t', usecols=load_cols)

    # Discretize LexiconScroe and preprocess
    if discretizeLexiconScore == True:
        dataIn['userLexiconScore'][dataIn['userLexiconScore'] >= target_bins[-1]] = target_bins[-1]-1
        dataIn['userLexiconScore'][dataIn['userLexiconScore'] <= target_bins[0]] = target_bins[0]+1
        dataIn['userLexiconScore'] = pd.cut(dataIn['userLexiconScore'].astype(int), bins=target_bins, right=False)

    # Run the classifier
    dataOut = main(dataIn, categorical_cols, numerical_cols, target_col, user_col)

#------------------------------------------------------------------------------