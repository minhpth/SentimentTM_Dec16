# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# TWITTER ACCOUNT TYPE CLASSIFIER - BUILD CLASSIFIER MODEL
#------------------------------------------------------------------------------

# Functions:
# [1] ...

# Version: 2.0
# Last edited: 20 Dec 2016
# Edited by: Minh PHAN

#------------------------------------------------------------------------------
# Global variables and settings
#------------------------------------------------------------------------------

# Seting working directory
import os
os.chdir('D:\\SentimentTM\\17_account_type_filter\\model')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

# Other functional packages
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc, roc_curve, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder

import pickle

#------------------------------------------------------------------------------
# Functions to import data
#------------------------------------------------------------------------------

# Function to create a list of image features names
def findFeaturesFeatureNames():
    f_names = ['contrast','avg_h','avg_s', 'avg_v', 'pleasure', 'arousal', 'dominance']

    f_names = f_names + ['hue_hist_' + str(i) for i in range(12)]
    f_names = f_names + ['saturation_hist_' + str(i) for i in range(5)]
    f_names = f_names + ['brightness_hist_' + str(i) for i in range(3)]
    f_names = f_names + ['std_hueHist', 'std_satHist', 'std_brHist']
    f_names = f_names + ['haralicks_f_' + str(i) for i in range(13)]
    return f_names
    
# Function to create a list of text vector features names
def word2vec_features_names():
    f_names = []
    for i in range(300):
        f_names = f_names + ['w2v_' + str(i)]
    return f_names

# Plot ROC curve with thresholds    
def  plot_ROC(y_score,target_test):
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds = roc_curve(target_test, y_score)
    roc_auc = auc(fpr, tpr) # false_positive_rate, true_positive_rate

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr,
             label='ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    
    ps=np.linspace(0, 1, num=10)
    ind_points = [np.argmin(np.abs(fpr-p)) for p in ps]
    plt.plot(fpr[ind_points],tpr[ind_points], 'o')    
    for i in ind_points:
        plt.annotate(np.around(thresholds[i],3), (fpr[i],tpr[i]))
    
    return plt
    
#------------------------------------------------------------------------------
# MAIN: xgboost + GridSearch
#------------------------------------------------------------------------------

# Import data with features
file_in = '.\\output\\17_user_profile_type_full_features.tsv'
userDataWithF = pd.read_csv(file_in, sep='\t', encoding='utf-8')

# Training features names
featureCols = findFeaturesFeatureNames() + word2vec_features_names()

# Target variables
y = (userDataWithF.account_type == 2).values.astype(int) # 1 = idv / 2 = org
le = LabelEncoder().fit(pd.Series(['idv', 'org']))

scale_pos_weight = np.sum(y==1) / np.sum(y==0) # Pos / Neg

# Train the model with xgboost + GridSearchCV
scoring_type = 'roc_auc'
xgb_model = xgb.XGBClassifier({'objective':'binary:logistic',
                               'seed':123,
                               'silent':0,
                               'scale_pos_weight':scale_pos_weight})
tuned_parameters = {'max_depth':[2,4,6,8,14,20],
                    'n_estimators':[10,50,100,200,400,500]}
xgb_tab = GridSearchCV(xgb_model, tuned_parameters, cv=5,
                       scoring=scoring_type, verbose=10)
xgb_tab.fit(userDataWithF[featureCols], y)

# Best parameters
print(xgb_tab.best_score_) # 0.906917743254
print(xgb_tab.best_params_) # n_estimators: 500 / max_depth: 6

#------------------------------------------------------------------------------
# MAIN: xgboost + best parameters + evaluation
#------------------------------------------------------------------------------

# Split data sets, prepare to train model
X_train, X_test, y_train, y_test = train_test_split(userDataWithF[featureCols], y, test_size=0.3,
                                                    random_state=123, stratify=y)

# Save xgboost model and selected features
clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
                            seed=123, silent=0,
                            max_depth=6, n_estimators=500,
                            scale_pos_weight=scale_pos_weight)
clf_xgb.fit(X_train, y_train)

# Evaluation and plot ROC curve
target_pred_xgb_proba = clf_xgb.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, target_pred_xgb_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.title('ROC curve (area = {0:0.2f})'.format(roc_auc))
plt.plot(fpr, tpr, linewidth=2)

target_pred_xgb = clf_xgb.predict(X_test)

print('Classification report:')
print(classification_report(y_test, target_pred_xgb))

print('Confusion matrix:')
print(confusion_matrix(y_test, target_pred_xgb))

print('Accuracy:')
print(accuracy_score(y_test, np.round(target_pred_xgb)))

# Compare with a dummy classifier
dm_clf = DummyClassifier(strategy='stratified', random_state=123).fit(X_train, y_train)
y_pred_dm = dm_clf.predict(X_test)
print('Accuracy baseline:')
print(accuracy_score(y_test, y_pred_dm))

# Plot ROC with thresholds
plot_ROC(target_pred_xgb_proba, y_test)

#------------------------------------------------------------------------------
# MAIN: Train xgboost model on all data + Save model
#------------------------------------------------------------------------------

clf_xgb.fit(userDataWithF[featureCols], y) # Train model on all data
file_out = '.\\output\\17_twitter_account_type_classifier_model.pkl'
pickle.dump([le, clf_xgb], open(file_out, 'wb'))

#------------------------------------------------------------------------------