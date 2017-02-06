# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# SENTIMENT MODEL - APPLY MODEL
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
os.chdir('D:\\SentimentTM\\23_sentiment_analysis')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

import pandas as pd
import pickle

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # Load model
    model_file = '.\\model\\output\\sentiment_classifier_model.pkl'
    labelEncoder, xgb_best_params = pickle.load(open(model_file, 'rb'))
    
    # Load encoded text data
    data_file = '.\\output\\14_tweets_location_cleaned_featureEncoded.tsv'
    dataIn = pd.read_csv(data_file, sep='\t', encoding='utf-8')
    
    # Apply model
    dataIn_arr = dataIn.as_matrix()
    target_pred = xgb_best_params.predict(dataIn_arr)
    target_label = labelEncoder.inverse_transform(target_pred)
    
    target_pred_proba = xgb_best_params.predict_proba(dataIn_arr)
    
    # Combine results
    result = pd.DataFrame({'sentiment_pred':target_label,
                           'sentiment_pred_confidence':target_pred_proba.max(axis=1)})
    
    # Merge with original text
    tweets_file = '.\\input\\14_tweets_location_cleaned.tsv'
    tweets_df = pd.read_csv(tweets_file, sep='\t', encoding='utf-8')
    
    file_out = '.\\output\\14_tweets_location_with_sentiment.tsv'
    dataOut = pd.concat((tweets_df, result), axis=1)
    dataOut.to_csv(file_out, sep='\t', index=False, encoding='utf-8')

#------------------------------------------------------------------------------