# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# AGE CLASSIFYING MODEL - APPLYING - JOIN FEATURES
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

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

# Read 2 input datasets
fileIn_userInfo = '.\\input\\14_tweets_location_cleaned.tsv'
fileIn_userFeatures = '.\\output\\allUsers_tweets_featuresAge_30k.tsv'

dataUsers = pd.read_csv(fileIn_userInfo, sep='\t', encoding='utf-8')
dataUsers.drop_duplicates('user_id_str', inplace=True)

dataFeatures = pd.read_csv(fileIn_userFeatures,sep='\t', encoding='utf-8')

# Join full data
dataFull = pd.merge(dataFeatures, dataUsers, how='left',
                    left_on=['user_id_str'], right_on=['user_id_str'],
                    left_index=True)

# Calculate some other features
dataFull['ratioFollowersFriends'] = dataFull['user_followers_count'] / dataFull['user_friends_count']
dataFull['ratioFollowersFriends'] = dataFull['ratioFollowersFriends'].replace(np.inf, 0)
dataFull['ratioFollowersFriends'] = dataFull['ratioFollowersFriends'].replace(np.nan, 0)

# A simple filter for organizational accounts
#followersThreshold = 5000
#dataFull = dataFull[dataFull['userFollowers_count'] < followersThreshold]

# Keep some features columns
keepCols = ['user_name', 'user_screen_name', 'user_id_str', 'user_description',
            'longitude', 'latitude', 'user_profile_image_url', 'user_listed_count',
            'user_favourites_count',
            
            'userLexiconScore', 'avg_isRT', 'avg_nUsers', 'avg_nLinks', 'avg_nHashtags',
            'avg_nEmoticons', 'avg_nAllCap', 'avg_nEnlongatedW', 'avg_tweetLength',
            'avg_wordLength', 'avg_nSingularP', 'ratioFollowersFriends']
                       
dataFull = dataFull[keepCols]
dataFull = dataFull.dropna(subset=['user_screen_name'])

# Rename all columns to fit with the pre-trained model
renameCols = ['userName', 'userScreen_name', 'userId', 'userDescription',
              'longitude', 'latitude', 'userProfile_image_url', 'userListed_count',
              'userFavourites_count',
              
              'userLexiconScore', 'avg_isRT', 'avg_nUsers', 'avg_nLinks', 'avg_nHashtags',
              'avg_nEmoticons', 'avg_nAllCap', 'avg_nEnlongatedW', 'avg_tweetLength',
              'avg_wordLength', 'avg_nSingularP', 'ratioFollowersFriends']
             
dataFull.columns = renameCols

# Save to file
fileOut = '.\\output\\allUsers_fullFeaturesAgeOUT_30k.tsv'
dataFull.to_csv(fileOut, sep='\t', index=False, encoding='utf-8')

#------------------------------------------------------------------------------