# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# AGE CLASSIFYING MODEL - BUILD MODEL - JOIN FEATURES
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

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

# Read 2 input datasets
fileIn_userAges = '.\\input\\user_age.tsv'
fileIn_userFeatures = '.\\output\\allUsers_tweets_features.tsv'

dataAges = pd.read_csv(fileIn_userAges, sep='\t', encoding='utf-8')
dataFeatures = pd.read_csv(fileIn_userFeatures,sep='\t', encoding='utf-8')

# Left join
dataFull = pd.merge(dataFeatures, dataAges, how='left',
                    left_on=['userScreen_name'], right_on=['userScreen_name'])

# New feature: ratio followers to friends as a measure of a user tendency to produce vs consume info
dataFull['ratioFollowersFriends'] = dataFull['userFollowers_count'] / dataFull['userFriends_count']
dataFull['ratioFollowersFriends'] = dataFull['ratioFollowersFriends'].replace(np.inf, 0)
dataFull['ratioFollowersFriends'] = dataFull['ratioFollowersFriends'].replace(np.nan, 0)

# A simple filter for organizational accounts
#followersThreshold = 5000
#dataFull = dataFull[dataFull['userFollowers_count'] < followersThreshold]

keepCols=['userName', 'userScreen_name', 'userId', 'userDescription',
          'longitude', 'latitude','userProfile_image_url','userLexiconScore', 'avg_isRT',
          'avg_nUsers', 'avg_nLinks', 'avg_nHashtags', 'avg_nEmoticons',
          'avg_nAllCap','avg_nEnlongatedW','avg_tweetLength', 'avg_wordLength',
          'avg_nSingularP', 'ratioFollowersFriends', 'userListed_count', 'userFavourites_count',
          'age']
                       
dataFull = dataFull[keepCols]
dataFull = dataFull.dropna(subset=['userScreen_name'])

# Save to file
fileOut = '.\\output\\allUsers_fullFeaturesOUT.tsv'
dataFull.to_csv(fileOut, sep='\t', index=False, encoding='utf-8')

#------------------------------------------------------------------------------