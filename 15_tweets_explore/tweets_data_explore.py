# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# TWITTER TWEETS DATA EXPLORATION
#------------------------------------------------------------------------------

# Functions:
# [1] Explore tweets data to have an overview
# [2] Check NA, unique userID, duplicated...

# Version: 2.0
# Last edited: 20 Dec 2016
# Edited by: Minh PHAN

#------------------------------------------------------------------------------
# Global variables and settings
#------------------------------------------------------------------------------

# Seting working directory
import os
os.chdir('D:\\SentimentTM\\15_tweets_explore')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from collections import Counter

#------------------------------------------------------------------------------
# Self-defined functions
#------------------------------------------------------------------------------

# Function to print out structure of data (mimic str() in R)
def strdf(df):
    print(type(df), ':\t', df.shape[0], 'obs. of', df.shape[1], 'variables:')
    if df.shape[0] > 4:
        df = df.head(4) # Take first 4 obs.
        dots = '...' # Print ... for the rest values  
    else:
        dots = ''
    space = len(max(list(df), key=len))
    for c in list(df):
        print(' $', '{:{align}{width}}'.format(c, align='<', width=space),
              ':', df[c].dtype, str(df[c].values)[1:-1], dots)

# Function to print out NAN values and their data types
def nadf(df):
    print(type(df), ':\t', df.shape[0], 'obs. of', df.shape[1], 'variables:')
    df_type = df.dtypes    
    df_NA = df.isnull().sum()
    space = len(max(list(df), key=len))
    space_type = len(max([d.name for d in df.dtypes.values], key=len))
    space_NA = len(str(max(df_NA)))
    for c in list(df):
        print(' $', '{:{align}{width}}'.format(c, align='<', width=space),
              ':', '{:{align}{width}}'.format(df_type[c], align='<', width=space_type),
              '{:{align}{width}}'.format(df_NA[c], align='>', width=space_NA),
              'NAN value(s)')

# Function to print out unique variables and its percentages in data
def proportion_print(pd_series):
    df = pd.DataFrame({pd_series.name:pd_series,
                       'count':1,
                       'percent':1/len(pd_series)})
    print(df.groupby(pd_series.name).sum())
        
#------------------------------------------------------------------------------
# MAIN: Twitter data analysis
#------------------------------------------------------------------------------
 
# Import tweets in TSV file
file_in = '.\\input\\13_tweets_location_converted.tsv'
tweet_df = pd.read_csv(file_in, sep='\t', encoding='utf-8')

# Data overview
strdf(tweet_df)

# Size: (186708, 56)

# Check NA
tweet_df.isnull().sum()

# How about user description?
sum(tweet_df.user_description.isnull()) # Some in blank
sum(tweet_df.user_description.isnull()) / len(tweet_df) # 12% blank

# How about user profile change?
sum(tweet_df.user_default_profile_image == True) # Some didn't change
sum(tweet_df.user_default_profile_image == True) / len(tweet_df) # 1.9%
tweet_df[tweet_df.user_default_profile_image == True].user_screen_name

# How many unique ID?
len(tweet_df.id_str.unique())
len(tweet_df.id_str.unique()) / len(tweet_df) # 86%

# Language of profile and tweets?
acc_lang_count = Counter(tweet_df.user_lang.str[:2])
labels, values = zip(*acc_lang_count.most_common())
pd.Series(values, labels, name='Account language').plot.pie(figsize=(6, 6), autopct='%.2f')
plt.show()

tweet_lang_count = Counter(tweet_df.lang.str[:2])
labels, values = zip(*tweet_lang_count.most_common())
pd.Series(values, labels, name='Tweet language').plot.pie(figsize=(6, 6), autopct='%.2f')
plt.show()

# How many retweeted?
sum(tweet_df.retweeted == True)
sum(tweet_df.text.str.contains('^RT @'))
sum(tweet_df.text.str.contains('^RT @')) / len(tweet_df) # 53%
tweet_df[tweet_df.text.str.contains('^RT @')].text

#------------------------------------------------------------------------------