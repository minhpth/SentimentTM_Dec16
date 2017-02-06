# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# COMBINE ALL RESULTS INTO 1 FILES
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
os.chdir('D:\\SentimentTM\\31_dashboard')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

import pandas as pd
from datetime import datetime

from geopy.geocoders import Nominatim
geolocator = Nominatim()

from bs4 import BeautifulSoup # Remove HTML entities

#------------------------------------------------------------------------------
# Self-defined functions
#------------------------------------------------------------------------------

# Wrapper function to get geo location information, return a None type if failed
def get_geo(location_name, country=''):
    try:
        query = location_name.replace('London', ' ').strip()
        query = query + ', ' + 'London' + ', ' + country
        print('Get geo-location info:', query)
        
        return geolocator.geocode(query)
    except:
         return None

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

# Import original tweets
full_twwets_file = '.\\input\\14_tweets_location_cleaned.tsv'
full_tweets = pd.read_csv(full_twwets_file, sep='\t', encoding='utf-8')

# Import the accounts classified results
accounts_type_file = '.\\input\\17_twitter_account_type_classified.tsv'
accounts_type = pd.read_csv(accounts_type_file, sep='\t', encoding='utf-8')

# Import the age classified results
age_group_file = '.\\input\\allUsers_fullFeaturesAgeOUT_30k_classified.tsv'
age_group = pd.read_csv(age_group_file, sep='\t', encoding='utf-8')

# Import the gender classified results
gender_file = '.\\input\\allUsers_fullFeaturesGenderOUT_30k_classified.tsv'
gender = pd.read_csv(gender_file, sep='\t', encoding='utf-8')

# Import the sentiment classified results
sentiment_file = '.\\input\\14_tweets_location_with_sentiment.tsv'
sentiment = pd.read_csv(sentiment_file, sep='\t', encoding='utf-8')

#------------------------------------------------------------------------------
# Merge all

tweets_full = full_tweets.copy(deep=True)

# Merge tweets with account types (tweet level)
tweets_full = pd.merge(tweets_full, accounts_type[['id_str', 'account_type_pred', 'account_type_pred_confidence']],
                       how='left', left_index=True, on='id_str')

# Merge with age group predition (account level)
tweets_full = pd.merge(tweets_full, age_group[['userId', 'pred_age_group', 'pred_age_group_confidence']],
                       how='left', left_index=True, left_on='user_id_str', right_on='userId')
tweets_full.drop('userId', axis=1, inplace=True)

# Merge with gender predition (account level)
tweets_full = pd.merge(tweets_full, gender[['userId', 'pred_gender', 'pred_gender_confidence']],
                       how='left', left_index=True, left_on='user_id_str', right_on='userId')
tweets_full.drop('userId', axis=1, inplace=True)

# Merge with sentiment prediction (tweet level)
tweets_full = pd.merge(tweets_full, sentiment[['id_str', 'sentiment_pred', 'sentiment_pred_confidence']],
                       how='left', left_index=True, on='id_str')

#------------------------------------------------------------------------------
# Add some more information

# Remove HTML entities from tweets
tweets_full['text'] = tweets_full['text'].apply(lambda x: BeautifulSoup(x, "lxml").get_text())

# Parse datetime
created_at_str = tweets_full.created_at.str.replace(r'\+0000', '')
created_at = created_at_str.apply(lambda x: datetime.strptime(x, '%a %b %d %H:%M:%S  %Y'))
tweets_full['created_at_date'] = created_at.apply(lambda x: x.date())
tweets_full['created_at_time'] = created_at.apply(lambda x: x.time())

# Add geo-location
locations_list = tweets_full.keyword.unique()         
geo_list = [get_geo(loc, 'United Kingdom') for loc in locations_list]
     
geo_result = pd.DataFrame()
for idx in range(len(locations_list)):
    geo = dict()
    geo['keyword'] = locations_list[idx]
    
    if geo_list[idx] is not None:
        geo['geopy_address'] = geo_list[idx].address
        geo['geopy_latitude'] = geo_list[idx].latitude
        geo['geopy_longitude'] = geo_list[idx].longitude
        
    geo_result = geo_result.append(geo, ignore_index=True)
            
tweets_full = pd.merge(tweets_full, geo_result,
                       how='left', left_index=True, on='keyword')

# If the geo information is still null, take the original one
tweets_full.loc[tweets_full['geopy_latitude'].isnull(), 'geopy_latitude'] = tweets_full.loc[tweets_full['geopy_latitude'].isnull(), 'latitude']
tweets_full.loc[tweets_full['geopy_longitude'].isnull(), 'geopy_longitude'] = tweets_full.loc[tweets_full['geopy_longitude'].isnull(), 'longitude']

# Save the final results
file_out = '.\\output\\31_tweets_final_v5.tsv'
tweets_full.to_csv(file_out, sep='\t', encoding='utf-8', index=False)

#------------------------------------------------------------------------------