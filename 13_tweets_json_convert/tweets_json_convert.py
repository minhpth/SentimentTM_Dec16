# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# JSON TWEETS FLATTENING
#------------------------------------------------------------------------------

# Functions:
# [1] Read and extract the tweets information store in JSON files
# [2] Convert to DataFrame and save to TSV file

# Version: 2.0
# Last edited: 20 Dec 2016
# Edited by: Minh PHAN

#------------------------------------------------------------------------------
# Global variables and settings
#------------------------------------------------------------------------------

# Seting working directory
import os
os.chdir('D:\\SentimentTM\\13_tweets_json_convert')

# Folders
json_location_tweets_folder = '.\\input\\12_json_tweets_locations'

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
from time import time

# Other functional packages
from bs4 import BeautifulSoup # Web scrapping
import json

#------------------------------------------------------------------------------
# Self-defined functions
#------------------------------------------------------------------------------

# Function to convert tweets in JSON format to data frame
def tweet_json2df(file_in):

    result = []
    error_list = []
    count = 0
    
    with open(file_in, 'r') as f:
        for line in f:
            try:                
                dt = json.loads(line)
                
                # Filling out error tweets from streaming
                # {'limit': {'timestamp_ms': '1468604173647', 'track': 1}}    
                if dt.get('limit') is not None: continue
                
                d = {} # Create an empty dict
                
                # Extract basic information from Tweet object    
                d['contributors'] = dt.get('contributors')
                d['created_at'] = dt.get('created_at')
                d['favorite_count'] = dt.get('favorite_count')
                d['favorited'] = dt.get('favorited')    
                d['id_str'] = dt.get('id_str')
                d['in_reply_to_screen_name'] = dt.get('in_reply_to_screen_name')
                d['id_in_reply_to_status_id_str'] = dt.get('in_reply_to_status_id')
                d['in_reply_to_user_id_str'] = dt.get('in_reply_to_user_id_str')
                d['is_quote_status'] = dt.get('is_quote_status')
                d['lang'] = dt.get('lang')
                d['retweet_count'] = dt.get('retweet_count')
                d['retweeted'] = dt.get('retweeted')
                d['source'] = BeautifulSoup(dt.get('source')).getText()
                d['text'] = dt.get('text')
                d['truncated'] = dt.get('truncated')
                    
                # Extract coordinates (if available)
                if dt.get('coordinates') is not None:
                    d['coordinates'] = True
                    d['longitude'] = dt.get('coordinates').get('coordinates')[0]
                    d['latitude'] = dt.get('coordinates').get('coordinates')[1]
                else:
                    d['coordinates'] = False
                    
                # Extract user information
                d['user_contributors_enabled'] = dt.get('user').get('contributors_enabled')
                d['user_created_at'] = dt.get('user').get('created_at')
                d['user_default_profile'] = dt.get('user').get('default_profile')
                d['user_default_profile_image'] = dt.get('user').get('default_profile_image')
                d['user_description'] = dt.get('user').get('description')
                d['user_favourites_count'] = dt.get('user').get('favourites_count')
                d['user_follow_request_sent'] = dt.get('user').get('follow_request_sent')
                d['user_followers_count'] = dt.get('user').get('followers_count')
                d['user_following'] = dt.get('user').get('following')
                d['user_friends_count'] = dt.get('user').get('friends_count')
                d['user_geo_enabled'] = dt.get('user').get('geo_enabled')
                d['user_has_extended_profile'] = dt.get('user').get('has_extended_profile')
                d['user_id_str'] = dt.get('user').get('id_str')
                d['user_is_translation_enabled'] = dt.get('user').get('is_translation_enabled')
                d['user_is_translator'] = dt.get('user').get('is_translator')
                d['user_lang'] = dt.get('user').get('lang')
                d['user_listed_count'] = dt.get('user').get('listed_count')
                d['user_location'] = dt.get('user').get('location')
                d['user_name'] = dt.get('user').get('name')
                d['user_notifications'] = dt.get('user').get('notifications')
                d['user_profile_background_color'] = dt.get('user').get('profile_background_color')
                d['user_profile_background_image_url'] = dt.get('user').get('profile_background_image_url')
                d['user_profile_background_tile'] = dt.get('user').get('profile_background_tile')
                d['user_profile_banner_url'] = dt.get('user').get('profile_banner_url')
                d['user_profile_image_url'] = dt.get('user').get('profile_image_url')
                d['user_profile_link_color'] = dt.get('user').get('profile_link_color')
                d['user_profile_sidebar_border_color'] = dt.get('user').get('profile_sidebar_border_color')
                d['user_profile_sidebar_fill_color'] = dt.get('user').get('profile_sidebar_fill_color')
                d['user_profile_text_color'] = dt.get('user').get('profile_text_color')
                d['user_profile_use_background_image'] = dt.get('user').get('profile_use_background_image')
                d['user_protected'] = dt.get('user').get('protected')
                d['user_screen_name'] = dt.get('user').get('screen_name')
                d['user_statuses_count'] = dt.get('user').get('statuses_count')
                d['user_time_zone'] = dt.get('user').get('time_zone')
                d['user_url'] = dt.get('user').get('url')
                d['user_utc_offset'] = dt.get('user').get('utc_offset')
                d['user_verified'] = dt.get('user').get('verified')
                    
                # Extract place (if available)
                # Extract entities (if available)
                    
                result.append(d)
            
                count += 1
                if count % 10000 == 0:
                    print(count, 'tweets converted')
            
            except:
                print('ERROR at line:', count)
                print(line)
                error = [file_in, count, line]
                error_list.append(error)
                continue
                # return line
    
    df = pd.DataFrame(result)
    
    # Clean the line break in tweets for saving TSV file
    df['text'] = df['text'].str.replace(r'[\n\r\t]', ' ')
    df['user_location'] = df['user_location'].str.replace(r'[\n\r\t]', ' ')
    df['user_description'] = df['user_description'].str.replace(r'[\n\r\t]', ' ')
    
    # Creat a field to store file name
    df['keyword'] = os.path.split(file_in)[-1].replace('.json', '').replace('_', ' ')
    
    print(count, 'tweets converted,', len(error_list), 'errors')
    
    return df, error_list
        
#------------------------------------------------------------------------------
# MAIN: JSON files extract + flattern to DataFrame
#------------------------------------------------------------------------------
        
# Create a list of all JSON file in the data folder  
file_list = [fn for fn in os.listdir(json_location_tweets_folder) if '.json' in fn]
print(len(file_list), 'JSON files')
for f in file_list: print(f)
print()

# Flattern JSON file into DataFrame
t0 = time()
tweet_df = pd.DataFrame()
error_list = []
for f in file_list:
    print('JSON file:', f)
    df, error = tweet_json2df(json_location_tweets_folder + '/' + f)
    tweet_df = tweet_df.append(df)
    error_list = error_list + error # Collect errors to fix
    print()
print('Running time:', time()-t0)

# Size: (186708, 56)
    
# Save to TSV file
file_out = '.\\output\\13_tweets_location_converted.tsv'
tweet_df.to_csv(file_out, sep='\t', index=False, encoding='utf-8')

#------------------------------------------------------------------------------