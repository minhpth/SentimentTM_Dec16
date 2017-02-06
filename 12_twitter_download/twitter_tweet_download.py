# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# TWITTER TWEETS DOWNLOAD
#------------------------------------------------------------------------------

# Functions:
# [1] Read the tourism locations downloaded by doing websrapping TripAdvisor
# [2] Request tweets from Twitter related to these locations
# [3] Save tweets to JSON file

# Version: 2.0
# Last edited: 20 Dec 2016
# Edited by: Minh PHAN

#------------------------------------------------------------------------------
# Global variables and settings
#------------------------------------------------------------------------------

# Seting working directory
import os
os.chdir('D:\\SentimentTM\\12_twitter_download')

# Folders
json_location_tweets_folder = '.\\output\\12_json_tweets_locations'

# Files
tourism_locations_file = '.\\input\\11_tripadvisor_tourism_locations.tsv'

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
import sys
from time import time

# Other functional packages
import json
import tweepy # Twitter package
from tweepy import OAuthHandler

#------------------------------------------------------------------------------
# Self-defined functions
#------------------------------------------------------------------------------

# Function to request tweets about a specific topic
def tweet_query(topic, n_tweet, file_out, mode='w', since_id=None, max_id=-1):
       
    # Choose the specific topic
    search_topic = topic
        
    search_file = file_out
    search_max = n_tweet # Set = -1 to collect unlimited tweets
    
    tweets_per_querry = 100
    search_count = 0
    search_list = []
    
    # Tweets will be querried by in order of created time
    # so, newest tweets will be downloaded first
    # since_id = None # Lower bound
    # max_id = -1 # Higher bound
        
    print('Request Twitter topic:', search_topic)
    
    with open(search_file, mode) as f:
        while search_count < search_max or search_max == -1:
            try:
                if max_id <= 0:
                    if not since_id:
                        new_tweets = api.search(q=search_topic, count=tweets_per_querry)
                    else:
                        new_tweets = api.search(q=search_topic, count=tweets_per_querry,
                                                since_id=since_id)
                else:
                    if not since_id:
                        new_tweets = api.search(q=search_topic, count=tweets_per_querry,
                                                max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=search_topic, count=tweets_per_querry,
                                                since_id=since_id, max_id=str(max_id - 1))
                                                
                if not new_tweets:
                    print('No more tweets to query')
                    break
                
                for tweet in new_tweets:
                    f.write(json.dumps(tweet._json) + '\n')
    
                search_list.append(new_tweets)                
                search_count += len(new_tweets)
                print(search_count, 'tweets downloaded')
                max_id = new_tweets[-1].id
            
            except tweepy.TweepError as e:
                print('ERROR:', e)
                break
    
    print()
    
    # Remove file if no tweet was downloaded
    if search_count == 0:
        os.remove(search_file)
       
#------------------------------------------------------------------------------
# MAIN: Query Twitter + Save to JSON files
#------------------------------------------------------------------------------

# Authorize Twitter API       
# Twitter account: minh_phan
# Twitter apps: tmSentiment
consumer_key = '6nlxgPwAvkWoKInRIuVH0JEL5'
consumer_secret = 'DdlP2azZasQfEkcrE5WLOcRL9tfeLmJdivK6OOqRRJKxVZ9tHp'
access_token = '560011434-lXTpAdAIz1YbAD0aulupgVnxoTPoos85W7EJbh0W'
access_secret = 'DAzyBXhZTkfEEY6TSRwV7H57X4KIuJVMRbfd9YO1qhHwU'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

if not(api):
    print('Cannot authenticate Twitter. System exit.')
    sys.exit(-1)

# Enable auto-waiting during the rate-limit time
api.wait_on_rate_limit = True
api.wait_on_rate_limit_notify = True

# Load all tourism locations from previous webscrapping step
df = pd.read_csv(tourism_locations_file, encoding='utf-8', sep='\t')

# Request tweets for all locations
t0 = time()
count = 0
for idx, row in df.iterrows():
    count += 1
    # if count == 100: break
    query =  row['city'] + ' ' + row['location_keyword'] # Create the query
    f_name = json_location_tweets_folder + '\\' + query.replace(' ', '_') + '.json' # Create file name
    tweet_query(query, n_tweet=-1, file_out=f_name) # Run the Twitter query

print('Running time:', time()-t0)

#------------------------------------------------------------------------------