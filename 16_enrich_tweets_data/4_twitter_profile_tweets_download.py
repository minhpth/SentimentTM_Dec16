# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# TWITTER PROFILES TWEETS DOWNLOAD
#------------------------------------------------------------------------------

# Functions:
# [1] ...

# Version: 2.0
# Last edited: 7 Nov 2016
# Edited by: Minh PHAN

#------------------------------------------------------------------------------
# Global variables and settings
#------------------------------------------------------------------------------

# Seting working directory
import os
os.chdir('D:\\SentimentTM\\16_enrich_tweets_data')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
from time import time
import re
import sys

# Other functional packages
import tweepy # Twitter package
from tweepy import OAuthHandler

#------------------------------------------------------------------------------
# Self-defined functions
#------------------------------------------------------------------------------

user_query = 0
user_fail = 0
fail_list = []

# Function to request tweets about a specific topic
def profile_tweet_query(user_id_str, n_tweet, file_out, mode='w', since_id=None, max_id=-1):
       
    global user_query, user_fail, fail_list
    
    user_query += 1
    
    # Choose the specific topic
    user_id = int(user_id_str)
        
    search_file = file_out
    search_max = n_tweet # Set = -1 to collect unlimited tweets
    
    tweets_per_querry = 100
    search_count = 0
    search_list = []
    
    # Tweets will be querried by in order of created time
    # so, newest tweets will be downloaded first
    # since_id = None # Lower bound
    # max_id = -1 # Higher bound
        
    print('UserID:', user_id, '[', user_query, ']')
    
    with open(search_file, mode) as f:
        while search_count < search_max or search_max == -1:
            try:
                if max_id <= 0:
                    if not since_id:
                        new_tweets = api.user_timeline(user_id, count=tweets_per_querry)
                    else:
                        new_tweets = api.user_timeline(user_id, count=tweets_per_querry,
                                                       since_id=since_id)
                else:
                    if not since_id:
                        new_tweets = api.user_timeline(user_id, count=tweets_per_querry,
                                                       max_id=str(max_id - 1))
                    else:
                        new_tweets = api.user_timeline(user_id, count=tweets_per_querry,
                                                       since_id=since_id, max_id=str(max_id - 1))
                                                
                if not new_tweets:
                    print('No more tweets to query')
                    break
                
                # Write tweets to JSON file
                for tweet in new_tweets:
                    #f.write(json.dumps(tweet._json) + '\n')
                    text = tweet.text.encode('utf-8')
                    text = re.sub(r'[\n\r\t]', ' ', text)
                    f.write(text + '\n')
    
                search_list.append(new_tweets)                
                search_count += len(new_tweets)
                print(search_count, 'tweets downloaded')
                max_id = new_tweets[-1].id
            
            except tweepy.TweepError as e:
                print('ERROR:', e)
                user_fail += 1
                fail_list.append(user_id)
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

# Import user information
file_in = '.\\input\\14_tweets_location_cleaned.tsv'
userData = pd.read_csv(file_in, sep='\t', encoding='utf-8')

# Keep unique user_id
userData.drop_duplicates('user_id_str', keep='first', inplace=True)

data_folder = '.\\output\\16_twitter_profile_tweets' # Folder to save profile tweets
count = 0 # Count and print to track process
count_success = 0

# Loop through user_id_str and download profile tweets
t0 = time()
for index, row in userData.iterrows():
    
    count += 1
    file_out = data_folder + '\\' + str(row['user_id_str']) + '.json'
    profile_tweet_query(row['user_id_str'], 100, file_out)
        
print()        
print('Total user_id requested:', count)
print('Fail rate:', user_fail/user_query) # 0.5%
print('Running time:', time()-t0) # 42989.875

#------------------------------------------------------------------------------
# Re-try with some fail user_id request
# Some user_id queries failed because the accounts are protected

count = 0
retry_list = fail_list[:] # Just make an deep copy

user_query = 0
user_fail = 0
fail_list = []

# Loop through fail_list and download profile tweets
t0 = time()
for user_id in retry_list:
    
    count += 1
    file_out = data_folder + '\\' + str(user_id) + '.json'
    profile_tweet_query(user_id, 100, file_out)
        
print()        
print('Total user_id retried:', count)
print('Fail rate:', user_fail/user_query) # 90%
print('Running time:', time()-t0) # 221.718

#------------------------------------------------------------------------------