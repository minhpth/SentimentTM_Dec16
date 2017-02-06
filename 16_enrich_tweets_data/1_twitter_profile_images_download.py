# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# TWITTER IMAGES PROFILES DOWNLOAD
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
os.chdir('D:\\SentimentTM\\16_enrich_tweets_data')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
from time import time

# Other functional packages
from urllib import urlretrieve

#------------------------------------------------------------------------------
# MAIN: Download profile pictures from Twitter
#------------------------------------------------------------------------------

# Import user information
file_in = '.\\input\\14_tweets_location_cleaned.tsv'
userData = pd.read_csv(file_in, sep='\t', encoding='utf-8')

# Keep unique user_id
userData.drop_duplicates('user_id_str', keep='first', inplace=True)

data_folder = '.\\output\\16_twitter_profile_images' # Folder to save images
count = 0 # Count and print to track process
count_success = 0

# Loop through user_id and download profile pictures
t0 = time()
for index, row in userData.iterrows():
    
    count += 1
    image_url = row['user_profile_image_url'].replace('_normal', '')
    file_ext = os.path.splitext(image_url)[-1]
    file_name = data_folder + '\\' + str(row['user_id_str']) + file_ext
    
    if image_url != '':
        try:
            urlretrieve(image_url, file_name)
            print(row['user_id_str'], ': Success')
            count_success += 1
        except:
            print(row['user_id_str'], ': ERROR while downloading')
    else:
        print(row['user_id_str'], ': NO profile pictures')
        
print()        
print('Total images requested:', count)
print('Total successful requests:', count_success)
print('Running time:', time()-t0)

#------------------------------------------------------------------------------