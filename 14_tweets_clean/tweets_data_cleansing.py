# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# TWEETS DATA CLEANSING
#------------------------------------------------------------------------------

# Functions:
# [1] Remove duplicated line, duplicated tweet_id
# [2] Keep only tweets in English
# [3] Remove Retweeted
# [4] Clean tweets' text, remove hyperlinks, line breaks, etc.
# [5] Save to TSV file

# Version: 2.0
# Last edited: 20 Dec 2016
# Edited by: Minh PHAN

#------------------------------------------------------------------------------
# Global variables and settings
#------------------------------------------------------------------------------

# Seting working directory
import os
os.chdir('D:\\SentimentTM\\14_tweets_clean')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
import re
from time import time
from bs4 import BeautifulSoup # Remove HTML entities

#------------------------------------------------------------------------------
# Function to tokenize tweet
   
eyes = r"[8:=;]"
nose = r"['`\-]?"

emoticons_patterm = [
    r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), # smile
    r"{}{}p+".format(eyes, nose), # lolface
    r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), # sadface
    r"{}{}[\/|l*]".format(eyes, nose), # neutralface
    r"<3" # heart
]
        
regex_patterm = [    
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

# NOTE: always put emoticon_str at first !!!
regex_patterm = emoticons_patterm + regex_patterm

tokens_re = re.compile(r'('+'|'.join(regex_patterm)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'('+'|'.join(emoticons_patterm)+')', re.VERBOSE | re.IGNORECASE)

def tweet_tokenize(s):
    return tokens_re.findall(s)
    
# Processing tweets
# [1] Tokenize
# [2] Lowercase
# [3] Remove stopwords
from nltk.corpus import stopwords
stopwords_list = stopwords.words("english")

import string
puncs_list = list(string.punctuation)

def tweet_processing(s, lowercase=False, stopwords=False, puncuation=False):
    
    # Remove HTML entities
    s = BeautifulSoup(s, "lxml").get_text()
    
    # Tokenize
    tokens = tweet_tokenize(s)
    
    # Lowercase
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
                  
    # Remove stopwords
    if stopwords:
        tokens = [token for token in tokens if token not in stopwords_list]
    
    # Remove punctuation
    if puncuation:
        tokens = [token for token in tokens if token not in puncs_list]
    
    return tokens 

#------------------------------------------------------------------------------
# MAIN: Clean tweets data
#------------------------------------------------------------------------------

# Read the tweets data
file_in = '.\\input\\13_tweets_location_converted.tsv'
df = pd.read_csv(file_in, sep='\t', encoding='utf-8')

t0 = time()

# Some statistical numbers
df.shape
c1 = len(df)
c1 # 186708 rows

# Check and remove duplicated rows
df_row_dup = df.duplicated(keep='first')
sum(df_row_dup) # 0 duplicated rows
df = df.drop_duplicates()
c2 = len(df)
c2 # 186708 rows
c2 / c1 # 100%

# Check and remove duplicated tweet_id
df_id_dup = df['id_str'].duplicated(keep='first')
sum(df_id_dup) # 25495 duplicated ids
df = df[-df_id_dup]
c3 = len(df)
c3 # 246263 rows
c3 / c2 # 86%
c3 / c1 # 86%

# Select only English tweets
df = df[df['lang'].str[:2] == 'en']
c4 = len(df)
c4 # 152821 rows
c4 / c3 # 94%
c4 / c1 # 81%

# Remove retweeted tweets
df = df[df['retweeted'] == False] # This field is not helpful
df = df[-df.text.str.contains('^RT +@')]
c5 = len(df)
c5 # 69852 rows
c5 / c4 # 45%
c5 / c1 # 37%

# Refresh index
df = df.reset_index(drop=True)
             
# Remove hyperlinks, and other items
item_remove = ['http\S+', # hyperlinks
               '&gt;', '&gt;', # signs
               '&amp;', # ampersand
               '&quot;', # quote
               '[\n\r\t]' # line breaks and tabs
               ]

df['text_clean'] = df['text'].str.replace('|'.join(item_remove), ' ')

# Tokenize and continue clean text
text_clean = [' '.join(tweet_processing(text, True, True, True)) for text in df['text_clean']]
df['text_clean'] = text_clean

# Remove extra spaces and trim the text
df.text_clean = df.text_clean.str.replace(' +', ' ')
df.text_clean = df.text_clean.str.strip()

print('Running time:', time()-t0)

# Save to file
file_out = '.\\output\\14_tweets_location_cleaned.tsv'
df.to_csv(file_out, sep='\t', encoding='utf-8', index=False)

# -----------------------------------------------------------------------------