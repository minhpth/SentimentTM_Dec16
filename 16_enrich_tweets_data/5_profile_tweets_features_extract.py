# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# ACCOUNT TYPE CLASSIFIER - TEXT FEATURE EXTRACT
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
import numpy as np
import re
from time import time

# Other functional packages
import string
from nltk.corpus import stopwords
from nltk.util import bigrams
from bs4 import BeautifulSoup # Remove HTML entities

# Load Google's pre-trained Word2Vec model
from gensim.models import Word2Vec
n_dimension = 300 # Word2Vec dimension

google_word2vec_file = '..\\Models\\google_word2vec\\GoogleNews-vectors-negative300.bin'
w2v_model = Word2Vec.load_word2vec_format(google_word2vec_file, binary=True)

#------------------------------------------------------------------------------
# Functions to pre-process and clean text
#------------------------------------------------------------------------------

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
 
def preprocess(s, lowercase=False):
    tokens = tweet_tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
 
# Set of patterns to remove
punc = list(string.punctuation) # Punctuation list
stop = stopwords.words('english') # Stopwords list
mention_pattern = r'(?:@[\w_]+)'
number_pattern = r'(?:(?:\d+,?)+(?:\.?\d+)?)'
link_pattern = r'http\S+'
html_entities_pattern = r'&[^ ]+;'
remove_terms = ['via', 'rt'] # Some other terms to remove
            
def sentence_preprocessing(sentence):
    
    sentence_clean = []
    
    # Remove HTML entities
    sentence = BeautifulSoup(sentence, "lxml").get_text()
     
    # Tokenize
    tokens = preprocess(sentence, lowercase=False)
    
    # Step 1: Remove all useless things
    tokens = [tk for tk in tokens if tk not in punc] # Punctuation
    #tokens = [tk for tk in tokens if tk not in stop] # Stopwords
    tokens = [tk for tk in tokens if re.match(link_pattern, tk) == None] # Link
    tokens = [tk for tk in tokens if re.match(html_entities_pattern, tk) == None] # HTML entities
    tokens = [tk for tk in tokens if tk.lower() not in remove_terms] # Some special terms to remove
    
    # Step 2: Add some lower/upper case
    tokens_lower = [tk.lower() for tk in tokens] # Lowercase
    tokens_upper = [tk.upper() for tk in tokens] # Uppercase
    tokens_title = [tk.title() for tk in tokens] # Title
    tokens = tokens + tokens_lower + tokens_upper + tokens_title
    
    # Step 3: Add bigram
    bigrams_words = ['_'.join(w) for w in list(bigrams(tokens))]
    sentence_clean = ' '.join(tokens) + ' ' + ' '.join(bigrams_words)
    
    return sentence_clean

try_count = 0
fail_count = 0
fail_list = []
    
def word2vec_wrapper(word):
    global try_count, fail_count
    
    try:
        try_count += 1
        vec = w2v_model[word]        
    except:
        fail_count += 1
        fail_list.append(word)
        vec = np.array([0]*n_dimension, dtype='float32') # Return a blank vector
    return vec
    
def word2vec_sentence(sentence):
    sentence_clean = sentence_preprocessing(sentence)
    sent_vec = np.mean([word2vec_wrapper(w) for w in sentence_clean.split()], axis=0)
    return sent_vec

# Function to create a list of image features names
def findFeaturesFeatureNames():
    f_names = ['contrast','avg_h','avg_s', 'avg_v', 'pleasure', 'arousal', 'dominance']

    f_names = f_names + ['hue_hist_' + str(i) for i in range(12)]
    f_names = f_names + ['saturation_hist_' + str(i) for i in range(5)]
    f_names = f_names + ['brightness_hist_' + str(i) for i in range(3)]
    f_names = f_names + ['std_hueHist', 'std_satHist', 'std_brHist']
    f_names = f_names + ['haralicks_f_' + str(i) for i in range(13)]
    return f_names
    
# Function to create a list of text vector features names
def word2vec_features_names():
    f_names = []
    for i in range(n_dimension):
        f_names = f_names + ['w2v_' + str(i)]
    return f_names

#------------------------------------------------------------------------------
# MAIN: Extract text features
#------------------------------------------------------------------------------

# Import and clean text data
file_in = '.\\output\\16_tweets_location_with_image_features.tsv'
userDataWithF = pd.read_csv(file_in, sep='\t', encoding='utf-8')
userDataWithF.user_description.fillna('', inplace=True)
userDataWithF.text.fillna('', inplace=True)
userDataWithF.user_location.fillna('', inplace=True)
userDataWithF['text_combined'] = userDataWithF.user_description + ' ' + \
                                 userDataWithF.text + ' ' + \
                                 userDataWithF.user_location

# Extract text features: word2vec
t0 = time()
X = pd.DataFrame([word2vec_sentence(s) for s in userDataWithF['text_combined']],
                  columns=word2vec_features_names())
print('Word2Vec fail rate:', fail_count/try_count)
print('Running time:', time()-t0)

# Add new word2vec features to the data
userDataWithF = pd.concat((userDataWithF, X), axis=1)

# Save to file
file_out = '.\\output\\16_tweets_location_full_features.tsv'
userDataWithF.to_csv(file_out, sep='\t', encoding='utf-8', index=False)

#------------------------------------------------------------------------------