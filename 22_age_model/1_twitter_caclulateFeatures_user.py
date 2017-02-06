# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# AGE CLASSIFYING MODEL - APPLYING - EXTRACTING FEATURES
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
import unicodedata
from bs4 import BeautifulSoup
import re
import nltk

from time import time

# Read lexiconns data
ageLex_file = '.\\model\\lexicons\\emnlp14age.csv'
ageLexicon = pd.read_csv(ageLex_file)
ageLexicon_idx = ageLexicon.set_index('term')
ageLexicon_dict = ageLexicon_idx.to_dict(orient='index')
ageLexicon_intercept = ageLexicon.weight[ageLexicon.term=='_intercept'][0]

#RT regex pattern
regexRT = ['(^RT @[a-zA-Z0-9]*\:?)', '(via @[a-z0-9]*)',
           '(retweeting @[a-z0-9]*)', '(retweet @[a-z0-9]*)']

#------------------------------------------------------------------------------
# Self-defined functions
#------------------------------------------------------------------------------

# Function to find the age score of a term (word) in the ageLexicon
# If cannot find that term (word) return a 0 value
def term_weight_wrapper(term):
    try:
        weight = ageLexicon_dict[term]['weight']
    except:
        weight = 0
    return weight

# Function to calcualte age score of a tweets using ageLexicon
# Check lexicons folder for more details
def caclulateLexiconScore(tokens):

    denominator = len(tokens)
    uniqueTokens = set(tokens)
    score = 0

    for t in uniqueTokens:
        weight = term_weight_wrapper(t)
        freq = tokens.count(t)
        score = score + freq*weight

    if denominator != 0:
        score = score/denominator + ageLexicon_intercept
    else:
        print('Denominator = 0. Division by 0. Let 0.')
        score = 0 + ageLexicon_intercept

    return score

# Function to add all text features
def add_featuresMsg(row):

    line = row['text_norm']
    row['isRT'] = isRT(line)
    row['nUsers'] = f_users(line)
    row['nLinks'] = f_links(line)
    row['nHashtags'] = f_hashtags(line)
    row['nEmoticons'] = f_emoticons(line)

    return row

# Function to preprocess and tokenize a tweet
def tokenize_line(text):

    # Ref: http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

    # Elminate RT
    if isRT(text) > 0:
        prog = re.compile(regexRT[0])
        rt = prog.findall(text)
        if len(rt) > 0:
            for l in rt:
                text = text.replace(l, '')

        for i in  range(1,len(regexRT)):
            pattern = regexRT[i]
            prog = re.compile(pattern, re.IGNORECASE)
            rt = prog.findall(text)
            if len(rt) > 0:
                for l in rt:
                    text = text.replace(l, '')

    # Remove links
    link_regex = '(https?:\/\/[^\s]*[\r\n]*)'
    link_re = re.compile(link_regex)
    links = link_re.findall(text)
    if len(links) > 0:
        for l in links:
            text = text.replace(l, '')

    # Eliminate user
    user_regex = '(@[A-Za-z\_]+[A-Za-z0-9_]+)'
    user_re = re.compile(user_regex)
    users = user_re.findall(text)
    if len(users) > 0:
        for l in users:
            text = text.replace(l, '')

    # Remove the punctuation using the character deletion step of translate
    ##no_punctuation = lowers.translate(None, string.punctuation)
    #no_punctuation = text.translate(None, string.punctuation)
    #tokens = nltk.word_tokenize(no_punctuation)

    text = text.strip()
    text = text.strip('.')

    # Tokenizer
    tweetTokenizer = nltk.tokenize.TweetTokenizer()
    tokens = tweetTokenizer.tokenize(text)

    #stems = stem_tokens(tokens, stemmer)
    #return stems
    stop = nltk.corpus.stopwords.words('english')
    filtered_tokens = [w for w in tokens if not w in stop]

    return filtered_tokens

# Function to check a tweet is RT or not
def isRT(msg):

    flagRT = 0

    prog = re.compile(regexRT[0])
    result = prog.findall(msg)
    if len(result) > 0:
        flagRT = 1

    for i in range(1, len(regexRT)):
        pattern = regexRT[i]
        prog = re.compile(pattern, re.IGNORECASE)
        result = prog.findall(msg)
        if len(result) > 0:
            flagRT = 1

    return flagRT

# Function to count number of users mentioned in a tweet
def f_users(line):
    if isRT(line) == 0:
        return line.count('@') # Count number of occurences of @/
    else:
        return (line.count('@')-1)

# Function to count number of links in a tweet
def f_links(line):
    return line.count('http:/') # Count number of occurences of http://

# Function to count number of hashtags in a tweet
def f_hashtags(line):
    return line.count('#') # Count number of occurences of #

# Function to count all-caps token (word)
# all-caps: the number of tokens with all characters in upper case
def f_allCaps(tokens):
    n = 0
    for t in tokens:
       if t.isupper() and t[0]!='#' and t[0]!='@' and len(t)>1:
           n = n+1

    return n

# Function to count long-tail tokens, e.g. sooooooooo gooddddd
def f_numberEnlongatedW(tokens):
    n = 0
    regex = r"([a-z]|[A-Z])\1{2,}"

    prog = re.compile(regex)
    for t in tokens:
        result = prog.findall(t)
        if len(result) > 0:
            n = n+1

    return n

# Function to count number of emoticons in a tweet, e.g. :) :D :(
def f_emoticons(line):

    emoticon_string = r"""
        \s(?:
          [<>]?
          [:;=8]                     # eyes
          [\-o\*\']?                 # optional nose
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          |
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          [\-o\*\']?                 # optional nose
          [:;=8]                     # eyes
          [<>]?
        )\s"""

    emoticon_re = re.compile(emoticon_string, re.VERBOSE | re.I | re.UNICODE)
    emoticons = emoticon_re.findall(line)
    #emoticons = emoticon_re.findall(line_noLink)

    return len(emoticons)

# Function to count the length of a tweet
def f_wordsLength(tokens):

    lenths = []
    for t in tokens:
        lenths.append(len(t))

    return lenths

# Function to count number of pronouns in a tweet, e.g. I, he, she...
def f_numberPronouns(pos_list):

    n_pronouns = 0
    for e in pos_list:
        pos = e[1]
        if pos == 'PRP' or pos == 'PRP$':
            n_pronouns = n_pronouns+1

    return n_pronouns

# Function to count number of singular pronouns in a tweet
singularP = ['I', 'me', 'my', 'mine', 'myself']

def f_nSingularP(tokens):

    n_pronouns = 0
    for t in tokens:
        if t in singularP:
            n_pronouns = n_pronouns+1

    return n_pronouns

# Function to clean the tweet
def normalizeTxt(msg):

    line = msg.strip() # Remove leading and trailing whitespace

    # Remove HTML entities
    soup = BeautifulSoup(line)
    line2 = soup.get_text()
    soup = BeautifulSoup(line2)
    line2 = soup.get_text()

    # Remove line break chars
    line2 = line2.replace('\r', ' ')
    line2 = line2.replace('\n', ' ')
    line2 = line2.replace('\r\n', ' ')

    return unicodedata.normalize('NFKD', line2).encode('ascii', 'ignore').strip() #.lower()

# Function to find a character "ch" in a string "s", return a list of positions
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

# Function to extract Twitter user_name from a file name
def extractUserName_fromFileName(fileNameIn):

    if len(find(fileNameIn, '\\')) > 0:
        indx1 = find(fileNameIn, '\\')[-1]+1
    else:
        indx1 = 0
    indx2 = find(fileNameIn, '_')[-1]
    userName = fileNameIn[indx1:indx2]

    return userName

# Function to extract Twitter user_id from a file name
def extractUserID_fromFileName(fullFileNameIn):
    file_name = os.path.basename(fullFileNameIn)
    return str(os.path.splitext(file_name)[0])

# Function to read a json text file into a data frame
def read_json_tweets(fileNameIn):
    dataIn = []
    with open(fileNameIn, 'r') as f:
        for line in f:
            dataIn.append(line)
    return pd.DataFrame(dataIn)

# Function to process a tweet file, combine all above functions
def parseTwitterFile(fileNameIn, maxTweet=-1):

    #dataIn = pd.read_csv(fileNameIn, header=None, sep=';;;;;')
    dataIn = read_json_tweets(fileNameIn)
    dataIn.columns = ['text']

    maxTweet = -1
    print('Full size:', dataIn.shape)
    if maxTweet != -1:
        dataIn = dataIn[:min(maxTweet, len(dataIn))]
    print('Reduced size:', dataIn.shape)

    #userScreen_name = extractUserName_fromFileName(fileNameIn)
    userScreen_name = extractUserID_fromFileName(fileNameIn)

    dataIn['text_norm'] = dataIn['text']
    dataIn['text_norm'].update(dataIn['text_norm'].apply(normalizeTxt))

    # Initialize
    dataIn['isRT'] = 0
    dataIn['nUsers'] = 0
    dataIn['nLinks'] = 0
    dataIn['nHashtags'] = 0
    dataIn['nEmoticons'] = 0
    dataIn['nAllCap'] = 0
    dataIn['nEnlongatedW'] = 0
    dataIn['tweetLength'] = 0
    dataIn['wordLength'] = 0
    #dataIn['nPronouns'] = 0 # Pos tagger
    dataIn['nSingularP'] = 0

    dataIn.update(dataIn.apply(add_featuresMsg, axis=1))

    tokens_series=dataIn['text_norm'].apply(tokenize_line)
    dataIn['nAllCap'].update(tokens_series.apply(f_allCaps))
    dataIn['nEnlongatedW'].update(tokens_series.apply(f_numberEnlongatedW))
    dataIn['tweetLength'].update(tokens_series.apply(len))
    dataIn['wordLength'].update(tokens_series.apply(f_wordsLength))
    #dataIn['nPronouns'].update(tokens_series.apply(nltk.pos_tag).apply(f_numberPronouns))
    dataIn['nSingularP'].update(tokens_series.apply(f_nSingularP))

    # Find features for this user
    userLexiconScore = caclulateLexiconScore(tokens_series.sum())
    nMsg = dataIn.shape[0]*1.0
    avg_isRT = dataIn['isRT'].sum()/nMsg
    avg_nUsers = dataIn['nUsers'].sum()/nMsg
    avg_nLinks = dataIn['nLinks'].sum()/nMsg
    avg_nHashtags = dataIn['nHashtags'].sum()/nMsg
    avg_nEmoticons = dataIn['nEmoticons'].sum()/nMsg
    avg_nAllCap = dataIn['nAllCap'].sum()/nMsg
    avg_nEnlongatedW = dataIn['nEnlongatedW'].sum()/nMsg
    avg_tweetLength = dataIn['tweetLength'].sum()//nMsg

    if len(dataIn['wordLength'].sum()) != 0:
        avg_wordLength = sum(dataIn['wordLength'].sum())/(len(dataIn['wordLength'].sum())*1.0)
    else:
        print('Division by 0. Let avg_wordLength = 0.')
        avg_wordLength = 0

    #avg_nPronouns=dataIn['nPronouns'].sum()//nMsg
    avg_nSingularP=dataIn['nSingularP'].sum()/nMsg

    feature_v = [userScreen_name, str(userLexiconScore), str(avg_isRT),
                 str(avg_nUsers), str(avg_nLinks), str(avg_nHashtags), str(avg_nEmoticons),
                 str(avg_nAllCap), str(avg_nEnlongatedW), str(avg_tweetLength), str(avg_wordLength),
                 str(avg_nSingularP)] #,str(avg_nPronouns)]

    return feature_v

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == '__main__':

    # Write header of output file with column names
    features_names = ['user_id_str', 'userLexiconScore', 'avg_isRT',
                      'avg_nUsers', 'avg_nLinks', 'avg_nHashtags', 'avg_nEmoticons',
                      'avg_nAllCap', 'avg_nEnlongatedW', 'avg_tweetLength', 'avg_wordLength',
                      'avg_nSingularP'] #,'avg_nPronouns']

    inputDir = '..\\16_enrich_tweets_data\\Output\\16_twitter_profile_tweets\\' # Input dir
    fileNameOut = '.\\output\\allUsers_tweets_featuresAge_30k.tsv'

    t0 = time()
    n_records = 0

    with open(fileNameOut, 'w') as fileOut:

        # Write the file headers
        row_string = "\t".join(features_names)
        fileOut.write(row_string)
        fileOut.write('\n')

        fileIn_list = os.listdir(inputDir)
        for fileNameIn in fileIn_list:
            if fileNameIn.endswith(".json"):

                #if n_records == 1: break

                n_records = n_records+1
                print(n_records)
                fullFileNameIn = inputDir + fileNameIn

                t1 = time()
                feature_v = parseTwitterFile(fullFileNameIn)
                print('Extracting time:', time()-t1)

                row_string = '\t'.join(feature_v)
                fileOut.write(row_string)
                fileOut.write('\n')

    print('Number of files: ', str(n_records))
    print('Total running time:', time()-t0)

#------------------------------------------------------------------------------