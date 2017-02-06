# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# SENTIMENT CLASSIFIER - APPLY MODEL - FEATURES EXTRACTION
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
os.chdir('D:\\SentimentTM\\23_sentiment_analysis')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import re
import copy
from time import time

# Tokenizer: https://github.com/myleott/ark-twokenize-py
# POS tagger: https://code.google.com/p/ark-tweet-nlp/downloads/detail?name=ark-tweet-nlp-0.3.2.tgz&can=2&q=
import CMUTweetTagger_windowsMod

#------------------------------------------------------------------------------
# Import different sentiment lexicons
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Twitter Word Clusters Lexicon
# Ref: http://www.cs.cmu.edu/~ark/TweetNLP/

# Import lexicon
cluster_file = '.\\model\\lexicons\\cmu_nlp_clustering\\50mpaths2.txt'
token_clusters = pd.read_csv(cluster_file, sep='\t', index_col=False,
                             dtype={"cluster":np.str, "token":np.str, "count":np.int})
clusters = np.array(token_clusters.cluster.unique(), dtype=np.str)
n_clusters = len(clusters)

# Convert lexicon to dict
token_clusters_dict = token_clusters.set_index('token').to_dict(orient='index')

clusters_df = pd.DataFrame({'cluster':clusters})
clusters_df['index'] = clusters_df.index.values
clusters_dict = clusters_df.set_index('cluster').to_dict(orient='index')

#------------------------------------------------------------------------------
# NRC Hashtag Sentiment Lexicon
# Ref: http://nparc.cisti-icist.nrc-cnrc.gc.ca/fra/voir/accept%C3%A9/?id=f3c48029-99e0-48c7-9aaf-271e9715465b
# Usage: https://github.com/balikasg/SemEval2016-Twitter_Sentiment_Evaluation

# Import lexicon
hashtag_sentiment_unigramFile = '.\\model\\lexicons\\HashtagSentimentAffLexNegLex\\HS-AFFLEX-NEGLEX-unigrams.txt'
HashtagSentimentAffLexNegLex = pd.read_csv(hashtag_sentiment_unigramFile, sep='\t',
                                           index_col=False, dtype={"token":np.str, "score":np.float, "n_pos":np.int, "n_neg":np.int})

# Convert lexicon to dict
HashtagSentimentAffLexNegLex_dict = HashtagSentimentAffLexNegLex.set_index('token').to_dict(orient='index')

#------------------------------------------------------------------------------
# Sentiment140 Context Lexicon
# Ref: http://saifmohammad.com/WebPages/lexicons.html

# Import lexicon
S140_LexiconsFile = '.\\model\\lexicons\\Sentiment140AffLexNegLex\\S140-AFFLEX-NEGLEX-unigrams.txt'
S140_Lexicons = pd.read_csv(S140_LexiconsFile, sep='\t', index_col=False,
                            dtype={"token":np.str, "score":np.float, "n_pos":np.int, "n_neg":np.int})

# Convert lexicon to dict
S140_Lexicons_dict = S140_Lexicons.set_index('token').to_dict(orient='index')

#------------------------------------------------------------------------------
# MPQA Lexicon
# Ref: http://mpqa.cs.pitt.edu/
# Ref: http://people.cs.pitt.edu/~wiebe/pubs/papers/emnlp05polarity.pdf

# Inport lexicon
mpqa_lexiconFile = '.\\model\\lexicons\\subjectivity_clues_hltemnlp05\\subjclueslen1-HLTEMNLP05.tff'
mpqa_lexicon = pd.read_csv(mpqa_lexiconFile, sep='\s+', index_col=False,
                           dtype={"strenght	":np.str, "lenght":np.str, "token":np.str, "POS":np.str, "stemmed1":np.str, "polarity":np.str})

# Preprocess this lexicon befoe using
def preprocess_mpqaLexicon(row):

    if 'weak' in row.strenght:
        row.strenght = 1
    else:
        row.strenght = 2

    row.token=row.token[6:]

    if 'negative' in row.polarity:
        row.strenght = row.strenght*(-1)

    return row

mpqa_lexicon.update(mpqa_lexicon.apply(preprocess_mpqaLexicon, axis=1))

# Convert lexicon to dict
mpqa_lexicon_dict = mpqa_lexicon.set_index('token').to_dict(orient='index')

#------------------------------------------------------------------------------
# NRC Word-Emotion Association Lexicon
# Warning: in this lexicon, each emoticon can have multiple score
# Ref: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

nrcEmotion_LexiconFile = '.\\model\\lexicons\\NRC-Emotion-Lexicon-v0.92\\NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'
nrcEmotion_Lexicon = pd.read_csv(nrcEmotion_LexiconFile, sep='\t', index_col=False,
                                 dtype={"token":np.str, "emotion":np.str, "acore":np.int})

# Convert lexicon to dict
nrcEmotion_Lexicon_dict = nrcEmotion_Lexicon.set_index(['token', 'emotion']).to_dict(orient='index')

#------------------------------------------------------------------------------
# Opinion Lexicon English
# Ref: http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
# Usage: https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107

ling_lexiconNegFile = '.\\model\\lexicons\\ling_lexicon\\negative-words.txt'
ling_lexiconPosFile = '.\\model\\lexicons\\ling_lexicon\\positive-words.txt'
ling_lexiconNeg = pd.read_csv(ling_lexiconNegFile, sep='\t', index_col=False, dtype={"token":np.str})
ling_lexiconPos = pd.read_csv(ling_lexiconPosFile, sep='\t', index_col=False, dtype={"token":np.str})

#------------------------------------------------------------------------------
# All features will be extracted

features_names = np.array(['POS_N', 'POS_V', 'POS_E', 'POS_Punct', 'all_caps', 'n_hashtags', 'n_neg',
                           'total_score', 'max_score', 'nonZero_score', 'last_score', 'sum_pos_aff_nrcEmo',
                           'sum_neg_aff_nrcEmo', 'sum_pos_neg_nrcEmo', 'sum_neg_neg_nrcEmo',
                           'sum_pos_aff_mpqa', 'sum_neg_aff_mpqa', 'sum_pos_neg_mpqa', 'sum_neg_neg_mpqa',
                           'sum_pos_aff_linLex', 'sum_neg_aff_linLex', 'sum_pos_neg_linLex',
                           'sum_neg_neg_linLex', 'lastTocket_exclamationFlag',
                           'number_contigousExcMarks', 'emoticons_pos', 'emoticons_neg', 'emoticons_last',
                           'number_EnlongatedW'])
features_names = np.hstack([features_names, clusters])

#------------------------------------------------------------------------------
# Self-defined functions to extract text features
#------------------------------------------------------------------------------

# Function to debug
def compare_2_list(list1, list2):
    for i in range(max(len(list1), len(list2))):
        try:
            if list1[i] != list2[i]:
                return False
        except:
            return False
    return True

# Function to separate positive and negative tokens
def filter_pos_neg_tokens(tokens_negEvaluated):

    token_pos = []
    token_neg = []

    for t in tokens_negEvaluated:
        if t[-4:] == '_NEG':
            token_neg.append(t[0:-4])
        else:
            token_pos.append(t)

    return token_pos, token_neg

# Function to calculate token score based on NRC Word-Emotion Association Lexicon
def f_nrcEmoLexicon(tokens_pos, tokens_neg):

    score_pos_aff = []
    score_neg_aff = []
    score_pos_neg = []
    score_neg_neg = []
    
    for t in tokens_pos:
        try:
            score_pos = nrcEmotion_Lexicon_dict[(t, 'positive')]['score']
            score_neg = nrcEmotion_Lexicon_dict[(t, 'negative')]['score']
            
            if score_pos > 0:
                score_pos_aff.append(score_pos)

            if score_neg > 0:
                score_neg_aff.append(score_neg)
        except:
            pass

    for t in tokens_neg:
        try:
            score_pos = nrcEmotion_Lexicon_dict[(t, 'positive')]['score']
            score_neg = nrcEmotion_Lexicon_dict[(t, 'negative')]['score']
            
            if score_pos > 0:
                score_pos_neg.append(score_pos)

            if score_neg > 0:
                score_neg_neg.append(score_neg)
        except:
            pass

    sum_pos_aff_nrcEmo = sum(score_pos_aff)
    sum_neg_aff_nrcEmo = sum(score_neg_aff)
    sum_pos_neg_nrcEmo = sum(score_pos_neg)
    sum_neg_neg_nrcEmo = sum(score_neg_neg)

    return sum_pos_aff_nrcEmo, sum_neg_aff_nrcEmo, sum_pos_neg_nrcEmo, sum_neg_neg_nrcEmo

# Function to calculate token score based on MPQA Lexicon
def f_mpqaLexicon(tokens_pos, tokens_neg):

    score_pos_aff = []
    score_neg_aff = []
    score_pos_neg = []
    score_neg_neg = []
    
    for t in tokens_pos:
        try:
            score = int(mpqa_lexicon_dict[t]['strenght'])
            
            if score > 0:
                score_pos_aff.append(score)

            if score < 0:
                score_neg_aff.append(score)
        except:
            pass
        
    for t in tokens_neg:
        try:
            score = int(mpqa_lexicon_dict[t]['strenght'])
            
            if score > 0:
                score_pos_neg.append(score)

            if score < 0:
                score_neg_neg.append(score)
        except:
            pass

    sum_pos_aff_mpqa = sum(score_pos_aff)
    sum_neg_aff_mpqa = sum(score_neg_aff)
    sum_pos_neg_mpqa = sum(score_pos_neg)
    sum_neg_neg_mpqa = sum(score_neg_neg)

    return sum_pos_aff_mpqa, sum_neg_aff_mpqa, sum_pos_neg_mpqa, sum_neg_neg_mpqa

# Function to calculate token score based on Opinion Lexicon English
def f_linLexicon(tokens_pos,tokens_neg):

    score_pos_aff = []
    score_neg_aff = []
    score_pos_neg = []
    score_neg_neg = []

    # If work not present in lexicon, disregard value, do not include 0
    for t in tokens_pos:
        if t in ling_lexiconPos.values:
            score_pos_aff.append(1)

        if  t in ling_lexiconNeg.values:
            score_neg_aff.append(-1)

    for t in tokens_neg:
        if t in ling_lexiconPos.values:
            score_pos_neg.append(1)

        if  t in ling_lexiconNeg.values:
            score_neg_neg.append(-1)

    sum_pos_aff_linLex = sum(score_pos_aff)
    sum_neg_aff_linLex = sum(score_neg_aff)
    sum_pos_neg_linLex = sum(score_pos_neg)
    sum_neg_neg_linLex = sum(score_neg_neg)

    return sum_pos_aff_linLex, sum_neg_aff_linLex, sum_pos_neg_linLex, sum_neg_neg_linLex

# Function to extract emoticons
def f_emoticons(line):

    emoticon_string = r"""
        (?:
          [<>]?
          [:;=8]                     # eyes
          [\-o\*\']?                 # optional nose
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          |
          [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
          [\-o\*\']?                 # optional nose
          [:;=8]                     # eyes
          [<>]?
        )"""

    emoticon_re = re.compile(emoticon_string, re.VERBOSE | re.I | re.UNICODE)
    emoticons = emoticon_re.findall(line)

    return emoticons

# Function to mark negative tokens
def negation_context(tokens, POS_types):

    negation_string = "(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't"
    endNegation_string = "^[.:;!?]$"

    negation_re = re.compile(negation_string, re.VERBOSE | re.I | re.UNICODE)
    endNegation_re = re.compile(endNegation_string, re.VERBOSE | re.I | re.UNICODE)

    negated_contexts = []
    i = 0
    context = []
    inNegationFlag = False

    for t in tokens:

        if inNegationFlag:
            context.append(i)

        if negation_re.match(t) is not None:
            context.append(i)
            inNegationFlag = True

        if endNegation_re.match(t) is not None and inNegationFlag == True:
            negated_contexts.append(context)
            context = []
            inNegationFlag = False

        i = i + 1

    if len(negated_contexts) == 0 and len(context) != 0:
        negated_contexts.append(context)

    tokens_neg = copy.deepcopy(tokens)

    for c in negated_contexts:
        for i in c:
            if POS_types[i] in ['N','V','A']: # noun, verb, adj
                tokens_neg[i] = tokens_neg[i] + '_NEG'

    return tokens_neg

# The number of words with one character repeated more than two times, e.g. soooo
def f_numberEnlongatedW(tokens):

    n = 0
    regex = '([a-z]|[A-Z]){2,}'
    prog = re.compile(regex)

    for t in tokens:
        result = prog.findall(t)
        if len(result) > 0:
            n = n + 1

    return n

# The number of contiguous sequences of exclamation marks, question marks, and
# both exclamation and question marks
def f_number_contigousExcMarks(line):

    regex = '[\?\!]{2,}'
    prog = re.compile(regex)
    result = prog.findall(line)

    return len(result)


# Whether the last token contains an exclamation or question mark
def f_lastToken_exclamation(token):
    if token[-1] == '!' or token[-1] == '?':
        return True
    else:
        return False

# Hashtags: the number of hashtags
def f_hashtags(line):
    return line.count('#')

# Feature: all-caps: the number of tokens with all characters in upper case
def f_allCaps(tokens):

    n = 0
    for t in tokens:
       if  t.isupper():
           n = n + 1

    return n

#------------------------------------------------------------------------------
# Find all features for a given line of text (status tweet)

def calculateFeatures(line, POS_line=None):

    # POS tagging and tokenization with CMU, all together
    # Can run POS tagger on whole tweets to save running time
    #t1 = time()
    if POS_line is None:
        POS_tagging = CMUTweetTagger_windowsMod.runtagger_parse([line])[0]
    else:
        POS_tagging = POS_line
        
    tokens = [element[0] for element in POS_tagging]
    POS_types = [element[1] for element in POS_tagging]
    POS_N =  ''.join(POS_types).count('N') # nouns
    POS_V = ''.join(POS_types).count('V') # verbs
    POS_E =  ''.join(POS_types).count('E') # emoticons
    POS_Punct = ''.join(POS_types).count(',') # punctuation
    #print('POS tagger:', time()-t1)

    # Number words with all caps char
    #t1 = time()
    all_caps = f_allCaps(tokens)
    #print('All caps words:', time()-t1)

    # Number of hashtags
    #t1 = time()
    n_hashtags = f_hashtags(line)
    #print('Hashtags:', time()-t1)

    # Negations
    #t1 = time()
    tokens_negEvaluated = negation_context(tokens, POS_types)
    n_neg = sum([1 for t in tokens_negEvaluated if t[-4:]=='_NEG'])
    #print('Negations:', time()-t1)

    # Hashtags and Sentiment140 lexicons
    #t1 = time()
    token_scores_autoLexicons = []
     
    for t in tokens_negEvaluated:
        # Sentiment140 Context Lexicon
        try:
            score1 = S140_Lexicons_dict[t]['score']
        except:
            score1 = 0
            
        # NRC Hashtag Sentiment Lexicon
        try:
            score2 = HashtagSentimentAffLexNegLex_dict[t]['score']
        except:
            score2 = 0

        scores = [score1, score2]

        if score1 == 0 and score2 == 0:
            # If not present in either lexicon, add 0
            token_scores_autoLexicons.append(0)
        else:
            # If present in 1 or 2, add the most extreme value
            total_score = scores[np.argmax(np.abs(scores))]
            token_scores_autoLexicons.append(total_score)

    total_score = np.sum(token_scores_autoLexicons)
    max_score = np.max(token_scores_autoLexicons)
    nonZero_score = sum(np.array(token_scores_autoLexicons) != 0)
    last_score = token_scores_autoLexicons[-1]
    #print('Hashtags and Sentiment140:', time()-t1)

    # Manual lexicons
    #t1 = time()
    tokens_pos, tokens_neg = filter_pos_neg_tokens(tokens_negEvaluated)
    sum_pos_aff_nrcEmo, sum_neg_aff_nrcEmo, sum_pos_neg_nrcEmo, sum_neg_neg_nrcEmo = f_nrcEmoLexicon(tokens_pos, tokens_neg)
    sum_pos_aff_mpqa, sum_neg_aff_mpqa, sum_pos_neg_mpqa, sum_neg_neg_mpqa = f_mpqaLexicon(tokens_pos, tokens_neg)
    sum_pos_aff_linLex, sum_neg_aff_linLex, sum_pos_neg_linLex, sum_neg_neg_linLex = f_linLexicon(tokens_pos, tokens_neg) 
    #print('Manual lexicons:', time()-t1)

    # Punctuation
    #t1 = time()
    lastTocket_exclamationFlag = f_lastToken_exclamation(tokens[-1])
    number_contigousExcMarks = f_number_contigousExcMarks(line)
    #print('Punctuation:', time()-t1)

    # Emoticons
    #t1 = time()
    emoticons_pos = 0
    emoticons_neg = 0
    emoticons_last = 0
    emoticons = f_emoticons(line)
    #print('Emoticons:', time()-t1)

    # Determine polarity based on lexicons
    #t1 = time()
    for emo in emoticons:

        exists = HashtagSentimentAffLexNegLex.score[HashtagSentimentAffLexNegLex.token==emo]

        if len(exists) > 0:

            score = exists.values[0]

            if score > 0:
                emoticons_pos = 1
            elif score < 0:
                emoticons_neg = 1

    if len(f_emoticons(tokens[-1])) > 0: # Last token is emoticon

        exists = HashtagSentimentAffLexNegLex.score[HashtagSentimentAffLexNegLex.token==tokens[-1]]

        if len(exists) > 0:

            score = exists.values[0]

            if score > 0:
                emoticons_last = 1
            elif score < 0:
                emoticons_last = -1
    #print('Emoticons:', time()-t1)

    # Elongated words
    #t1 = time()
    number_EnlongatedW = f_numberEnlongatedW(tokens)
    #print('Elongated words:', time()-t1)

    # Word clusters
    #t1 = time()
    cluster_encoding = np.zeros(n_clusters, dtype=np.int)

    for t in tokens:
        try:
            c_code = token_clusters_dict[t]['cluster']
            c_index = clusters_dict[c_code]['index']
            cluster_encoding[c_index] = 1
        except:
            pass
    #print('Word clusters:', time()-t1)

    # Put together all features in one vector
    features_tot = np.array([POS_N, POS_V, POS_E, POS_Punct, all_caps, n_hashtags, n_neg,
                             total_score, max_score, nonZero_score, last_score, sum_pos_aff_nrcEmo,
                             sum_neg_aff_nrcEmo, sum_pos_neg_nrcEmo, sum_neg_neg_nrcEmo,
                             sum_pos_aff_mpqa, sum_neg_aff_mpqa, sum_pos_neg_mpqa, sum_neg_neg_mpqa,
                             sum_pos_aff_linLex, sum_neg_aff_linLex, sum_pos_neg_linLex,
                             sum_neg_neg_linLex, lastTocket_exclamationFlag,
                             number_contigousExcMarks, emoticons_pos, emoticons_neg, emoticons_last,
                             number_EnlongatedW])
    features_tot = np.hstack([features_tot,cluster_encoding])

    return features_tot

# Function to extract features of all tweets, line by line
def parse_file(dataIn, col_msg):

    n_records = 0

    dataOut = np.zeros([dataIn.shape[0], len(features_names)], dtype=np.float)

    for line in dataIn[col_msg]:

        t0 = time()
        print(n_records)
        
        POS_line = POS_tagging_all[n_records]
        
        features_tot = calculateFeatures(line, POS_line)
        dataOut[n_records,:] = features_tot

        n_records = n_records + 1
        print('Running time:', time()-t0)

    print('Number of records:', str(n_records))

    return dataOut  

# Convert some target variable labels
def preProcess_polarityCol(p):

    # The original file contains incorrect polatity names,
    # can be neutral, objective or objective-OR-neutral

    if 'objective' in p:
        return 'neutral'
    else:
        return p

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # Extract text features from training dataset
    fileNameIn = '.\\inpurt\\14_tweets_location_cleaned.tsv'
    fileNameOut = '.\\output\\14_tweets_location_cleaned_featureEncoded.tsv'

    dataIn = pd.read_csv(fileNameIn, sep='\t', error_bad_lines=False, encoding='utf-8')

    col_msg = 'text'
    
    tweets_list = dataIn[col_msg].tolist()
    POS_tagging_all = CMUTweetTagger_windowsMod.runtagger_parse(tweets_list)
    
    dataOut = parse_file(dataIn, col_msg)

    includeGT = False # Work with real twets if False
    if includeGT == True: # Training data
        dataOut_pd = pd.DataFrame(dataOut, index=dataIn.index, columns=features_names)
        dataOut_pd['polarity'] = dataIn.polarity
        dataOut_pd['polarity'].update(dataOut_pd['polarity'].apply(preProcess_polarityCol))
        dataOut_pd.to_csv(fileNameOut, index=False)
    else: # Real tweet data
        dataOut_pd = pd.DataFrame(dataOut, index=dataIn.index, columns=features_names)
        dataOut_pd.to_csv(fileNameOut, sep='\t', index=False)
        #np.savetxt(fileNameOut, dataOut, delimiter=",")
        
#------------------------------------------------------------------------------