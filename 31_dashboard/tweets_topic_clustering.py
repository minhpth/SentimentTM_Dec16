# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# TWEETS TOPIC CLUSTERING ON FINAL RESULT
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

# Essential packages
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.style.use('ggplot')
from time import time
from collections import Counter

# Other functional packages
import cld # Language classifier
from pprint import pprint # Print in a nicer way
from wordcloud import WordCloud # Show wordcloud

from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF
from sklearn.cluster import KMeans, MiniBatchKMeans # Topics clustering
from gensim.models import Word2Vec

import seaborn as sns
from glob import glob

import string
from nltk.corpus import stopwords
from nltk.util import bigrams
from bs4 import BeautifulSoup # Remove HTML entities

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import scale

from numpy.random import random_sample
from math import sqrt, log

from sklearn.decomposition import PCA

#------------------------------------------------------------------------------
# Self-defined functions
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Functions to pre-process and clean text

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
remove_terms = ['via', 'rt',
                'london', '#london', "london's", "i'm"] # Some other terms to remove
            
def sentence_preprocessing(sentence):
    
    sentence_clean = []
    
    # Remove HTML entities
    sentence = BeautifulSoup(sentence, "lxml").get_text()
     
    # Tokenize
    tokens = preprocess(sentence, lowercase=False)
    
    # Step 1: Remove all useless things
    tokens = [tk for tk in tokens if tk not in punc] # Punctuation
    tokens = [tk for tk in tokens if tk.lower() not in stop] # Stopwords
    tokens = [tk for tk in tokens if re.match(link_pattern, tk) == None] # Link
    tokens = [tk for tk in tokens if emoticon_re.search(tk) == None] # Emoticons
    #tokens = [tk for tk in tokens if re.match(html_entities_pattern, tk) == None] # HTML entities
    tokens = [tk for tk in tokens if tk.lower() not in remove_terms] # Some special terms to remove
      
    # Step 2: Remove short words and non char
    tokens = [tk for tk in tokens if len(tk) >= 3] # Remove short words
    tokens = [tk for tk in tokens if re.match(number_pattern, tk) == None] # Remove number

    # Step 3: Add bigram
    tokens = [tk.lower() for tk in tokens] # Lowercase
    bigrams_words = ['_'.join(w) for w in list(bigrams(tokens))]
    sentence_clean = ' '.join(tokens) + ' ' + ' '.join(bigrams_words)
    
    return sentence_clean

# Simple tokenizer functions   
def tokenizer_simple(s):
    tokens = s.split(' ')

    # Remove tokens less than 3 chars
    tokens = [tk for tk in tokens if len(tk) >= 3]
                  
    # Remove numeric tokens
    tokens = [tk for tk in tokens if re.match(number_pattern, tk) == None]

    return tokens
    
def word2vec_wrapper(w2v_model, word):
    try:
        vec = w2v_model[word]
    except:
        vec = np.array([0]*300, dtype='float32') # Return a blank vector
    return vec
    
def word2vec_sentence(w2v_model, sentence):
    #sentence_clean = sentence_preprocessing(sentence)
    sentence_clean = tokenizer_simple(sentence)
    if len(sentence_clean) != 0:
        sent_vec = np.mean([word2vec_wrapper(w2v_model, w) for w in sentence_clean], axis=0)
    else:
        sent_vec = np.array([0]*300, dtype='float32') # Return a blank vector
    return sent_vec

#------------------------------------------------------------------------------
# Function to do text analysis and clustering

def freq_count(documents):
    
    token_count = Counter()
    for row in documents:
        tokens = row.split(' ')
        
        # Remove tokens less than 3 chars
        #tokens = [tk for tk in tokens if len(tk) >= 3]
                  
        # Remove numeric tokens
        #tokens = [tk for tk in tokens if re.match(number_pattern, tk) == None]
        
        token_count.update(tokens)
        
    return token_count
   
# Plot the wordcloud
def wordcloud_plot(token_count):

    # Calculate word frequencies
    words_freq = []
    for item in token_count.items():
        if isinstance(item[0], tuple):
            words = ' '.join(item[0])
        else:
            words = item[0]
        freq = item[1]
        words_freq.append([words, freq])

    # Generate wordcloud from words frequencies
    wc = WordCloud(background_color="black",
                   width=1024, height=768, margin=10)
    wc.generate_from_frequencies(words_freq)
    
    plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    plt.gcf().set_size_inches(6, 12)
    plt.show()

    # Show picture in a separate window
    # image = wc.to_image()
    # image.show()
    
    # Save to file
    # image.save('words_cloud_100816.jpg')
    
    # return wc
    
# Returns series of random values sampled between min and max values of passed col
def get_rand_data(col):
	rng = col.max() - col.min()
	return pd.Series(random_sample(len(col))*rng + col.min())

def iter_kmeans(df, n_clusters, num_iters=5):
	rng =  range(1, num_iters + 1)
	vals = pd.Series(index=rng)
	for i in rng:
		k = KMeans(n_clusters=n_clusters, n_init=3)
		k.fit(df)
		#print "Ref k: %s" % k.get_params()['n_clusters']
		vals[i] = k.inertia_
	return vals

def gap_statistic(df, max_k=10):
	gaps = pd.Series(index = range(1, max_k + 1))
	for k in range(1, max_k + 1):
		km_act = KMeans(n_clusters=k, n_init=3)
		km_act.fit(df)

		# get ref dataset
		ref = df.apply(get_rand_data)
		ref_inertia = iter_kmeans(ref, n_clusters=k).mean()

		gap = log(ref_inertia - km_act.inertia_)

		#print "Ref: %s   Act: %s  Gap: %s" % ( ref_inertia, km_act.inertia_, gap)
		gaps[k] = gap

	return gaps
    
#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------

# Import and clean text data
file_in = '.\\output\\31_tweets_final_v5.tsv'
df = pd.read_csv(file_in, sep='\t', encoding='utf-8')
documents = [sentence_preprocessing(t) for t in df['text']]
    
#------------------------------------------------------------------------------
# Analyse text frequencies

sw_list = stopwords.words("english")
number_pattern = r'(?:(?:\d+,?)+(?:\.?\d+)?)'
rm_terms = ['via', 'london', '#london', "london's", "i'm"]

# Top words for all
token_count = freq_count(documents)
print('Top 50 words in bookings:')
pprint(token_count.most_common(50))
print()

# Plot top words for all
tk_count_df = pd.DataFrame(token_count.most_common(50),
                           columns=['word', 'counts'])
sns.barplot(x='counts', y='word', color='blue', data=tk_count_df,
            orient='h')
sns.plt.title('Top 50 word counts')
sns.plt.ylabel('Word')
sns.plt.xlabel('Counts')
sns.plt.gcf().set_size_inches(6, 12)
sns.plt.show()

# Display wordcloud
wordcloud_plot(token_count)

# Save word freq to file
#file_out = '.\\word_counts.tsv'
#dataOut = pd.DataFrame(token_count.most_common(1000), columns=['word', 'word_counts'])
#dataOut.to_csv(file_out, sep='\t', encoding='utf-8', index=False)
    
#------------------------------------------------------------------------------
# Topic clustering - TF-IDF with K-means method

# Vectorize the text i.e. convert the strings to numeric features
t0 = time()
vectorizer = TfidfVectorizer(tokenizer=tokenizer_simple, max_df=0.95, min_df=5)                           
X_tfidf = vectorizer.fit_transform(documents)
print('Running time:', time()-t0)
print(X_tfidf.shape)
print()

# Elbow plot using interia
# Note: thist step could take very long time

np.random.seed(123)

t0 = time()

initial_tfidf = [] # List of kmeans models
wss_list_tfidf = []
hilhouette_list_tfidf = []

max_cluster = 15 # Maximum cluster

for k in range(1, max_cluster+1):
    print('Number of cluster:', k)    
    model_tfidf = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10,
                         n_jobs=-1, random_state=0, verbose=10)
    model_tfidf.fit(X_tfidf)
    initial_tfidf.append(model_tfidf)
    
    # WSS
    wss_list_tfidf.append(model_tfidf.inertia_)
    
    # Silhouette average score
    if k > 1:
        cluster_labels = model_tfidf.predict(X_tfidf)
        silhouette_avg_tfidf = silhouette_score(X_tfidf, cluster_labels, sample_size=20000)
        hilhouette_list_tfidf.append(silhouette_avg_tfidf)
    else:
        hilhouette_list_tfidf.append(0)

print('Running time:', time()-t0) # 2234.68500018

# Plot the elbow using intertia values method
plt.plot(range(1, max_cluster+1), wss_list_tfidf)
plt.title('K-means Elobow Plot \n 1-' + str(max_cluster) + ' clusters')
plt.ylabel('Sum of distances \n to the closest cluster centers')
plt.xlabel('Number of cluster')
plt.show()
# plt.savefig("kmeans_elbow_090816.jpg", dpi=600, bbox_inches='tight')

# Plot the elbow using silhouette score method
plt.plot(range(1, max_cluster+1), hilhouette_list_tfidf)
plt.title('K-means Elobow Plot \n 1-' + str(max_cluster) + ' clusters')
plt.ylabel('Silhouette score')
plt.xlabel('Number of cluster')
plt.show()

# Plot using gap statistic method
X_tfidf_sample = pd.DataFrame(X_tfidf[np.random.choice(X_tfidf.shape[0], 20000, replace=False), :])
gaps_tfidf = gap_statistic(X_tfidf_sample, max_cluster)

plt.plot(gaps_tfidf)
plt.title('K-means Elobow Plot \n 1-' + str(max_cluster) + ' clusters')
plt.ylabel('Gaps stats')
plt.xlabel('Number of cluster')
plt.show()

#------------------------------------------------------------------------------
# Topic clustering - Word2Vec with K-means method

# Train Word2Vec model
documents_token = [tokenizer_simple(d) for d in documents]
w2v_model = Word2Vec(documents_token, min_count=5, size=300)

# Convert all documents to vector
documents_vec = [word2vec_sentence(w2v_model, s) for s in documents]

# Feature selections
X_w2v = np.array(documents_vec)

pca = PCA()
pca.fit(X_w2v)
print(pca.explained_variance_ratio_)

plt.bar(range(0, X_w2v.shape[1]), pca.explained_variance_)
plt.show()

plt.bar(range(0, X_w2v.shape[1]), pca.explained_variance_ratio_)
plt.show()

plt.plot(pca.explained_variance_ratio_[:15])
plt.show()

plt.plot(np.cumsum(pca.explained_variance_ratio_[:15]))
plt.show()

X_w2v_selected = X_w2v[:,:10] # Top 10 features explained >90% data
                
# Elbow plot using interia
# Note: thist step could take very long time

np.random.seed(123)

t0 = time()

model_list_w2v = [] # List of kmeans models
wss_list_w2v = [] # List of inertia
silhouette_list_w2v = [] # List of silhouette score

max_cluster = 15 # Maximum cluster

for k in range(1, max_cluster+1):
    print('Number of cluster:', k)    
    model_w2v = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10,
                       n_jobs=-1, random_state=0, verbose=10)
    model_w2v.fit(X_w2v_selected)
    model_list_w2v.append(model_w2v)
    
    # WSS
    wss_list_w2v.append(model_w2v.inertia_)
    
    # Silhouette average score
    if k > 1:
        cluster_labels = model_w2v.predict(X_w2v_selected)
        silhouette_avg_w2v = silhouette_score(X_w2v_selected, cluster_labels, sample_size=20000)
        silhouette_list_w2v.append(silhouette_avg_w2v)
    else:
        silhouette_list_w2v.append(0)

print('Running time:', time()-t0) # 2234.68500018

# Plot the elbow using intertia values
plt.plot(range(1, max_cluster+1), wss_list_w2v)
plt.title('K-means Elobow Plot \n 1-' + str(max_cluster) + ' clusters')
plt.ylabel('Sum of distances \n to the closest cluster centers')
plt.xlabel('Number of cluster')
plt.show()
# plt.savefig("kmeans_elbow_090816.jpg", dpi=600, bbox_inches='tight')

# Plot the elbow using silhouette score
plt.plot(range(1, max_cluster+1), silhouette_list_w2v)
plt.title('K-means Elobow Plot \n 1-' + str(max_cluster) + ' clusters')
plt.ylabel('Silhouette score')
plt.xlabel('Number of cluster')
plt.show()
# plt.savefig("kmeans_elbow_090816.jpg", dpi=600, bbox_inches='tight')                 

# Plot using gap statistic method
X_w2v_sample = pd.DataFrame(X_w2v_selected[np.random.choice(X_w2v_selected.shape[0], 20000, replace=False), :])
gaps_w2v = gap_statistic(X_w2v_sample, max_cluster)

plt.plot(gaps_w2v)
plt.title('K-means Elobow Plot \n 1-' + str(max_cluster) + ' clusters')
plt.ylabel('Gaps stats')
plt.xlabel('Number of cluster')
plt.show()

# 7 cluster is the best

#------------------------------------------------------------------------------
# Run the clustering again with best k

X = X_w2v_selected # Use Word2Vec model
k_best = 10

# Clustering
model = KMeans(n_clusters=k_best, init='k-means++', max_iter=300, n_init=10,
               n_jobs=-1, random_state=0, verbose=10)
model.fit(X)

# Evaluate with silhouette score
cluster_labels = model.predict(X)
silhouette_avg = silhouette_score(X, cluster_labels, sample_size=10000)
print('Silhouette average score:', silhouette_avg)

# Save the topic clustering results
file_out = '.\\31_tweets_final_v5.tsv'
df['topic_cluster'] = model.labels_
df.to_csv(file_out, sep='\t', encoding='utf-8', index=False)

# Plot the cluster size
documents_df = pd.DataFrame({'account_type_pred':df['account_type_pred'],
                             'text':df['text'],
                             'text_clean':documents,
                             'cluster':model.labels_})
documents_df = documents_df[documents_df['account_type_pred'] == 'idv']
documents_df.drop('account_type_pred', axis=1, inplace=True)

documents_df['cluster'].value_counts()
documents_df['cluster'].value_counts().plot(kind='bar')
   
# Extract cluster information
word_count_by_topic = pd.DataFrame()
for cluster in range(0, k_best):
    
    # Extract documents related to the cluster
    docs = documents_df[documents_df['cluster'] == cluster]
    
    # Top words for all
    token_count = freq_count(docs['text_clean'])
    
    print('Cluster', cluster, ':')
    print()
    print('Top words:')
    for tk in token_count.most_common(10): print(tk)
    print()
    
    # Some tweets
    print('Sample tweets:')
    for t in docs['text'].head(5).values: print(t)
    print()
    
    # Construct the word counts data
    wc = pd.DataFrame(token_count.most_common(100), columns=['word', 'counts'])
    wc['topic_cluster'] = cluster   
    word_count_by_topic = word_count_by_topic.append(wc, ignore_index=True)

# Save to file    
file_out = '.\\output\\word_count_by_topic_cluster.tsv'    
word_count_by_topic.to_csv(file_out, sep='\t', encoding='utf-8', index=False)

#------------------------------------------------------------------------------