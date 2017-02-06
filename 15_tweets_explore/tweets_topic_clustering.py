# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# TWEETS TOPIC CLUSTERING
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
os.chdir('D:\\SentimentTM\\15_tweets_explore')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from time import time
from collections import Counter

# Other functional packages
import cld # Language classifier
from pprint import pprint # Print in a nicer way
from wordcloud import WordCloud # Show wordcloud

from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF
from sklearn.cluster import KMeans # Topics clustering
from gensim.models import Word2Vec

import seaborn as sns
from glob import glob

#------------------------------------------------------------------------------
# Import data
#------------------------------------------------------------------------------

file_in = '.\\input\\14_tweets_location_cleaned.tsv'
df = pd.read_csv(file_in, sep='\t', encoding='utf-8')
documents = df['text_clean']

#------------------------------------------------------------------------------
# Analyse text frequencies
#------------------------------------------------------------------------------

from nltk.corpus import stopwords

sw_list = stopwords.words("english")

number_pattern = r'(?:(?:\d+,?)+(?:\.?\d+)?)'

def freq_count(documents):
    
    token_count = Counter()
    for row in documents:
        tokens = row.split()
        
        # Remove tokens less than 3 chars
        tokens = [tk for tk in tokens if len(tk) >= 3]
                  
        # Remove numeric tokens
        tokens = [tk for tk in tokens if re.match(number_pattern, tk) == None]
        
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

# Top words for all
token_count = freq_count(documents)
print('Top 50 words in bookings:')
pprint(token_count.most_common(50))
print()

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
    
#------------------------------------------------------------------------------
# Topic clustering - TF-IDF with K-means method
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Vectorize the documents

t0 = time()

# Simple tokenizer functions
# Since the text is already pre-processing in the previous steps
def tokenizer_simple(s):
    tokens = s.split()

    # Remove tokens less than 3 chars
    tokens = [tk for tk in tokens if len(tk) >= 3]
                  
    # Remove numeric tokens
    tokens = [tk for tk in tokens if re.match(number_pattern, tk) == None]

    return tokens

# vectorize the text i.e. convert the strings to numeric features
vectorizer = TfidfVectorizer(tokenizer=tokenizer_simple,
                             ngram_range=(1, 2),
                             max_df=0.95, min_df=5)
                             
X = vectorizer.fit_transform(documents)
print('Running time:', time()-t0)
print(X.shape)
print()

#------------------------------------------------------------------------------
# Elbow plot using interia
# Note: thist step could take very long time

np.random.seed(123)

t0 = time()

initial = [] # List of kmeans models
max_cluster = 50 # Maximum cluster    
for k in range(1, max_cluster+1):
    print('Number of cluster:', k)    
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=2)
    model.fit(X)
    initial.append(model)

print('Running time:', time()-t0) # 2234.68500018

# Plot the elbow using intertia values
plt.plot(range(1, max_cluster+1), [md.inertia_ for md in initial])
plt.title('K-means Elobow Plot \n 1-50 clusters')
plt.ylabel('Sum of distances \n to the closest cluster centers')
plt.xlabel('Number of cluster')
plt.show()
# plt.savefig("kmeans_elbow_090816.jpg", dpi=600, bbox_inches='tight')

#------------------------------------------------------------------------------
# Topic clustering - Word2Vec with K-means method
#------------------------------------------------------------------------------

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

# Train Word2Vec model
documents_token = [tokenizer_simple(d) for d in documents]
w2v_model = Word2Vec(documents_token, min_count=5, size=300)

# Convert all documents to vector
documents_vec = [word2vec_sentence(w2v_model, s) for s in documents]
                
#------------------------------------------------------------------------------
# Elbow plot using interia
# Note: thist step could take very long time

np.random.seed(123)

t0 = time()

X = np.array(documents_vec)

initial = [] # List of kmeans models
max_cluster = 50 # Maximum cluster    
for k in range(1, max_cluster+1):
    print('Number of cluster:', k)    
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=2)
    model.fit(X)
    initial.append(model)

print('Running time:', time()-t0) # 2234.68500018

# Plot the elbow using intertia values
plt.plot(range(1, max_cluster+1), [md.inertia_ for md in initial])
plt.title('K-means Elobow Plot \n 1-50 clusters')
plt.ylabel('Sum of distances \n to the closest cluster centers')
plt.xlabel('Number of cluster')
plt.show()
# plt.savefig("kmeans_elbow_090816.jpg", dpi=600, bbox_inches='tight')                 

# 7 cluster is the best

#------------------------------------------------------------------------------
# Run the clustering again with best k

model = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=10)
model.fit(X)

documents_df = pd.DataFrame({'remark_clean':documents,
                             'cluster':model.labels_})
documents_df['cluster'].value_counts()
documents_df['cluster'].value_counts().plot(kind='bar')

#------------------------------------------------------------------------------
# Extract documents in the same cluster

cluster_no = 3

docs = documents_df[documents_df['cluster'] == cluster_no]

# Top words for all
token_count = freq_count(docs['remark_clean'])
print('Top 50 words:')
pprint(token_count.most_common(50))
print()

tk_count_df = pd.DataFrame(token_count.most_common(50),
                           columns=['word', 'counts'])
sns.barplot(x='counts', y='word', color='blue', data=tk_count_df,
            orient='h')
sns.plt.title('Top 50 word counts')
sns.plt.ylabel('Word')
sns.plt.xlabel('Counts')
sns.plt.gcf().set_size_inches(6, 12)
sns.plt.show()

wordcloud_plot(token_count)

#------------------------------------------------------------------------------
# Visualize all clusters information
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Create all WordCloud photos and save to files

# Save the wordcloud to file
def wordcloud_save(token_count, title, fname):

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
    plt.title(title, fontsize=40)
    plt.imshow(wc)
    plt.axis("off")
    #plt.gcf().set_size_inches(6, 12)
    #plt.show()
    plt.savefig(fname, dpi=600)
    
def cluster_to_wordcloud(cluster_no, documents_df):
    docs = documents_df[documents_df['cluster'] == cluster_no]
    token_count = freq_count(docs['remark_clean'])
    title = 'Top words in cluster ' + str(cluster_no) # Create title for the plot
    fname = '.\\output\\wordclound_cluster_' + str(cluster_no) + '.png' # Create file name for the plot
    wordcloud_save(token_count, title, fname)
    
k = 7 # Number of clusters
for i in range(k): cluster_to_wordcloud(i, documents_df)

#------------------------------------------------------------------------------
# Show WordCloud for all clusters at one

img_list = sorted(glob('.\\output\\wordclound_cluster_*.png')) # Get all images names

import pylab
#import matplotlib.cm as cm
from PIL import Image

k = 7 # Total number of clusters

f = pylab.figure()
for n, fname in enumerate(img_list):
    image = Image.open(fname)
    arr = np.asarray(image)
    f.add_subplot(7//3+1, 3, n+1)
    pylab.imshow(arr)
    pylab.axis('off')
    pylab.gcf().set_size_inches(14, 10)
    #ylab.title('Hello world')
    
#pylab.title('Double image')
pylab.show()

#------------------------------------------------------------------------------