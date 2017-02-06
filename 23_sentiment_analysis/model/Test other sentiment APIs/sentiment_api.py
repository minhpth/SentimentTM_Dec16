# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# SENTIMENT API TESTING
#------------------------------------------------------------------------------

import os
os.chdir('D:\\SentimentTM\\23_sentiment_analysis\\model\\Test other sentiment APIs')

#------------------------------------------------------------------------------
# Test 1: text-processing.com
#------------------------------------------------------------------------------

import pandas as pd
import urllib
import urllib2
import json
from time import sleep

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import pickle

# Import input data
file_in = '.\\input\\data_challenge2013\\tweeti-b.dev.distFULL.tsv'
dataIn = pd.read_csv(file_in, sep='\t', error_bad_lines=False)

# Drop data
dataIn.loc[dataIn['polarity'] == 'objective', 'polarity'] = 'neutral'
dataIn.loc[dataIn['polarity'] == 'objective-OR-neutral', 'polarity'] = 'neutral'

def get_sentiment(text):
    
    # Get sentiment from Microsoft text-processing.com
    url = "http://text-processing.com/api/sentiment/"
    data = urllib.urlencode({"text": text})
    
    num_retry = 3 # Retry 3 times
    count_retry = 0
    delay = 30 # secs
    
    u = None
    while  count_retry < num_retry:
        try:
            u = urllib2.urlopen(url, data, timeout=30)
            break # If no error, exit while loop
            
        except urllib2.URLError as e:
            print('Page load error:', e)
            print('Waiting and retrying...')
            count_retry += 1
            sleep(delay)
    
    if u is not None:
        the_page = u.read()
        the_page_json = json.loads(the_page)
        
        # Print some results
        print(text)
        print('Sentiment:', the_page_json['label'])
        print('neg proba.:', the_page_json['probability']['neg'])
        print('neutral proba.:', the_page_json['probability']['neutral'])
        print('pos proba.:', the_page_json['probability']['pos'])
        print()
    else:
        the_page = None
        the_page_json = None
        
        # Print some errors
        print(text)
        print('Cannot load page.')
        print()
    
    return the_page_json

# Request sentiment from page api
text_col = 'text'
sentiment_list = [get_sentiment(t) for t in dataIn[text_col]]

# Backup results                  
file_out = '.\\output\\text_processing_results.pkl'
pickle.dump(sentiment_list, open(file_out, 'wb'))

# Evaluation
target_col = 'polarity'
#dataIn.loc[dataIn[target_col] == 'objective'] = 'neutral'
#dataIn.loc[dataIn[target_col] == 'objective-OR-neutral'] = 'neutral'

le = LabelEncoder().fit(dataIn['polarity'])
target_test = le.fit_transform(dataIn['polarity'])

def convert_target(s):
    s = str(s['label'])
    if s == 'pos':
        return 'positive'
    if s == 'neg':
        return 'negative'
    if s == 'neutral':
        return 'neutral'
    return None

target_pred = [convert_target(s) for s in sentiment_list]
target_pred = le.transform(target_pred)

print(classification_report(target_test, target_pred))
print(confusion_matrix(target_test, target_pred))
print(accuracy_score(target_test, target_pred))

#------------------------------------------------------------------------------
# Microsoft Cognitive Services (Text Analytics API)
#------------------------------------------------------------------------------

# Ref: https://github.com/Microsoft/ProjectOxford-ClientSDK/blob/master/Face/Python/Jupyter%20Notebook/Face%20Detection%20Example.ipynb

# Variables
_url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment'
_key = '4b1ccc301d804e668505101057f7cf71'

# Create the request data body
request_body = pd.DataFrame({'language':'en',
                             'text':dataIn[text_col]})
request_body['id'] = request_body.index.values.astype('str')

# Converting the Request body JSON
request_body_json = '{"documents":' + request_body.to_json(orient='records') + '}'

headers = dict()
headers['Ocp-Apim-Subscription-Key'] = _key
headers['Content-Type'] = 'application/json' 

# Request from Microsoft Cognitive Services API
req = urllib2.Request(_url, request_body_json, headers)
response = urllib2.urlopen(req)
result = response.read()
obj = json.loads(result)

# Backup results                  
file_out = '.\\output\\microsoft_cognitive_results.pkl'
pickle.dump(obj, open(file_out, 'wb'))

# Conver JSON results to DataFrame
obj_df = pd.DataFrame.from_dict(obj['documents'])

# Encode [1: positive, 0: negative]
obj_df['polarity'] = 'neutral' # Create new column
obj_df.loc[obj_df['score'] >= 1/3*2, 'polarity'] = 'positive'
obj_df.loc[obj_df['score'] < 1/3, 'polarity'] = 'negative'

# Evaluation
le = LabelEncoder().fit(dataIn['polarity'])
target_test = le.fit_transform(dataIn['polarity'])

target_pred = obj_df['polarity']
target_pred = le.transform(target_pred)

print(classification_report(target_test, target_pred))
print(confusion_matrix(target_test, target_pred))
print(accuracy_score(target_test, target_pred))

#------------------------------------------------------------------------------