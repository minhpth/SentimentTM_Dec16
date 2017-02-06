#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# WRAPPER OF CMU POS TAGGER
#------------------------------------------------------------------------------

# Ref: https://github.com/ianozsvald/ark-tweet-nlp-python/blob/master/CMUTweetTagger.py

"""Simple Python wrapper for runTagger.sh script for CMU's Tweet Tokeniser and Part of Speech tagger: http://www.ark.cs.cmu.edu/TweetNLP/

Usage:
results=runtagger_parse(['example tweet 1', 'example tweet 2'])
results will contain a list of lists (one per tweet) of triples, each triple represents (term, type, confidence)
"""

import subprocess
import copy

RUN_TAGGER_CMD = ['java',
                  '-XX:ParallelGCThreads=2', '-Xmx500m' ,'-jar', 
                  './/model//pos_tagger//ark-tweet-nlp-0.3.2//ark-tweet-nlp-0.3.2.jar']

def _split_results(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0]
                tags = parts[1]
                confidence = float(parts[2])
                yield tokens, tags, confidence


def _call_runtagger(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""
    n_tweets=len(tweets)#lo agwegue yo
    # remove carriage returns as they are tweet separators for the stdin
    # interface
    tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
    message = "\n".join(tweets_cleaned)

    # force UTF-8 encoding (from internal unicode type) to avoid .communicate encoding error as per:
    # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
    print(message)
    
    #message = message.encode('utf-8')#separa los mesg con \n=enter, los pega
    #I removed this, it was ok when unicode. but for "clean msg" it gave error    
    
    
    #save in temporary text file
    with open('text_aux.txt', 'w') as f:#write con overwrite
        f.write(message)
    
    # build a list of args
    args = copy.deepcopy(run_tagger_cmd)
    args.append('--output-format')
    args.append('conll')
    args.append('.//text_aux.txt')

    
    #print args
    po = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = [line for line in po.stdout]
    #print result    
    pos_results={}
    tweet_res=[]
    i=0#numer of tweets
    for j in range(1,len(result)-1):#discard first and two last lines        
        element=result[j]
        if element=='\r\n':
            pos_results[i]=tweet_res
            i=i+1
            tweet_res=[]
        else:          
            line_aux=element.strip('\n')
            sub_list=line_aux.split('\t')
            tweet_res.append(sub_list)
       
     #me contrullo un dict, dict[i] tiene info sobre tweet i,
     #dict[i] tiene una list ade lista, la list atiene lista con listas de
     #3 componentes, la palabra parseada, el tipo de palabre y la prob 
     
    # expect a tuple of 2 items like:
    # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
    # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')
    #print result
    #print len(result)
    
    #pos_result = result[0].strip('\n\n')  # get first line, remove final double carriage return
    #pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
    #pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
    #pos_results=result[1:]#first line has "detected text input format"

    #print     pos_results
    return  pos_results

def runtagger_parse(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
    #print 'comando= '
    #print run_tagger_cmd     
    pos_raw_results = _call_runtagger(tweets, run_tagger_cmd)
    #print 'termino java'    
    #print pos_raw_results
    
    #print 'ahora python'    
    #pos_result = []
    #for pos_raw_result in pos_raw_results:
    #    pos_result.append([x for x in _split_results(pos_raw_result)])
    #return pos_result
    return pos_raw_results

def check_script_is_present(run_tagger_cmd=RUN_TAGGER_CMD):
    """Simple test to make sure we can see the script"""
    success = False
    try:
        
        args=copy.deepcopy(run_tagger_cmd)
        args.append('--help')
        
        po = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #print run_tagger_cmd
        #lines=[]        
        #for line in po.stdout:
        #    lines.append(line)
        #while not po.poll():#no anda en windoes
        lines = [line for line in po.stdout]
        #print lines
                
        assert "RunTagger [options]" in lines[0]
        #print 'Line0= '+ lines[0]
        success = True
    except OSError as err:
        print("Caught an OSError, have you specified the correct path to runTagger.sh? We are using \"%s\". Exception: %r" % (run_tagger_cmd, repr(err)))
    return success

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------
    
if __name__ == "__main__":
    print("Checking that we can see \"%s\", this will crash if we can't" % (RUN_TAGGER_CMD))
   
    success = check_script_is_present()
    if success:
        print("Success.")
        print("Now pass in two messages, get a list of tuples back:")
        tweets = ['this is a message', 'and a second message']
        print(runtagger_parse(tweets))
        
#------------------------------------------------------------------------------