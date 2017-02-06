#======================================================================
# TWITTER SENTIMENT ANALYSIS PROJECT
#======================================================================

Module Sentiment Classification Model [APPLY MODEL]

NOTE: In this step, we use CMU Tweet Tagger to do POS Tagging on the
tweets. If you have any error when running the POS Tagging, go to the
file CMUTweetTagger_windowsMod.py and try to make it run first.

1. Copy the result of previous module Tweets Cleansing (14_tweets_clean)
   into INPUT folder. Run 1_feature_extraction.py to extract the text
   features from all the tweets.
   
2. Run the script 2_classifier_model.py to apply the classification model
   on the new tweets data. Get the result in OUTPUT folder.