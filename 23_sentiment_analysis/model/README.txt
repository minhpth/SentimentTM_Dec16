#======================================================================
# TWITTER SENTIMENT ANALYSIS PROJECT
#======================================================================

Module Sentiment Classification Model [BUILD MODEL]

NOTE: In this step, we use CMU Tweet Tagger to do POS Tagging on the
tweets. If you have any error when running the POS Tagging, go to the
file CMUTweetTagger_windowsMod.py and try to make it run first.

1. Run the script 1_feature_extraction.py to extract all text
   features from the train data set. Get the features file in OUTPUT
   folder.
      
2. Run the 2_classifier_model.py to build the classification model and
   get the model file (*.pkl) in OUTPUT folder.
   
   Note: You can try with more classification models and save the best
   one in the script.