#======================================================================
# TWITTER SENTIMENT ANALYSIS PROJECT
#======================================================================

Module Gender Classification Model [APPLY MODEL]

NOTE: This module links with the previous module ENRICH TWEETS DATA
(16_enrich_tweets_data), you have to finish running the previous module
before going with this module.

1. Run the 1_twitter_caclulateFeatures_user.py to extract all text 
   features of the download tweets in previous ENRICH TWEETS DATA
   module. The result file will be in OUTPUT folder.
   
2. Copy the result of previous module Tweets Cleansing (14_tweets_clean)
   into INPUT folder. Run 2_twitter_join_gender_features_users.py to
   join text features with the original tweets file.
   
3. Run the script 3_twitter_classification.py to apply classification
   model on the new tweets data. Get the result in OUTPUT folder.