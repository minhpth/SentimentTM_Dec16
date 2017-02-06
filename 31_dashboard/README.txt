#======================================================================
# TWITTER SENTIMENT ANALYSIS PROJECT
#======================================================================

Module Dashboard Tableau

1. Copy the result of module TWEETS CLEANSING (14_tweets_clean) into
   INPUT folder.
   File: 14_tweets_location_cleaned.tsv
   
2. Copy the result of module Organization vs. Individual Classification
   (17_account_type_filter) into INPUT folder.
   File: 17_twitter_account_type_classified.tsv
   
3. Copy the result of module Gender Classification Model
   (21_gender_model) into INPUT folder.
   File: allUsers_fullFeaturesGenderOUT_30k_classified.tsv
   
4. Copy the result of module Age-Group Classification Model
   (22_age_model) into INPUT folder.
   File: allUsers_fullFeaturesAgeOUT_30k_classified.tsv
   
5. Copy the result of module Sentiment Classification Model
   (23_sentiment_analysis) into INPUT folder. 
   File: 14_tweets_location_with_sentiment.tsv
   
6. Run the script combine_results.py to combine all results of the
   previous steps. And enrich the data with geo-locations information.
   Get the final result in OUTPUT folder.
   
7. Run the script tweets_topic_clustering.py to explore the topics
   inside the final tweets result.
   
   This script will also create a cluster file, using in the final
   dashboard. Find the result in OUTPUT folder.
   
8. Open the Tableau Dashboard (SentimentTM_Dashboard_v2.twb) then
   connect with the 2 results files in OUTPUT folder. Use the dashboard
   to explore all the results.