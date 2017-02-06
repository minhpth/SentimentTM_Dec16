#======================================================================
# TWITTER SENTIMENT ANALYSIS PROJECT
#======================================================================

Module Enrich Tweets Data (with Image and Text Features)

WARNING: This module can take very long time too run.
NOTE: Download, extract image features first, then text features.

1. Copy the result of module TWEETS CLEANSING (14_tweets_clean) and put
   into the INPUT folder.
   
2. Run the 1_twitter_profile_images_download.py to download profile
   images of all Twitter profiles. This step can take very long time to
   run.
   
   The profile image can be in *.JPG, *.JPEG, *.GIF, *.BMP or *.PNG.
   
3. Run the 2_profile_images_cleaner.py to convert *.GIF, *.BMP, *.PNG
   profile images to *.JPG.
   
4. Run the 3_profile_images_features_extract.py to extract all image
   features. This step can take very long time to run.
   
5. Run the 4_twitter_profile_tweets_download.py to extract historical
   tweets of all Twitter profiles. If the Twitter App doesn't work,
   create a new Twitter App and replace the keys and tokens.
   
6. Run the script 5_profile_tweets_features_extract.py to extract all
   text features and combine them with image features. Get the final
   result in OUTPUT folder.
   
   Note: This step will require Google_Word2Vec pretrained model
   in D:\SentimentTM\Models folder. If the model is missing, find and
   download it from here: https://code.google.com/archive/p/word2vec/