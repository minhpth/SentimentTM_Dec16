#======================================================================
# TWITTER SENTIMENT ANALYSIS PROJECT
#======================================================================

Module Organization vs. Individual Classification Model [BUILD MODEL]

NOTE: Always extract image features firstly then text features.

1. Make sure to copy the training data (text and profile pictures) into
   INPUT folder.
   
2. Run the script 1_user_profile_image_feature_extract.py to extract all
   image features.
   
3. Run the 2_user_profile_text_feature_extract.py to extract all text
   features and combine them with image features. Get the result in
   OUTPUT folder.
   
   Note: This step will require Google_Word2Vec pretrained model
   in D:\SentimentTM\Models folder. If the model is missing, find and
   download it from here: https://code.google.com/archive/p/word2vec/
   
4. Run the script 3_account_type_classifier_build_model.py to train the
   xgboost model and get the model file in OUTPUT folder.