#======================================================================
# TWITTER SENTIMENT ANALYSIS PROJECT
#======================================================================

Module Age-Group Classification Model [BUILD MODEL]

1. Run the script 1_caclulateFeatures_user.py to extract all text
   features from the train data set. Get the features file in OUTPUT
   folder.
   
2. Run the 2_join_age_features_users.py to combine all text features
   with the original text data. Get the result in OUTPUT folder.
   
3. Run the 3_build_classification.py to build the classification model and
   get the model file (*.pkl) in OUTPUT folder.
   
   Note: You can try with more classification models and save the best
   one in the script.