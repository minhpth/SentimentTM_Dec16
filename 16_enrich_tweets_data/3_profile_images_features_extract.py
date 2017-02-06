# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# ACCOUNT TYPE CLASSIFIER - IMAGE FEATURE EXTRACT
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
os.chdir('D:\\SentimentTM\\16_enrich_tweets_data')

#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
import numpy as np
from time import time

# Other functional packages
import mahotas as mh
from skimage import io
from skimage.color import rgb2luv, rgb2hsv

#------------------------------------------------------------------------------
# Functions to extract image features
#------------------------------------------------------------------------------

def balanceTrainigSet(dataFull, col_target, p_balance, trainTest_balance):

    classes = np.array(dataFull[col_target].unique())
    size_clases = [dataFull[col_target].value_counts()[clas] for clas in classes]

    index_total = dataFull.index
    index_train = []

    if p_balance == -1: # 50/50
        n_split = int(np.round(np.min(size_clases) * trainTest_balance))
        for c in classes:
            index_c = dataFull[col_target][dataFull[col_target] == c].index.tolist()
            #np.random.shuffle(index_c)
            #i_c_train = index_c[0:n_split]
            i_c_train = np.random.choice(index_c, size=n_split, replace=False).tolist()
            index_train = index_train + i_c_train
    else:
        size_min = np.min(size_clases)
        list_split = [s*trainTest_balance if s==size_min else p_balance*s*trainTest_balance for s in size_clases]
        for i in range(len(classes)):
            c = classes[i]
            n_split = int(round(list_split[i]))
            index_c = dataFull[col_target][dataFull[col_target] == c].index.tolist()
            #np.random.shuffle(index_c)
            #i_c_train = index_c[0:n_split]
            i_c_train = np.random.choice(index_c, size=n_split, replace=False).tolist()
            index_train = index_train + i_c_train

    index_train = np.array(index_train)
    np.random.shuffle(index_train)
    index_test = np.array(list(set(index_total)-set(index_train)))
    np.random.shuffle(index_test)

    return index_train, index_test

# Create a list of image features names
def findFeaturesFeatureNames():
    f_names = ['contrast','avg_h','avg_s', 'avg_v', 'pleasure', 'arousal', 'dominance']

    f_names = f_names + ['hue_hist_' + str(i) for i in range(12)]
    f_names = f_names + ['saturation_hist_' + str(i) for i in range(5)]
    f_names = f_names + ['brightness_hist_' + str(i) for i in range(3)]
    f_names = f_names + ['std_hueHist', 'std_satHist', 'std_brHist']
    f_names = f_names + ['haralicks_f_' + str(i) for i in range(13)]
    return f_names

# Extract image features
def findFeatures_img(pic_name):

    try:
        imgRGB = io.imread(pic_name)
        if imgRGB.shape[2] > 3:
            imgRGB = imgRGB[:, :, 0:3] # Remove alpha channel

        # Texture
        haralicks_f = mh.features.haralick(imgRGB).mean(0) # 13 features

        # Luminnace
        yuv_img = rgb2luv(imgRGB)
        luminance = yuv_img[:, :, 0]
        contrast = (luminance.max()-luminance.min())/luminance.mean()

        # HSV
        hsv_img = rgb2hsv(imgRGB)
        avg_h = hsv_img[:,:,0].mean() # hue
        avg_s = hsv_img[:,:,1].mean() # saturation
        avg_v = hsv_img[:,:,2].mean() # value/brightess
        pleasure = 0.69*avg_v+0.22*avg_s
        arousal = -0.31*avg_v+0.6*avg_s
        dominance = 0.76*avg_v+0.32*avg_s

        # Itten features
        hue_hist = np.histogram(hsv_img[:,:,0],bins=12)
        saturation_hist = np.histogram(hsv_img[:,:,1],bins=5)
        brightness_hist = np.histogram(hsv_img[:,:,2],bins=3)
        std_hueHist = np.std(hue_hist[0])
        std_satHist = np.std(saturation_hist[0])
        std_brHist = np.std(brightness_hist[0])

        features = [[contrast,avg_h, avg_s, avg_v, pleasure, arousal, dominance],hue_hist[0],
                    saturation_hist[0], brightness_hist[0], [std_hueHist, std_satHist,std_brHist,
                    ], haralicks_f]
        features = [item for sublist in features for item in sublist] # Flatten

        return features

    except:
        print('Unable to read image.')
        return None

# Function to create image features for 1 image file
def calulateFeatures(row):

    idCol='user_id_str'
    fileExt='.jpg'
    #print row
    pic_name = os.path.join(imgs_dir, str(row[idCol]  )+fileExt )
    print(pic_name)
    features = findFeatures_img(pic_name)
    if features is None: # otherwise leave the 0s
        print('Cannot extract features. leave the 0.')
    else:
        newCols = findFeaturesFeatureNames()
        row[newCols] = features

    return row

# Function to process all profile images and return result in a DataFrame
def process_csv(df):

     newCols = findFeaturesFeatureNames()
     for c in newCols: df[c] = 0 # Add new columns
     df.update(df.apply(calulateFeatures, axis=1))

     return df

#------------------------------------------------------------------------------
# MAIN: Extract all features
#------------------------------------------------------------------------------

# Read user profile information
file_in = '.\\input\\14_tweets_location_cleaned.tsv'
userData = pd.read_csv(file_in, sep='\t', encoding='utf-8')

# Keep unique user_id
userData.drop_duplicates('user_id_str', keep='first', inplace=True)

# Check the image files
imgs_dir = '.\\output\\16_twitter_profile_images' #'./imgs'
filesImgs = os.listdir(imgs_dir)
filesImgs_noExt = [int(fn.split('.')[0]) for fn in filesImgs] # Get all files names

file_ext = set([(fn.split('.')[1]) for fn in filesImgs]) # only jpg, just to check
print(file_ext)

# Drop some user_id in case some picture was not downloaded
userData = userData[userData['user_id_str'].isin(filesImgs_noExt)]
userData = userData.reset_index(drop=True)

# Extract images features
t0 = time()
userDataWithF = process_csv(userData)
print('Running time:', time()-t0)

# Save to file
file_out = '.\\output\\16_tweets_location_with_image_features.tsv'
userDataWithF.to_csv(file_out, sep='\t', encoding='utf-8', index=False)

#------------------------------------------------------------------------------