# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# PROFILE IMAGES CLEANER
#------------------------------------------------------------------------------

# Functions:
# [1] Delete profile image files that are not matches with user_id
# [2] Convert all image file types to JPG

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
from time import time

# Other functional packages
from PIL import Image
from os import listdir
from os.path import isfile, join

#------------------------------------------------------------------------------
# MAIN: Delete all image files not related to user_id
#------------------------------------------------------------------------------

img_folder = '.\\output\\16_twitter_profile_images'

# Image file list and file name
file_list = [fn for fn in listdir(img_folder) if isfile(join(img_folder, fn))]
file_name = [int(os.path.splitext(fn)[0]) for fn in file_list]
             
# Usable user_id list

# user_id from downloaded tweets
file_in = '.\\input\\14_tweets_location_cleaned.tsv'
userData = pd.read_csv(file_in, sep='\t', encoding='utf-8')             
userid_list = list(userData['user_id_str'])

# user_id from train data of organizational vs. individual model
#file_in = user_profile_type_train_data
#userData2 = pd.read_csv(file_in, sep='\t', encoding='utf-8')
#userid_list = userid_list + list(userData2['id_str'])

userid_list = set(userid_list) # Unique user_id

# Find file are not in user_id list             
index = dict((y,x) for x,y in enumerate(userid_list))

def in_index(userid):
    global index
    try:
        index[userid]
        return True
    except:
        return False

in_userid_list = [in_index(userid) for userid in file_name] # 52784
delete_file_list = list(pd.Series(file_list)[-pd.Series(in_userid_list)]) # 37422
                  
# Delete not related files
rm_file = [os.remove(img_folder + '\\' + fn) for fn in delete_file_list]

#------------------------------------------------------------------------------
# MAIN: Convert other photo formats (e.g. PNG, GIF, etc.) to JPG
#------------------------------------------------------------------------------

# Function to convert image
def convert_img(file_in, ext_out):
    file_out = os.path.splitext(file_in)[0] + ext_out
    if os.path.isfile(file_out): os.remove(file_out) # Delete file exist
    try:    
        im = Image.open(file_in)
        im.convert('RGB').save(file_out)
        return True
    except:
        print('ERROR:', file_in)
        return False

# Summary file list and types
file_list = [fn for fn in listdir(img_folder) if isfile(join(img_folder, fn))]
file_type = set([os.path.splitext(fn)[-1].lower() for fn in file_list])
print(file_type)

# Delete file without extension
noExt_list = [fn for fn in file_list if os.path.splitext(fn)[-1].lower() == '']
noExt_delete = [os.remove(img_folder + '\\' + fn) for fn in noExt_list]

# Convert all other file types to JPG        
jpg_list = [fn for fn in file_list if os.path.splitext(fn)[-1].lower() == '.jpg']
jpeg_list = [fn for fn in file_list if os.path.splitext(fn)[-1].lower() == '.jpeg']
png_list = [fn for fn in file_list if os.path.splitext(fn)[-1].lower() == '.png']
bmp_list = [fn for fn in file_list if os.path.splitext(fn)[-1].lower() == '.bmp']
gif_list = [fn for fn in file_list if os.path.splitext(fn)[-1].lower() == '.gif']

png_convert = [convert_img(join(img_folder, fn), '.jpg') for fn in png_list] # Convert all PNG to JPG
jpeg_convert = [convert_img(join(img_folder, fn), '.jpg') for fn in jpeg_list] # Convert all JPEG to JPG
gif_convert = [convert_img(join(img_folder, fn), '.jpg') for fn in gif_list] # Convert all GIF to JPG
bmp_convert = [convert_img(join(img_folder, fn), '.jpg') for fn in bmp_list] # Convert all BMP to JPG

#------------------------------------------------------------------------------