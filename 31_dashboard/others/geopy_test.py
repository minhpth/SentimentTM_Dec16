# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 11:23:49 2016

@author: mphan
"""

from __future__ import print_function, division

import os
os.chdir('D:\\pre-alpha\\31_dashboard')

import pandas as pd

#------------------------------------------------------------------------------
# Test geopy functions
#------------------------------------------------------------------------------

# To geolocate a query to an address and coordinates
from geopy.geocoders import Nominatim
geolocator = Nominatim()
location = geolocator.geocode("175 5th Avenue NYC")
print(location.address)
print((location.latitude, location.longitude))
print(location.raw)

# To find the address corresponding to a set of coordinates
from geopy.geocoders import Nominatim
geolocator = Nominatim()
location = geolocator.reverse("52.509669, 13.376294")
print(location.address)
print((location.latitude, location.longitude))
print(location.raw)

#------------------------------------------------------------------------------
# Try with tweets data
#------------------------------------------------------------------------------

file_in = '.\\31_tweets_final.tsv'
tweets_full = pd.read_csv(file_in, sep='\t', encoding='utf-8')

tweets_full.coordinates.sum() / tweets_full.shape[0] * 100 # 7% has geo info

locations = tweets_full.keyword.unique()

# Test
count = 0
count_fail = 0

for loc in locations:
    count += 1
    
    print('=====================================================')
    print('Keyword:', loc)
    
    try:
        location = geolocator.geocode(loc)
        print('Address:', location.address)
        print('Geo:', (location.latitude, location.longitude))
        #print(location.raw)
    except:
        print('CANNOT FIND')
        count_fail += 1
    
    print('=====================================================')
    print()
    
print('Success rate:', (count-count_fail)/count*100)