# -*- coding: utf-8 -*-
from __future__ import print_function, division

#------------------------------------------------------------------------------
# TRIPADVISOR WEB SCRAPPING
#------------------------------------------------------------------------------

# Functions:
# [1] Read the location (e.g. city) in the settings.txt file
# [2] Go to TripAdvisor, download all tourism locations related to that city
# [3] Process and save tourism locations to file for next steps

# Version: 2.0
# Last edited: 20 Dec 2016
# Edited by: Minh PHAN

#------------------------------------------------------------------------------
# Global variables and settings
#------------------------------------------------------------------------------

# Seting working directory
import os
os.chdir('D:\\SentimentTM\\11_webscrapping')
        
#------------------------------------------------------------------------------
# Initiating
#------------------------------------------------------------------------------

# Essential packages
import pandas as pd
import re
from time import time, sleep

# Other functional packages
from bs4 import BeautifulSoup # Web scrapping
import urllib2
from urllib2 import urlopen # Download link

#------------------------------------------------------------------------------
# Function to do web scrapping TripAdvisor.com
#------------------------------------------------------------------------------

def urlopen_wrapper(url):
    num_retry = 3 # Retry 3 times
    count_retry = 0
    delay = 10 # secs
    while  count_retry < num_retry:
        try:
            page = urlopen(url, timeout=30)
            break # If no error, exit while loop
            
        except urllib2.URLError as e:
            print('Page load error:', e)
            print('Waiting and retrying...')
            count_retry += 1
            sleep(delay)
    
    return page
   
def scrapping_tripadvisor(begin_url):
    
    result = []
    count = 0
    page_count = 0
    
    while True: # Loop until break
    
        try:
            page = urlopen_wrapper(begin_url)
        except:
            print('Page load failed. Webscrapping stopped.')
            break

        soup = BeautifulSoup(page, "html.parser")
        page_count += 1
        print('Scrapping page #' + str(page_count) + ': \n')
        
        # Extract all location text
        for dest in soup.findAll(attrs={"class": "property_title"}):
            count += 1
            text = re.match(r'\n.*\n', dest.text).group().strip()
            print(count, ':', text)
            result.append(text)
        print()
            
        # Check the NEXT button, if it was blocked, end loop
        if soup.findAll(attrs={"class": "nav next disabled"}):
            print()
            break
        
        # Else, jump to next page
        next_button = soup.findAll(attrs={"class": "nav next rndBtn ui_button primary taLnk"})
        next_url = 'https://www.tripadvisor.com' + str(next_button[0].get('href'))
        begin_url = next_url
    
    print('Total pages were scrapped:', page_count)
    print('Total locations:', count)
    
    return result

#------------------------------------------------------------------------------
# MAIN: Web scrapping + save to file
#------------------------------------------------------------------------------

# Select location and find its link from TripAdvisor.com
city = 'London'
city_TripAdvisor_url = 'https://www.tripadvisor.com/Attractions-g186338-Activities-London_England.html'

# Do web scrapping for a location or city
print('City selected:', city)

# Go to TripAdvisor, looking for "Things to Do"
# Then select a location or city, pick the url of the result page
begin_url = city_TripAdvisor_url

# Run webscrapping
t0 = time()
locations = scrapping_tripadvisor(begin_url)
print('Running time:', time()-t0)

# Create clean location keywords
df = pd.DataFrame({'city':city, 'location':locations})
df['location_keyword'] = df['location'].str.replace(r'(\([0-9]+\))', ' ')
df['location_keyword'] = df['location_keyword'].str.replace(r'[^a-zA-Z0-9]', ' ')
df['location_keyword'] = df['location_keyword'].str.replace(r' +', ' ')
df['location_keyword'] = df['location_keyword'].str.strip().str.lower()

# Save to file
file_out = '.\\output\\11_tripadvisor_tourism_locations.tsv'
df.to_csv(file_out, encoding='utf-8', index=False, sep='\t')

#------------------------------------------------------------------------------