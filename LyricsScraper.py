#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:23:58 2020

@author: Saif
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

def fetch(url, params=None, headers=None):
    
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    
    return response

def parse(response):
    
    content = BeautifulSoup(response.text, 'html.parser')
    [h.extract() for h in content('script')]
    lyrics = content.find('', {'class': 'lyrics'})
    
    return lyrics

def clean_lyrics(lyrics):
    
    lyrics = re.sub('[\[].*?[\]]', '', lyrics)
    lyrics = lyrics.strip()
    
    return lyrics

def clean_album_titles(album):
    
    base = re.sub('\([^)]*\)', '', album)
    stripped = base.strip()
    
    return stripped
    

def main():
    
    df = pd.read_csv('songs-with-urls.csv', index_col=0)
    headers={ 'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36' }
    
    l = {}
    lyric_set = []
    
    df['album'] = df['album'].apply(lambda x: clean_album_titles(x))
    
    for item, row in df.iterrows():
        
        lyrics = None
        url = row['url']
        while lyrics is None:
            response = fetch(url, headers=headers)
            lyrics = parse(response)
        
        lyrics = clean_lyrics(lyrics.get_text())
        
        l.update({
                'album': row['album'],
                'title': row['title'],
                'lyrics': lyrics
                })
        
        l_copy = l.copy()
        lyric_set.append(l_copy)
        
    df = pd.DataFrame(lyric_set)
    df.to_csv('taylor-swift-song-lyrics.csv')

if __name__ == "__main__":
    main()
