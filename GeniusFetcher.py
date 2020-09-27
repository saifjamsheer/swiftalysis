#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:42:48 2020

@author: Saif
"""

import requests
import pandas as pd

BASE_URL = "https://api.genius.com"
CLIENT_ACCESS_TOKEN = 'V8sY6tMoPunW5oMksyJTAeSrGtm-v2GAduGckC0qkvAGW2kCMFtUOizSJH231rr-'
ARTIST_NAME = 'Taylor Swift'

headers = {'Authorization' : 'Bearer ' + CLIENT_ACCESS_TOKEN}

def get(path, params=None):
    
    requrl = '/'.join([BASE_URL, path])
    response = requests.get(url=requrl, params=params, headers=headers)
    response.raise_for_status()
    
    return response.json()

def get_artist_id(response):
    
    for hit in response['response']['hits']:
        if hit['result']['primary_artist']['name'] == ARTIST_NAME:
            artist_id = hit['result']['primary_artist']['id']
        break
    
    return artist_id

def get_songs(artist_id):
    
    path = path = '/'.join(['artists', str(artist_id), 'songs'])
    
    page = 1
    cont = True
    songs = []
    
    while cont:
        
        params = {'page': page}
        data = get(path, params=params)
        
        songs_on_page = data['response']['songs']
        
        if songs_on_page:
            
            songs += songs_on_page
            page += 1
            
        else:
            
            cont = False
    
    songs = [song['id'] for song in songs if song['primary_artist']['id'] == artist_id]
   
    return songs

def get_data(songs):
    
    song_data = {}
    
    for i, song_id in enumerate(songs):
        
        path = '/'.join(['songs', str(song_id)])
        data = get(path)['response']['song']
        
        song_data.update({
                i: {'title': data['title'].strip(),
                    'album': data['album']['name'] if data['album'] else 'n/a',
                    'release': data['release_date'] if data['release_date'] else 'n/a',
                    'url': data['url'] if data['url'] else 'n/a'}
                })
    
    return song_data
        
def main():
    
    response = get('search', {'q': ARTIST_NAME})
    artist_id = get_artist_id(response)
    songs = get_songs(artist_id)
    song_data = get_data(songs)
    
    song_list = [info for info in song_data.values()]
    
    df = pd.DataFrame(song_list)
    df.to_csv('taylor-swift-songs.csv')
    

if __name__ == "__main__":
    main()
