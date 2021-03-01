import requests
import pandas as pd
import re

AUTH_URL = 'https://accounts.spotify.com/api/token'
CLIENT_ID = 'ab8f4a8357cf452b97415503d5d9b3a1'
CLIENT_SECRET = '1965b8d3bcbe44faa94cf75ab3444dc6'

auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})

auth_response_data = auth_response.json()

CLIENT_ACCESS_TOKEN = auth_response_data['access_token']
BASE_URL = 'https://api.spotify.com/v1'
ARTIST_ID = '06HL4z0CvFAxyc27GXpf02'
headers = {'Authorization' : 'Bearer {token}'.format(token=CLIENT_ACCESS_TOKEN)}

def get(path, params=None):
    
    requrl = '/'.join([BASE_URL, path])
    response = requests.get(url=requrl, params=params, headers=headers)
    response.raise_for_status()
    return response.json()

album_ids = ['1pzvBxYgT6OVwJLtHkrdQK', 
                '1NAmidJlEaVgA3MpcPFYGq', 
                '6DEjYFkNZh67HP7R9PSZvv',
                '1yGbNOtRIgdIiGHOEBaZWf',
                '1KVKqWeRuXsJDLTW0VuD29',
                '6Ar2o9KCqcyYF9J0aQP3au',
                '5EpMjweRD573ASl7uNiHym',
                '2gP2LMVcIFgVczSJqn340t',
                '5eyZZoQEFQWRHkV2xgAeBw']

f = {}
track_list = []
values = []

for album_id in album_ids:
    
    album_name = get('albums/{album}'.format(album=album_id))['name']
    data = get('albums/{album}/tracks'.format(album=album_id))
    tracks = data['items']
    
    for track in tracks:
        
        track_name = track['name'].split('-', 1)[0].strip()
        track_name = re.split('\([^O]', track_name)[0].strip()
        
        if track_name.lower() in track_list:
            continue
        
        f.update({
        'title': track_name,
        'album': album_name
        })

        f_copy = f.copy()
        
        track_list.append(track_name.lower())
        values.append(f_copy)
        
df = pd.DataFrame(values)
df.to_csv('taylor-swift-tracks.csv')