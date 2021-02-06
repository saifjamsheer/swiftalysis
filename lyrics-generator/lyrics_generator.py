import numpy as np
from lyrics_provider import LyricsProvider

lyrics = 'datasets/inputs.txt'

BATCH_SIZE = 16
SEQUENCE_LENGTH = 25

def build():
    
    lyrics_provider = LyricsProvider(lyrics, BATCH_SIZE, SEQUENCE_LENGTH)