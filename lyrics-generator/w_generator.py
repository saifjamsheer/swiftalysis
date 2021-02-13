import io

#### Word-Level Text Generator ###

LYRICS_PATH = 'datasets/inputs.txt'
BATCH_SIZE = 128
EPOCHS = 5000
DIVERSITY = 1.0

with io.open(LYRICS_PATH, 'r', encoding='utf8') as f: text = f.read()