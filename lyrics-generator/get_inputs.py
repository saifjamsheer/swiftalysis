import pandas as pd 

df = pd.read_csv('datasets/lyrics.csv', index_col=0)
lyrics = ''

for item, row in df.iterrows():

    lyrics += row['lyrics'] + '\n\n'

f = open('datasets/inputs.txt', 'w')
f.write(lyrics)
f.close()