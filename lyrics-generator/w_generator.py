import io
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import Adam

#### WORD-LEVEL LYRICS GENERATOR ###

LYRICS_PATH = 'datasets/inputs.txt'
BATCH_SIZE = 128
EPOCHS = 1
DIVERSITY = 1.0
MIN_FREQ = 10
SEQUENCE_LENGTH = 5
SEQUENCE_STEP = 1

with io.open(LYRICS_PATH, 'r', encoding='utf8') as f: text = f.read()

text = text.replace('\n', ' \n ')
text = text.replace('?', '')
text = text.replace('!', '')
text = text.replace('(', '')
text = text.replace(')', '')
text = text.replace('.', '')
corpus = [w for w in text.split(' ') if w.strip() != '' or w == '\n'
        and (w[0] not in ["(", "["] and w[-1] not in [")", "]"])]

while "" in corpus:
    corpus.remove("")

word_freq = {}
for word in corpus:
    word_freq[word] = word_freq.get(word, 0) + 1

ignored_words = set()
for k, v in word_freq.items():
    if word_freq[k] < MIN_FREQ:
        ignored_words.add(k)

vocab = set(corpus)
# vocab = sorted(set(vocab) - ignored_words)

word_ix, ix_word = dict((c, i) for i, c in enumerate(vocab)), dict((i, c) for i, c in enumerate(vocab))

sequences = []
next_words = []

for i in range(0, len(corpus) - SEQUENCE_LENGTH, SEQUENCE_STEP):
        sequences.append(corpus[i: i + SEQUENCE_LENGTH])
        next_words.append(corpus[i + SEQUENCE_LENGTH])

# Vectorize the words
X = np.zeros((len(sequences), SEQUENCE_LENGTH, len(vocab)), dtype=np.bool)
y = np.zeros((len(sequences), len(vocab)), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, word_ix[word]] = 1
    y[i, word_ix[next_words[i]]] = 1

# split_count = int(0.8 * len(sequences))
# sentences_test = sequences[split_count:]
# next_words_test = next_words[split_count:]
# sentences = sequences[:split_count]
# next_words = next_words[:split_count]

# print(sentences)
model = Sequential()
model.add(LSTM(64, input_shape=(SEQUENCE_LENGTH, len(vocab))))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

generated = []
seed_index = int(np.random.randint(len(sequences)))
seed = sequences[seed_index]
sentence = seed

[generated.append(word) for word in sentence]
print(generated)

for i in range(50):
    x = np.zeros((1, SEQUENCE_LENGTH, len(vocab)))
    print(sentence)
    for t, word in enumerate(sentence):
        print(x[0, t, word_ix[word]])
        x[0, t, word_ix[word]] = 1.

    predictions = model.predict(x, verbose=0)[0]

    preds = np.asarray(predictions).astype('float64')
    preds = np.log(preds) / DIVERSITY
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    next_index = np.argmax(probs)

    next_word = ix_word[next_index]

    generated.append(next_word)
    sentence = sentence[1:].append(next_word)

print(generated)