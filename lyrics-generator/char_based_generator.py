import numpy as np
import io
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import Adam

SEQUENCE_LENGTH = 21
SEQUENCE_STEP = 3
LYRICS_PATH = 'datasets/inputs.txt'
BATCH_SIZE = 64
EPOCHS = 100

# Extract unique characters from lyrics
with io.open(LYRICS_PATH, 'r', encoding='utf8') as f: text = f.read()
chars = sorted(list(set(text)))

# Sequences are inputs into the network and next_chars will be used as labels for training
sequences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, SEQUENCE_STEP):
    sequences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])

vocab, indices = dict((c, i) for i, c in enumerate(chars)), dict((i, c) for i, c in enumerate(chars))

# Vectorize the characters and strings
X = np.zeros((len(sequences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, char in enumerate(sentence):
        X[i, t, vocab[char]] = 1
    y[i, vocab[next_chars[i]]] = 1

model = Sequential()
model.add(LSTM(64, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

generated = ''
    
sentence = "Today was a fairytale"

generated += sentence

print('Generating with seed: "' + sentence + '"')
# sys.stdout.write(generated)

for i in range(1500):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, vocab[char]] = 1.

    predictions = model.predict(x, verbose=0)[0]

    preds = np.asarray(predictions).astype('float64')
    preds = np.log(preds) / 1.0
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    next_index = np.argmax(probs)

    next_char = indices[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    # sys.stdout.write(next_char)
    # sys.stdout.flush()
print(generated)