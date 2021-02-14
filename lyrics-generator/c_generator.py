import numpy as np
import io
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback

"""""""""""""""""""""""""""""""""
CHARACTER-LEVEL LYRICS GENERATOR
"""""""""""""""""""""""""""""""""

def load_lyrics(path):
    """
    Load lyrics from input path
    """
    with io.open(LYRICS_PATH, 'r', encoding='utf8') as f: lyrics = f.read()
    return lyrics

def create_sequences(text, length, step):
    """
    Create inputs and labels for training
    """
    sequences = []
    next_chars = []
    for i in range(0, len(text) - length, step):
        sequences.append(text[i: i + length])
        next_chars.append(text[i + length])

    return sequences, next_chars

def get_index_mappings(chars):
    """
    Create mappings from character to index and index to character
    """
    return {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}

def vectorize_lyrics(chars, char_ix, sequences, next_chars, length):
    """
    Convert characters and sequences into numerical representations
    """
    X = np.zeros((len(sequences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, char in enumerate(sentence):
            X[i, t, char_ix[char]] = 1
        y[i, char_ix[next_chars[i]]] = 1
    return X, y

def build(length, chars, n_layers=1, dropout=0.2):
    """
    Build the network structure and compile the model
    """
    model = Sequential()
    for _ in range(n_layers):
        model.add(LSTM(64, input_shape=(SEQUENCE_LENGTH, len(chars))))
    model.add(Dropout(dropout))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer=Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model

def sample(preds, temperature=1.0):
    """
    Samples a character index based through probabilistic means
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    """
    Callback after each epoch to write the generated lyrics to the output file
    """
    
    output_file.write('Generating lyrics after epoch: {}\n'.format(epoch))

    sentence = seed
    output_file.write('Generating with seed: {}\n\n'.format(sentence))
    generated = ''

    for _ in range(500):
        x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_ix[char]] = 1.0

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, DIVERSITY)

        next_char = ix_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    output_file.write('Generated lyrics:\n')
    output_file.write(generated)
    output_file.write('\n')
    output_file.write('-'*50 + '\n')

# Defining global variables
SEQUENCE_LENGTH = 21
SEQUENCE_STEP = 3
LYRICS_PATH = 'datasets/inputs.txt'
OUTPUT_PATH = 'datasets/outputs.txt'
BATCH_SIZE = 128
EPOCHS = 2
DIVERSITY = 1.0

# Load lyrics and extract unique characters
text = load_lyrics(LYRICS_PATH)
chars = sorted(list(set(text)))

# Create inputs (sequences) and labels (next_chars) for model training
sequences, next_chars = create_sequences(text, SEQUENCE_LENGTH, SEQUENCE_STEP)
char_ix, ix_char = get_index_mappings(chars)

# Convert text into numerical representations
X, y = vectorize_lyrics(chars, char_ix, sequences, next_chars, SEQUENCE_LENGTH)

# Build the LSTM model
model = build(SEQUENCE_LENGTH, chars, n_layers=1, dropout=0.0)

# Open file to output generated text
output_file = open(OUTPUT_PATH, 'w')

# Define callbacks for model training
write_callback = LambdaCallback(on_epoch_end=on_epoch_end)
callbacks_list = [write_callback]

# Define seed for text generation
seed = "Today was a fairytale"

# Model training and text generation
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list)

# Close output file
output_file.close()