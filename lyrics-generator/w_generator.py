import io
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, GRU
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from sklearn.model_selection import train_test_split

"""""""""""""""""""""""""""
WORD-LEVEL LYRICS GENERATOR
"""""""""""""""""""""""""""

LYRICS_PATH = 'datasets/inputs.txt'
OUTPUT_PATH = 'results/w_outputs.txt'
BATCH_SIZE = 128
EPOCHS = 15
DIVERSITY = 1.0
MIN_FREQ = 10
SEQUENCE_LENGTH = 5

def load_lyrics(path):
    """
    Load lyrics from input path
    """
    with io.open(LYRICS_PATH, 'r', encoding='utf8') as f: lyrics = f.read()
    return lyrics

def clean_text(text):
    """
    Replaces punctuation and special characters
    """
    text = text.replace('\n', ' \n ')
    for ch in ['?', '!', '(', ')', '.', ',', '"']:
        if ch in text:
            text = text.replace(ch, '')
    return text

def create_vocab(text):
    """
    Creates the a dictionary of all words in a string
    """
    word_freq = {}
    for word in corpus:
        word_freq[word] = word_freq.get(word, 0) + 1
    ignored_words = set()
    for k, _ in word_freq.items():
        if word_freq[k] < MIN_FREQ:
            ignored_words.add(k)
    vocab = set(corpus)
    return vocab, ignored_words

def create_sequences(text, max_length):
    """
    Create inputs and labels for training
    """
    sequences = []
    next_words = []
    for i in range(0, len(text) - max_length):
            sequences.append(text[i: i + max_length])
            next_words.append(text[i + max_length])
    return sequences, next_words

def get_index_mappings(words):
    """
    Create mappings from word to index and index to word
    """
    return {c: i for i, c in enumerate(words)}, {i: c for i, c in enumerate(words)}

def vectorize_words(sequences, next_words, max_length, vocab):
    """
    Convert words and sentences into numerical representations
    """
    X = np.zeros((len(sequences), max_length, len(vocab)), dtype=np.bool)
    y = np.zeros((len(sequences), len(vocab)), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for t, word in enumerate(sequence):
            X[i, t, word_ix[word]] = 1
        y[i, word_ix[next_words[i]]] = 1
    return X, y

def build_LSTM(max_length, vocab, dropout=0.2):
    """
    Build the LSTM network structure and compile the model
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_length, len(vocab))))
    model.add(Dropout(dropout))
    model.add(LSTM(128))
    model.add(Dropout(dropout))
    model.add(Dense(len(vocab)))
    model.add(Activation('softmax'))
    optimizer=Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model

def build_GRU(max_length, vocab, dropout=0.2):
    """
    Build the GRU network structure and compile the model
    """
    model = Sequential()
    model.add(GRU(128, input_shape=(max_length, len(vocab))))
    model.add(Dropout(dropout))
    model.add(GRU(128))
    model.add(Dropout(dropout))
    model.add(Dense(len(vocab)))
    model.add(Activation('softmax'))
    optimizer=Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model

def generate_seed(sequences):
    """
    Generate initial input for sampling
    """
    seed_index = int(np.random.randint(len(sequences)))
    seed = sequences[seed_index]
    return seed

def sample(preds, temperature=1.0):
    """
    Samples a word index through probabilistic means
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

    sentence = generate_seed(sequences)
    output_file.write('Generating with seed: {}\n\n'.format(' '.join(sentence)))
    output_file.write('----Generated lyrics:\n')
    output_file.write(' '.join(sentence))

    for _ in range(50):
        x = np.zeros((1, SEQUENCE_LENGTH, len(vocab)))
        for t, word in enumerate(sentence):
            x[0, t, word_ix[word]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, DIVERSITY)

        next_word = ix_word[next_index]
        prev_word = sentence[-1]
        sentence = sentence[1:] + [next_word]

        if prev_word == '\n':
            output_file.write(next_word)
        else:
            output_file.write(' ' + next_word)

    output_file.write('\n')
    output_file.write('-'*50 + '\n')

text = load_lyrics(LYRICS_PATH)
text = clean_text(text)

corpus = [w for w in text.split(' ') if w.strip() != '' or w == '\n' and (w[0] not in ["(", "["] and w[-1] not in [")", "]"])]
while "" in corpus: corpus.remove("")

vocab, ignored_words = create_vocab(corpus)
# vocab = sorted(set(vocab) - ignored_words)

sequences, next_words = create_sequences(corpus, SEQUENCE_LENGTH)
word_ix, ix_word = get_index_mappings(vocab)

X, y = vectorize_words(sequences, next_words, SEQUENCE_LENGTH, vocab)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# model = build_LSTM(SEQUENCE_LENGTH, vocab)
model = build_GRU(SEQUENCE_LENGTH, vocab)

output_file = open(OUTPUT_PATH, 'w', encoding='utf8')

write_callback = LambdaCallback(on_epoch_end=on_epoch_end)
callbacks_list = [write_callback]

model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list, validation_data=(X_val, y_val), validation_batch_size=BATCH_SIZE)
# model.save("models/word_level_model.h5")

output_file.close()
