import io
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback

"""""""""""""""""""""""""""
WORD-LEVEL LYRICS GENERATOR
"""""""""""""""""""""""""""

LYRICS_PATH = 'datasets/inputs.txt'
OUTPUT_PATH = 'datasets/outputs.txt'
BATCH_SIZE = 128
EPOCHS = 1
DIVERSITY = 1.0
MIN_FREQ = 10
SEQUENCE_LENGTH = 5
SEQUENCE_STEP = 1

def load_lyrics(path):
    """
    Load lyrics from input path
    """
    with io.open(LYRICS_PATH, 'r', encoding='utf8') as f: lyrics = f.read()
    return lyrics

def clean_text(text):
    """
    Explain
    """
    text = text.replace('\n', ' \n ')
    for ch in ['?', '!', '(', ')', '.', ',', '"']:
        if ch in text:
            text = text.replace(ch, '')
    return text

def create_vocab(text):
    """
    Explain
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
    Explain
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
    Explain
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
    Explain
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_length, len(vocab))))
    model.add(Dropout(dropout))
    model.add(Dense(len(vocab)))
    model.add(Activation('softmax'))
    optimizer=Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model

def build_GRU():
    """
    Explain
    """
    return 1

def generate_seed():
    """
    Explain
    """
    return 1

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
    return 1 

text = load_lyrics(LYRICS_PATH)
text = clean_text(text)

corpus = [w for w in text.split(' ') if w.strip() != '' or w == '\n' and (w[0] not in ["(", "["] and w[-1] not in [")", "]"])]
while "" in corpus: corpus.remove("")

vocab, ignored_words = create_vocab(corpus)
# vocab = sorted(set(vocab) - ignored_words)

sequences, next_words = create_sequences(corpus, SEQUENCE_LENGTH)
word_ix, ix_word = get_index_mappings(vocab)

X, y = vectorize_words(sequences, next_words, SEQUENCE_LENGTH, vocab)

model = build_LSTM(SEQUENCE_LENGTH, vocab)

output_file = open(OUTPUT_PATH, 'w', encoding='utf8')

write_callback = LambdaCallback(on_epoch_end=on_epoch_end)
callbacks_list = [write_callback]

# model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

seed_index = int(np.random.randint(len(sequences)))
seed = sequences[seed_index]
sentence = seed
print(sentence)


for i in range(50):
    x = np.zeros((1, SEQUENCE_LENGTH, len(vocab)))
    for t, word in enumerate(sentence):
        x[0, t, word_ix[word]] = 1.

    predictions = model.predict(x, verbose=0)[0]

    preds = np.asarray(predictions).astype('float64')
    preds = np.log(preds) / DIVERSITY
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    next_index = np.argmax(probs)

    next_word = ix_word[next_index]

    sentence = sentence[1:] + [next_word]
    print(sentence)
