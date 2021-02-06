import collections
import numpy as np

class LyricsProvider():

    def __init__(self, lyrics, batch_size, sequence_length):

        f = open(lyrics, 'r')
        data = f.read()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.pointer = 0
        count_pairs = sorted(collections.Counter(data).items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.array(list(map(self.vocab.get, data)))
        self.batches_size = int(self.tensor.size / (self.batch_size * self.sequence_length))
        self.tensor = self.tensor[:self.batches_size * self.batch_size * self.sequence_length]
        inputs = self.tensor
        targets = np.copy(self.tensor)
        targets[:-1] = inputs[1:]
        targets[-1] = inputs[0]
        self.input_batches = np.split(inputs.reshape(self.batch_size, -1), self.batches_size, 1)
        self.target_batches = np.split(targets.reshape(self.batch_size, -1), self.batches_size, 1)

    def next_batch(self):
        inputs = self.input_batches[self.pointer]
        targets = self.target_batches[self.pointer]
        self.pointer += 1
        return inputs, targets

    def reset_pointer(self):
        self.pointer = 0