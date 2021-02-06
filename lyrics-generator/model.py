import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq

class RNN:

    def __init__(self, vocab_size, batch_size, sequence_length, hidden_nodes, cells_size, gradient_clip=5.0, training=True):

        cells = []
        [cells.append(rnn.LSTMCell(hidden_nodes)) for _ in range(cells_size)]
        self.cell = rnn.MultiRNNCell(cells)