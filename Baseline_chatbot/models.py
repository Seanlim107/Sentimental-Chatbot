# define the GRU encoder and Decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# encode the 'question'
class EncoderGRU(nn.Module):

    # seq2seq model: embedding_size = hidden_size
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1):
        super(EncoderGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=False)
        # init weights
        total_weights = 0
        for n, p in self.state_dict().items():
            total_weights += p.numel()
        print("Encoder has a total of {0:d} parameters".format(total_weights))

    # this assumes input dialogue of dimensions (1 x num_seq x seq_length)
    # get rid of the first dimension
    def forward(self, input_dialogue):
        x = self.embedding(input_dialogue)
        x = x.view(x.size()[1], x.size()[0], -1)
        output, hidden = self.gru(x)
        # output the whole sequence + last hidden state
        return output, hidden


# decode the 'reply'
class DecoderGRU(nn.Module):
    # embedding_size:
    # hidden_size: number of features in a hidden layer
    # output_size: number of words in the vocabulary
    def __init__(
        self, vocab_size, hidden_size, embedding_size, num_layers=1, feature_size=128
    ):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.relu = nn.ReLU(inplace=False)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=False)
        self.out = nn.Linear(hidden_size, feature_size)
        self.out2 = nn.Linear(feature_size, vocab_size)
        total_weights = 0
        for n, p in self.state_dict().items():
            total_weights += p.numel()
        print("Decoder has a total of {0:d} parameters".format(total_weights))
        ####################################################

    # input: previous token
    # hidden: initialized as the context vector,
    #   output of the last hidden state of the encoder
    def forward(self, x, hidden):
        x = self.embedding(x)
        o, h = self.gru(x, hidden)
        o = F.relu(self.out(o))
        o = self.out2(o)
        return o, h
