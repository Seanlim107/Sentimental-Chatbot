import numpy as np
import os, sys, re
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_senti_size,
                 embedding_dim,
                 output_size = 1,
                 lstm_hidden_dim = 256,
                 hidden_dim = 512,
                 hidden_dim2 = 256,
                 n_layers = 3,
                 drop_prob = 0.5):
        
        super(SentimentLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        
        self.embedding = nn.Embedding(vocab_senti_size, self.embedding_dim, sparse=False)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, n_layers, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(self.drop_prob)
        
        self.fc1 = nn.Linear(2*self.lstm_hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, output_size)
        
        #Initialize weights and biases
        for x in self.named_parameters():
            if('weights' in x[0]):
                torch.nn.init.xavier_uniform(x[1])
            else:
                x[1].data.fill_(0.01)
        
    def forward(self, x):
        out_embed = self.embedding(x)
        
        out_lstm, (h_t,o_t) = self.lstm(out_embed)
        
        out_lstm = out_lstm[:,-1,:]
        
        output = self.fc1(out_lstm)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        
        return output
    