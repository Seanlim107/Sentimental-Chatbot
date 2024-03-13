import numpy as np
import os, sys, re
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentGRU(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, output_size=1, hidden_dim_gru=128,
                 hidden_dim=128, n_layers=3, dropout_prob=0.5):

        super(SentimentGRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim_gru = hidden_dim_gru
        self.out_batch_s = None
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=False)  # 1-hot encoding to embedding space
        # embedding_dim: input_size,
        # hidden_dim: number of hidden neurons
        # n_layers: stacked LSTM for n>1
        # if batch_first is false, h_t and o_t have dims 1 x seq_length x num_features
        # otherwise it's seq_length x 1 x num_features
        # note sequence length is not a hyperparameter
        self.gru = nn.GRU(embedding_dim, hidden_dim_gru, n_layers, batch_first=True, bidirectional=False)
        # linear and sigmoid layers
        self.fc1 = nn.Linear(hidden_dim_gru, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        # self.dropout = nn.Dropout(dropout_prob)
        ####################################################
        # init weights
        total_weights = 0
        for x in self.named_parameters():
            # print (x[0])
            if 'weight' in x[0]:
                torch.nn.init.kaiming_normal_(x[1])
            elif 'bias' in x[0]:
                x[1].data.fill_(0.01)
            if 'gru' in x[0]:
                total_weights += x[1].numel()
        ####################################################

    # forward method takes tensor size batch x
    # input size batch x num of tokens in the review
    def forward(self, x):
        out_batch = self.embedding(x.long())  # Embedding takes a sequence of integer(long) inputs
        self.out_batch_s = out_batch.size()
        # gru outputs
        out_gru, h_t = self.gru(out_batch)
        out_fc1 = self.fc1(out_gru[:, -1, :])  # some filtering
        # out_fc1  = self.dropout(out_fc1)
        out_fc2 = self.fc2(out_fc1)

        return out_fc2

    def ret_current_idx(self):
        return self.out_batch_s

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
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim, self.n_layers, batch_first=True, bidirectional=False)
        
        self.dropout = nn.Dropout(self.drop_prob)
        
        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, output_size)
        
        #Initialize weights and biases
        for x in self.named_parameters():
            if 'weight' in x[0]:
                torch.nn.init.xavier_uniform_(x[1])
            elif 'bias' in x[0]:
                x[1].data.fill_(0.01)
        
    def forward(self, x):
        # print(x)
        out_embed = self.embedding(x.long())
        print(out_embed.shape)
        # print(out_embed)
        out_lstm, (h_t,o_t) = self.lstm(out_embed)
        # print(h_t)
        # print(out_lstm)
        out_lstm = out_lstm[:,-1,:]
        # print(out_lstm)
        
        output = self.fc1(out_lstm)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        
        return output
    