import numpy as np
import os, sys, re
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_senti_size,
                 embedding_dim,
                 output_size = 3,
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
        
        self.embedding = nn.Embedding(vocab_senti_size, self.embedding_dim)
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
        h0 = torch.zeros(self.n_layers, x.size(0), self.lstm_hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.lstm_hidden_dim).to(x.device)
        xlong=x.long()
        out_embed = self.embedding(xlong)
        out_lstm, _ = self.lstm(out_embed, (h0,c0))
        out_lstm = out_lstm[:,-1,:]
        out_lstm = self.dropout(out_lstm)
        output1 = self.fc1(out_lstm)
        
        output1 = F.relu(output1)
        output1 = self.dropout(output1)
        output2 = self.fc2(output1)
        
        output2 = F.relu(output2)
        output2 = self.dropout(output2)
        output3 = self.fc3(output2)
        
        
        return output3
    