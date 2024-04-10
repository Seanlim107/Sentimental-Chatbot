# define the GRU encoder and Decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# encode the 'question'
class EncoderGRU(nn.Module):

    # seq2seq model: embedding_size = hidden_size
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        # init weights
        total_weights = 0
        for n, p in self.state_dict().items():
            total_weights += p.numel()
        print("Encoder has a total of {0:d} parameters".format(total_weights))

    # this assumes input dialogue of dimensions (1 x num_seq x seq_length)
    # get rid of the first dimension
    def forward(self, input_dialogue, lengths, hidden=None):
        
        x = self.embedding(input_dialogue)
        # x = x.reshape(x.shape[1],x.shape[2],x.shape[0])
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # print(x)
        # x = x.view(x.size()[1], x.size()[0], -1)
        output, hidden = self.gru(x, hidden)
        # print(output.data.shape)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # output the whole sequence + last hidden state
        # print(outputs[:, :, :self.hidden_size])
        return outputs, hidden


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

class Attention(nn.Module):
    def __init__(self, input_size, output_size):
        super(Attention, self).__init__()
        self.output_size = output_size
        self.prompt_layer = nn.Linear(in_features=input_size, out_features=output_size)
        self.key_layer = nn.Linear(in_features=input_size, out_features=output_size)
        self.value_layer = nn.Linear(in_features=input_size, out_features=output_size)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out_prompt = F.relu(self.prompt_layer(x))
        out_key = F.relu(self.key_layer(x))
        out_value = F.relu(self.value_layer(x))
        
        out_q_k = torch.div(torch.bmm(out_prompt, out_key.transpose(1, 2)), math.sqrt(self.output_size))
        softmax_q_k = self.softmax(out_q_k)
        out_combine = torch.bmm(softmax_q_k, out_value)
        
        return out_combine
    
class DecoderGRU_Attention(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, num_layers, feature_size):
        # super(DecoderGRU_Attention, self).__init__(vocab_size=vocab_size, hidden_size=hidden_size, embedding_size=embedding_size, num_layers=num_layers, feature_size=feature_size)
        super(DecoderGRU_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.relu = nn.ReLU(inplace=False)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=False)
        self.attention = Attention(hidden_size, feature_size)  # Using your SelfAttention module
        self.out = nn.Linear(hidden_size + feature_size, feature_size)  # Adjust output size based on your requirements
        self.out2 = nn.Linear(feature_size, vocab_size)
        self.attention = Attention(hidden_size, feature_size)
        total_weights = 0
        for n, p in self.state_dict().items():
            total_weights += p.numel()
        print("Decoder has a total of {0:d} parameters".format(total_weights))
        
    def forward(self,x,hidden,encoder_outputs):
        x=self.embedding(x)
        output_x, hidden_x = self.gru(x,hidden)
        
        contexte = self.attention(output_x)
        
        combined_decoder_context = torch.cat((output_x, contexte), dim=-1)
        # print(combined_decoder_context.shape)
        
        output = F.relu(self.out(combined_decoder_context))
        output = self.out2(output)
        
        return output,hidden_x
# test = DecoderGRU_Attention(18000, 64, 128)
# print(test)