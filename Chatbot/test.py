import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
import os
os.chdir('/')
from logger import Logger
from utils import parse_arguments, read_settings, load_checkpoint, save_checkpoint
from dataset import inputVar, outputVar, batch2TrainData, indexesFromSentence, zeroPadding, binaryMatrix, trimRareWords, printLines, loadLinesAndConversations, extractSentencePairs, Voc, unicodeToAscii, normalizeString, readVocs, filterPair, filterPairs, loadPrepareData
from models import EncoderRNN, Attn, LuongAttnDecoderRNN, GreedySearchDecoder, SimpleDecoderRNN
from train_test_funcs import maskNLLLoss, train, trainIters, evaluate, evaluateInput
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

filepath=os.path.dirname(os.path.realpath(__file__))
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

corpus_name = "movie-corpus"
corpus = os.path.join(filepath, 'data',corpus_name)

# Define path to new file
datafile = os.path.join(filepath, corpus, "formatted_movie_lines.txt")

args = parse_arguments()
# print(args)
# Read settings from the YAML file

settings = read_settings(filepath+args.config)

# Access and use the settings as needed
data_settings = settings.get('data_ende', {})
model_settings = settings.get('model_ende', {})
train_settings = settings.get('train_ende', {})
MAX_LENGTH = data_settings['max_seq']  # Maximum sentence length to consider
print(MAX_LENGTH)

# # Load/Assemble voc and pairs
save_dir = os.path.join(filepath, "data", "checkpoints")
voc = Voc(datafile)

# Configure models
ende_mode = 'LSTM' if model_settings['lstm'] else 'GRU'
attn_mode = 'Att' if model_settings['use_attention'] else 'NoAtt' 
attn_method_mode_list = ['dot', 'general', 'concat']
attn_method_mode = attn_method_mode_list[model_settings['attn_method']]
model_name = f'{ende_mode}_{attn_mode}_{attn_method_mode}'
attn_model = 'dot'
#``attn_model = 'general'``
#``attn_model = 'concat'``
hidden_size = model_settings['hidden_dim']
encoder_n_layers = model_settings['encoder_num_layers']
decoder_n_layers = model_settings['decoder_num_layers']
dropout = model_settings['dropout']
batch_size = data_settings['batch_size']

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = model_settings['checkpoint_iter']

ende_mode = 'LSTM' if model_settings['lstm'] else 'GRU'
attn_mode = 'Att' if model_settings['use_attention'] else 'NoAtt' 
attn_method_mode_list = ['dot', 'general', 'concat']
attn_method_mode = attn_method_mode_list[model_settings['attn_method']]
model_name = f'{ende_mode}_{attn_mode}_{attn_method_mode}'
directory = os.path.join(save_dir , model_name)
attn_mode = ['dot', 'general', 'concat']
attn_model = 'dot'
#``attn_model = 'general'``
#``attn_model = 'concat'``
# filename_sentimental = f"{model_name_chatbot}_{model_name}_ckpt_.pth"
# iteration = model_settings['n_iteration']
# chatbot_filename=os.path.join('{}_{}.tar'.format(iteration, 'checkpoint'))
# checkpoint_filename = '{}_checkpoint.tar'.format(checkpoint_iter)

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = model_settings['checkpoint_iter']

loadFilename = os.path.join(save_dir, model_name,
                    '{}_{}.tar'.format(checkpoint_iter, 'checkpoint'))
    
print(f"Loading model from: {loadFilename}")
# Load model if a ``loadFilename`` is provided
if os.path.exists(loadFilename):
    print('Checkpoint Detected')
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename, map_location=device)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
    print(voc.num_words)
else:
    raise Exception('No Checkpoint detected')
    print(f'Checkpoint {loadFilename} not detected, starting from scratch')
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir, data_settings['max_seq'])

# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)

# Initialize encoder & decoder models
# encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout, rnn_cell='LSTM')
# decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='LSTM')

if(model_settings['lstm']):
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout, rnn_cell='LSTM') #GRU
    if(model_settings['use_attention']):
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='LSTM') #GRU
    else:
        decoder = SimpleDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='LSTM')
else:
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout, rnn_cell='GRU') #GRU

    if(model_settings['use_attention']):
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='GRU') #GRU
    else:
        decoder = SimpleDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='GRU')


if os.path.exists(loadFilename):
    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    print('Checkpoint Loaded')
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = train_settings['clip']
teacher_forcing_ratio = train_settings['teacher_forcing_ratio']
learning_rate = train_settings['lr']
decoder_learning_ratio = train_settings['decoder_learning_ratio']
n_iteration = train_settings['n_iteration']
print_every = train_settings['print_every']
save_every = train_settings['save_every']


# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have CUDA, configure CUDA to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

# Run training iterations
# print("Starting Training!")
# trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
#            embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
#            print_every, save_every, clip, MAX_LENGTH, teacher_forcing_ratio, decoder_n_layers, corpus_name, loadFilename)

encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)


evaluateInput(encoder, decoder, searcher, voc)