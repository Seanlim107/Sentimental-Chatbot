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
import sys
sys.path.insert(0, 'Chatbot')
# os.chdir('/')
from Chatbot.logger import Logger
from Chatbot.utils import parse_arguments, read_settings, load_checkpoint, save_checkpoint
from Chatbot.dataset import inputVar, outputVar, batch2TrainData, indexesFromSentence, zeroPadding, binaryMatrix, trimRareWords, printLines, loadLinesAndConversations, extractSentencePairs, Voc, unicodeToAscii, normalizeString, readVocs, filterPair, filterPairs, loadPrepareData
from Chatbot.models import EncoderRNN, Attn, LuongAttnDecoderRNN, GreedySearchDecoder, SimpleDecoderRNN
from Chatbot.train_test_funcs import maskNLLLoss, train, trainIters, evaluate, evaluateInput

sys.path.insert(0,'Senti_Classifier')
from Senti_Classifier.models import SentimentLSTM
from Senti_Classifier.utils import parse_arguments, read_settings, load_checkpoint, save_checkpoint, load_checkpoint_reg, save_checkpoint_reg
from Senti_Classifier.logger import Logger
from Senti_Classifier.dataset import DialogData
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

USE_CUDA = torch.cuda.is_available()

device = torch.device("cuda" if USE_CUDA else "cpu")

# Everything Chatbot
filepath=os.path.dirname(os.path.realpath(__file__))
chatbot_filepath = os.path.join(filepath, 'Chatbot')
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

corpus_name = "movie-corpus"
corpus = os.path.join(chatbot_filepath, 'data', corpus_name)

# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

configpath =  '/config.yaml'
# print(args)
# Read settings from the YAML file

def main():
    settings = read_settings(filepath+configpath)

    # Access and use the settings as needed
    data_settings_chatbot = settings.get('data_ende', {})
    model_settings_chatbot = settings.get('model_ende', {})
    train_settings_chatbot = settings.get('train_ende', {})
    MAX_LENGTH = data_settings_chatbot['max_seq']  # Maximum sentence length to consider
    # print(MAX_LENGTH)

    # # Load/Assemble voc and pairs
    save_dir = os.path.join(chatbot_filepath, "data", "checkpoints")
    voc = Voc(datafile)

    # Configure models
    ende_mode = 'LSTM' if model_settings_chatbot['lstm'] else 'GRU'
    attn_mode = 'Att' if model_settings_chatbot['use_attention'] else 'NoAtt' 
    attn_method_mode_list = ['dot', 'general', 'concat']
    attn_method_mode = attn_method_mode_list[model_settings_chatbot['attn_method']]
    model_name = f'{ende_mode}_{attn_mode}_{attn_method_mode}'
    attn_mode = ['dot', 'general', 'concat']
    attn_model = 'dot'
    #``attn_model = 'general'``
    #``attn_model = 'concat'``
    hidden_size = model_settings_chatbot['hidden_dim']
    encoder_n_layers = model_settings_chatbot['encoder_num_layers']
    decoder_n_layers = model_settings_chatbot['decoder_num_layers']
    dropout = model_settings_chatbot['dropout']
    batch_size = data_settings_chatbot['batch_size']

    # Set checkpoint to load from; set to None if starting from scratch
    checkpoint_iter = model_settings_chatbot['checkpoint_iter']

    loadFilename = os.path.join(save_dir, model_name,
                        '{}_checkpoint.tar'.format(checkpoint_iter))
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
    else:
        raise Exception('Checkpoint must exist for Chatbot')

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)

    # Initialize encoder & decoder models

    if(model_settings_chatbot['lstm']):
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout, rnn_cell='LSTM') #GRU
        if(model_settings_chatbot['use_attention']):
            decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='LSTM') #GRU
        else:
            decoder = SimpleDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='LSTM')
    else:
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout, rnn_cell='GRU') #GRU

        if(model_settings_chatbot['use_attention']):
            decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='GRU') #GRU
        else:
            decoder = SimpleDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='GRU')

    # encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    # decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)


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
    clip = train_settings_chatbot['clip']
    teacher_forcing_ratio = train_settings_chatbot['teacher_forcing_ratio']
    learning_rate = train_settings_chatbot['lr']
    decoder_learning_ratio = train_settings_chatbot['decoder_learning_ratio']
    n_iteration = train_settings_chatbot['n_iteration']
    print_every = train_settings_chatbot['print_every']
    save_every = train_settings_chatbot['save_every']


    # Ensure dropout layers are in train mode
    # encoder.train()
    # decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        print('Optimizers built!')

    # If you have CUDA, configure CUDA to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    searcher = GreedySearchDecoder(encoder, decoder)


    # Everything Sentimental Classifier

    # Access and use the settings as needed
    data_settings_senti = settings.get('data_senti', {})
    model_settings_senti = settings.get('model_senti', {})
    train_settings_senti = settings.get('train_senti', {})

    voc_init=data_settings_senti['voc_init']
    regression = model_settings_senti['regression']
    if regression:
        senti_model_name = 'Baseline_LSTM_Regressor'
    else:
        senti_model_name = 'Baseline_LSTM_Classifier'
    senti_filedir = os.path.join(filepath, 'Senti_Classifier')
    senti_filename = os.path.join(senti_filedir, f"{senti_model_name}_ckpt_.pth")
    dialogdata = DialogData(voc_init_cache=voc_init, max_seq=data_settings_senti['max_seq'], regression=regression)

    senti_voc = dialogdata.voc_keys


    my_lstm = SentimentLSTM(vocab_senti_size=len(senti_voc), embedding_dim=model_settings_senti['embedding_dim'],
                            output_size=data_settings_senti['num_output'], lstm_hidden_dim=model_settings_senti['lstm_hidden_dim'], hidden_dim=model_settings_senti['hidden_dim'],
                            hidden_dim2=model_settings_senti['hidden_dim2'],n_layers=model_settings_senti['n_layers'],
                            drop_prob=model_settings_senti['drop_prob'], regression=regression)

    my_lstm = my_lstm.to(device)
    optimizer = torch.optim.Adam(list(my_lstm.parameters()), lr = train_settings_senti['learning_rate'])
    max_test_acc = 0
    max_valid_acc = 0
    min_test_loss = -1
    min_valid_loss = -1
    ckpt_epoch = 0


    if os.path.exists(senti_filename):
        if(regression):
            ckpt_epoch, min_test_loss, min_valid_loss, senti_voc = load_checkpoint_reg(my_lstm, optimizer, min_test_loss, min_valid_loss, senti_filename, senti_voc)
        else:
            ckpt_epoch, max_test_acc, max_valid_acc, senti_voc = load_checkpoint(my_lstm, optimizer, max_test_acc, max_valid_acc, senti_filename, senti_voc)
        print(f'Checkpoint detected')
    else:
        raise Exception('No file detected')

    evaluate_chatbot_input(encoder, decoder, searcher, voc, dialogdata, my_lstm, regression)

    
def evaluate_chatbot_input(encoder, decoder, searcher, voc, dialogdata, my_lstm, regression):
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('You > ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            input_sentence_senti = dialogdata.preprocess_tokens(input_sentence)
            input_sentence_senti = torch.tensor(input_sentence_senti, dtype=torch.int).unsqueeze(0)
            input_ypred_senti = my_lstm(input_sentence_senti)
            if(not regression):
                input_ypred_senti = torch.argmax(input_ypred_senti, axis=1, keepdims=False) -1
            print('You Sentiment: ', input_ypred_senti)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            output_sentence = ' '.join(output_words)
            print('Bot:', output_sentence)
            output_sentence = normalizeString(output_sentence)
            output_sentence_senti = dialogdata.preprocess_tokens(output_sentence)
            output_sentence_senti = torch.tensor(output_sentence_senti, dtype=torch.int).unsqueeze(0)
            output_ypred_senti = my_lstm(output_sentence_senti)
            if(not regression):
                output_ypred_senti = torch.argmax(output_ypred_senti, axis=1, keepdims=False) -1
            print('Bot Sentiment:', output_ypred_senti)
            

        except KeyError:
            print("Error: Encountered unknown word.")
            
if __name__ == '__main__':
    main()
# encoder.eval()
# decoder.eval()

# # Initialize search module
# searcher = GreedySearchDecoder(encoder, decoder)


# evaluateInput(encoder, decoder, searcher, voc)