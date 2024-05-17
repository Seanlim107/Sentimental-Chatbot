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
from tqdm.auto import tqdm
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

    # _______________________Everything Chatbot____________________________
    # Access and use the settings as needed
    data_settings_chatbot = settings.get('data_ende', {})
    model_settings_chatbot = settings.get('model_ende', {})
    train_settings_chatbot = settings.get('train_ende', {})
    MAX_LENGTH = data_settings_chatbot['max_seq']  # Maximum sentence length to consider
    # print(MAX_LENGTH)

    # # Load/Assemble voc and pairs
    save_dir = os.path.join(filepath, "SC_data", "checkpoints_improved")
    # voc = Voc(datafile)
    save_dir_chatbot = os.path.join("Chatbot", "data", "movie-corpus")
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir_chatbot, data_settings_chatbot['max_seq'])

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
        print('No Checkpoint, Starting from scratch')

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding = embedding.to(device)
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
    encoder.train()
    decoder.train()
    

    # Initialize optimizers
    # print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if os.path.exists(loadFilename):
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        print('Optimizers Loaded!')

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
    # _______________________Everything Chatbot____________________________

    # ________________Everything Sentimental Classifier____________________

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
    # my_lstm.eval()
    optimizer = torch.optim.Adam(list(my_lstm.parameters()), lr = train_settings_senti['learning_rate'])
    max_test_acc = 0
    max_valid_acc = 0
    min_test_loss = -1
    min_valid_loss = -1
    ckpt_epoch = 0


    if os.path.exists(senti_filename):
        print(f'Checkpoint detected for Sentimental Classifier')
        if(regression):
            ckpt_epoch, min_test_loss, min_valid_loss, senti_voc = load_checkpoint_reg(my_lstm, optimizer, min_test_loss, min_valid_loss, senti_filename, senti_voc)
        else:
            ckpt_epoch, max_test_acc, max_valid_acc, senti_voc = load_checkpoint(my_lstm, optimizer, max_test_acc, max_valid_acc, senti_filename, senti_voc)
        print('Checkpoints loaded for Sentimental Classifier')
    else:
        print('No Checkpoint detected for Sentimental Classifier, starting from scractch')
    # ________________Everything Sentimental Classifier____________________
    
    train_sentimental_chatbot(loadFilename, voc, pairs, encoder, decoder, embedding,
                              encoder_optimizer, decoder_optimizer, 
                              optimizer, my_lstm, dialogdata,
                              model_settings_chatbot, train_settings_chatbot, data_settings_chatbot,
                              model_settings_senti, train_settings_senti, data_settings_senti, save_dir,
                              loadFilename)

def train_sentimental_chatbot(model_name_chatbot, voc, pairs, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer,
          senti_optimizer, sentimental_classifier, dialogdata,
          model_settings_chatbot, train_settings_chatbot, data_settings_chatbot,
          model_settings_senti, train_settings_senti, data_settings_senti, save_dir,
          loadFilename=None):
    
    voc_senti = dialogdata.len_voc_keys
    
    regression = model_settings_senti['regression']
    batch_size_senti = 1
    batch_size_chatbot = data_settings_chatbot['batch_size']
    clip_chatbot = train_settings_chatbot['clip']
    teacher_forcing_ratio = train_settings_chatbot['teacher_forcing_ratio']
    encoder_n_layers = model_settings_chatbot['encoder_num_layers']
    decoder_n_layers = model_settings_chatbot['decoder_num_layers']
    n_iteration = train_settings_chatbot['n_iteration']
    print_every = train_settings_chatbot['print_every']
    save_every = train_settings_chatbot['save_every']
    max_length = data_settings_chatbot['max_seq']
    emotion_mode = torch.tensor([train_settings_chatbot['emotion']]).expand(batch_size_chatbot).unsqueeze(1)
    
    
    wandb_logger = Logger(
        f"inm706_sentiment_chatbot_with_backprop_ende",
        project='inm706_CW')
    logger = wandb_logger.get_logger()
    
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size_chatbot)])
                      for _ in range(n_iteration)]
    print_loss_total = 0
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if os.path.exists(loadFilename):
        checkpoint = torch.load(loadFilename, map_location=device)
        start_iteration = checkpoint['iteration'] + 1
        print(f'Checkpoint detected, beginning from {start_iteration}')
    
    if regression:
        model_name = 'Baseline_LSTM_Regressor'
    else:
        model_name = 'Baseline_LSTM_Classifier'
        
    filename_sentimental = f"{model_name_chatbot}_{model_name}_ckpt_.pth"
    
    # Variables to compare testing and validation accuracy FOR CHECKPOINTS ONLY
    min_test_loss = -1
    min_valid_loss = -1
    ckpt_epoch = 0
    
    for iteration in tqdm(range(start_iteration, n_iteration + 1), desc="Training Chatbot"):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        senti_optimizer.zero_grad()
        
        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)
        # Lengths for RNN packing should always be on the CPU
        lengths = lengths.to("cpu")
        
        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0
        
        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size_chatbot)]])
        decoder_input = decoder_input.to(device)
        
        # Parity Check for LSTM or GRU
        if isinstance(encoder_hidden, tuple):  # This checks if the encoder is an LSTM
            # Handle LSTM's hidden and cell states
            # Averaging the forward and backward states from each layer
            decoder_hidden = (
                torch.mean(encoder_hidden[0].view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1),
                torch.mean(encoder_hidden[1].view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1)
            )
        else:
            decoder_hidden = torch.mean(encoder_hidden.view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1)
            
        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        sentimental_classifier.eval()
        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                decoder_to_senti = [voc.index2word[decoder_input[:,my_iter].item()] for my_iter in range(decoder_input.shape[1])]
                senti_input_word = [dialogdata.preprocess_tokens(batch_tokens) for batch_tokens in decoder_to_senti]
                senti_input_word = torch.tensor(senti_input_word, dtype=torch.int).to(device)
                ypred_decoder = sentimental_classifier(senti_input_word)
                if(regression):
                    loss_senti = F.mse_loss(ypred_decoder, emotion_mode.float().to(device))
                else:
                    loss_senti = F.cross_entropy(ypred_decoder, emotion_mode.long().to(device))
                    
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss + loss_senti
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal 
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size_chatbot)]])
                decoder_input = decoder_input.to(device)
                decoder_to_senti = [voc.index2word[decoder_input[:,my_iter].item()] for my_iter in range(decoder_input.shape[1])]
                senti_input_word = [dialogdata.preprocess_tokens(batch_tokens) for batch_tokens in decoder_to_senti]
                senti_input_word = torch.tensor(senti_input_word, dtype=torch.int)
                ypred_decoder = sentimental_classifier(senti_input_word)
                if(regression):
                    loss_senti = F.mse_loss(ypred_decoder, emotion_mode.float())
                else:
                    loss_senti = F.cross_entropy(ypred_decoder, emotion_mode.long())
                    
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss + loss_senti
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        
        
        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip_chatbot)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip_chatbot)
        _ = nn.utils.clip_grad_norm_(sentimental_classifier.parameters(), clip_chatbot)
        
        loss.backward()
        
        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()
        # senti_optimizer.step()
        
        loss_iter_total = sum(print_losses) / n_totals
        # print(loss_iter, n_totals)
        print_loss_total += loss_iter_total
        
        if iteration % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            logger.log({'train_loss': print_loss_avg})
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss_total = 0
            
        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir , model_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict(),
                
                'senti_state_dict': sentimental_classifier.state_dict(),
                'senti_optimizer_state': senti_optimizer.state_dict(),
                'senti_voc': voc_senti
            }, os.path.join(directory, '{},{}_{}.tar'.format(filename_sentimental, iteration, 'checkpoint')))
            print('Checkpoint Saved')
            
    

    
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