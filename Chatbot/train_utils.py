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
from logger import Logger
from dataset import batch2TrainData, normalizeString, indexesFromSentence
from tqdm.auto import tqdm

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

#Loss function
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length, teacher_forcing_ratio):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    if isinstance(encoder_hidden, tuple):  # Handle LSTM's hidden and cell states
        decoder_hidden = (
            torch.mean(encoder_hidden[0].view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1),
            torch.mean(encoder_hidden[1].view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1)
        )
    else:  # Handle GRU's hidden state
        decoder_hidden = torch.mean(encoder_hidden.view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_variable[t].view(1, -1)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def validate(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
             batch_size, max_length, teacher_forcing_ratio):
    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    if isinstance(encoder_hidden, tuple):  # Handle LSTM's hidden and cell states
        decoder_hidden = (
            torch.mean(encoder_hidden[0].view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1),
            torch.mean(encoder_hidden[1].view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1)
        )
    else:  # Handle GRU's hidden state
        decoder_hidden = torch.mean(encoder_hidden.view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_variable[t].view(1, -1)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    return sum(print_losses) / n_totals

def trainIters(model_name, voc, training_batches, validation_batches, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               train_batches_per_val, val_batches_per_epoch, save_every, clip, teacher_forcing_ratio, loadFilename=None):
    wandb_logger = Logger(model_name, project='inm706_CW')
    logger = wandb_logger.get_logger()

    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    training_loss_list = []
    validation_loss_list = []

    if os.path.exists(loadFilename):
        checkpoint = torch.load(loadFilename)
        start_iteration = checkpoint['iteration'] + 1

    print("Training...")
    print(f"Start Iteration: {start_iteration}")
    for start_iteration in tqdm(range(start_iteration, start_iteration + n_iteration), desc="Training Chatbot"):
        # Training
        encoder.train()
        decoder.train()
        train_loss = 0
        for _ in range(train_batches_per_val):
            if start_iteration > n_iteration:
                break
            training_batch = training_batches[start_iteration - 1]
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                         decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, 10, teacher_forcing_ratio)
            train_loss += loss

        avg_train_loss = train_loss / train_batches_per_val
        training_loss_list.append(avg_train_loss)
        logger.log({'train_loss': avg_train_loss})

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(val_batches_per_epoch):
                validation_batch = validation_batches[i % len(validation_batches)]
                input_variable, lengths, target_variable, mask, max_target_len = validation_batch
                val_batch_loss = validate(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                                          decoder, embedding, batch_size, 10, teacher_forcing_ratio)
                val_loss += val_batch_loss

        avg_val_loss = val_loss / val_batches_per_epoch
        validation_loss_list.append(avg_val_loss)
        logger.log({'val_loss': avg_val_loss})

        print(f"Iteration: {start_iteration}; Avg Training Loss: {avg_train_loss:.4f}; Avg Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        if start_iteration % save_every == 0:
            directory = os.path.join(save_dir, model_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': start_iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': avg_train_loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(start_iteration, 'checkpoint')))
            print('Checkpoint Saved')


