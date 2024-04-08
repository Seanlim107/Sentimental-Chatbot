import os, re
import time
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from dataset import MoviePhrasesData
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils import parse_arguments, read_settings
from tokenizers import SentencePieceBPETokenizer
from logger import Logger
from models import EncoderGRU, DecoderGRU, DecoderGRU_Attention
from transformers import PreTrainedTokenizerFast, AutoTokenizer
# from test import chat

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)



def train(data_settings, model_settings, train_settings):
    voc_init = False
    moviephrasesdata = MoviePhrasesData(voc_init=voc_init, max_seq_len=data_settings['max_seq'])
    
    voc = moviephrasesdata.vocab
    
    data_len = len(moviephrasesdata)
    train_len = int(train_settings['train_size']*data_len)
    test_len =  int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len
    
    # print(train_len, test_len, val_len)
    
    train_dataset, test_dataset, val_dataset = random_split(moviephrasesdata, [train_len, test_len, val_len])
    #________________________________________________________TURN OFF FOR DEBUGGING__________________________________________________________________
    # wandb_logger = Logger(
    #     f"inm706_sentiment_chatbot",
    #     project='inm706_Chatbot')
    # logger = wandb_logger.get_logger()
    #________________________________________________________TURN OFF FOR DEBUGGING__________________________________________________________________
    
    train_dataloader = DataLoader(train_dataset, batch_size=data_settings['batch_size'], shuffle=True)
    test_train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    encoder = EncoderGRU(vocab_size=len(voc), embedding_size=model_settings['embedding_dim'], hidden_size=model_settings['hidden_dim'])
    if model_settings['use_attention']:
        decoder = DecoderGRU_Attention(vocab_size=len(voc), embedding_size=model_settings['embedding_dim'], hidden_size=model_settings['hidden_dim'], num_layers=model_settings['num_layers'], feature_size=model_settings['feature_size'])
    else:
        decoder = DecoderGRU(vocab_size=len(voc), embedding_size=model_settings['embedding_dim'], hidden_size=model_settings['hidden_dim'], num_layers=model_settings['num_layers'], feature_size=model_settings['feature_size'])
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    if train_settings['optimizer']==1:
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=train_settings['lr'], weight_decay=train_settings['decay_ratio'])
        filename = "model_and_optimizer_1.pth"
    elif train_settings['optimizer']==2:
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=train_settings['lr'], weight_decay=train_settings['decay_ratio'])
        decoder_optimizer = decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=train_settings['lr']*train_settings['decoder_decay_ratio'])
        filename = "model_and_optimizer_2.pth"
    torch.autograd.set_detect_anomaly(True)
    # 'encoder_state_dict': encoder.state_dict(),
    #             'decoder_state_dict': decoder.state_dict(),
    #             'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
    #             'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
    if os.path.exists(filename):
        ckpt = torch.load(filename, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        
        if train_settings['optimizer']==1:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            encoder_optimizer.load_state_dict(ckpt['encoder_optimizer_state_dict'])
            decoder_optimizer.load_state_dict(ckpt['decoder_optimizer_state_dict'])
        print(f'Checkpoint detected, starting from checkpoint')
    else:
        print('No checkpoint, starting from scratch')
    # print(encoder)
    # encoder = encoder.to(device)
    # decoder = decoder.to(device)
    
    encoder.train()
    decoder.train()
    
    total_epochs = train_settings['epochs']
        
    num_seqs = train_settings['num_seq_process']
    current_batch = 0
    total_phrase_pairs = 0
    loss_function = nn.CrossEntropyLoss()
    
    for epoch in range(total_epochs):
        print(f'Epoch {epoch}')
        epoch_loss = 0
        processed_total_batches = 0
        for id, (idx, (prompt,reply)) in enumerate(train_dataloader):
            if not current_batch:
                total_phrase_pairs = 0
                loss_batch = 0
            if prompt.size()[1] == 0:
                continue
            if train_settings['optimizer']==1:
                optimizer.zero_grad()
            else:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
            # print(prompt.size())
            current_batch += 1
            prompt = prompt.to(device)
            reply = reply.to(device)
            batch_size = prompt.size()[1]
            
            seq_length = prompt.size()[2]
            total_phrase_pairs += batch_size
            
            data = prompt
            data = data.view(-1, batch_size, seq_length)[0]

            output_encoder, hidden_encoder = encoder(data)

            true_response = reply

            decoder_input = torch.LongTensor([[moviephrasesdata.vocab.index(moviephrasesdata.bos_token) for _ in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            
            hidden_encoder = hidden_encoder[-1, :, :].unsqueeze(0)

            if model_settings['use_attention']:
                output_decoder, hidden_decoder = decoder(decoder_input, hidden_encoder, output_encoder)
            else:
                output_decoder, hidden_decoder = decoder(decoder_input, hidden_encoder)

            top_k_predict = output_decoder.topk(1).indices
            # print(top_k_predict)
            targets = true_response[0, :, 1]
            # print(targets)
            targets = targets.to(device)
            
            loss = loss_function(output_decoder.squeeze(0), targets)
            loss_dialogue = 0
            loss_dialogue += loss
            
            for idx in range(1, true_response.size()[2] - 1):
                decoder_input = top_k_predict.view(-1, batch_size)
                # print('Predicted Response:', moviephrasesdata.tokenizer.decode((decoder_input.squeeze(0))))
                if model_settings['use_attention']:
                    output_decoder, hidden_decoder = decoder(decoder_input, hidden_decoder, output_encoder)
                else:
                    output_decoder, hidden_decoder = decoder(decoder_input, hidden_decoder)
                
                # print(output_decoder)
                # print(output_decoder.topk(5).indices)
                top_k_predict = output_decoder.topk(1).indices
                
                targets = true_response[0, :, idx + 1]
                # loss from the predicted vs true tokens
                loss = loss_function(output_decoder.squeeze_(0), targets)
                loss_dialogue += loss
                
                
            loss_dialogue = loss_dialogue / seq_length
            # print(loss_dialogue)
            # add dialogue loss to the batch loss
            loss_batch += loss_dialogue
            
            if not current_batch % num_seqs:
                current_batch = 0
                loss_batch = loss_batch / num_seqs
                epoch_loss += loss_batch.item()
                processed_total_batches += 1
                # print('Loss={0:.6f}, total phrase pairs in the batch = {1:d}'.format(loss_batch, total_phrase_pairs))
                loss_batch.backward()
                # print("IOANDASJNDASNDAJWNDOAXDA")
                if train_settings['optimizer']==1:
                    optimizer.step()
                else:
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    
        # print(f'Prompt: {prompt}')
        # print(f'Response: {pred_response}')
        
        if(train_settings['optimizer']==1):
            torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epoch,
        }, 'model_and_optimizer_1.pth')
        else:
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'epochs': epoch,
            }, 'model_and_optimizer_2.pth')
            
        # print(chat("How is life my boy"))
                    
        #________________________________________________________TURN OFF FOR DEBUGGING__________________________________________________________________
        # epoch_loss = epoch_loss / processed_total_batches
        # logger.log({"epoch_loss": epoch_loss})
        # print('Loss={0:.6f}, total phrase pairs in the batch = {1:d}, total batches processed = {2:d}'.format(epoch_loss,
        #                                                                                                     total_phrase_pairs,
        #                                                                                                     processed_total_batches))
        # ________________________________________________________TURN OFF FOR DEBUGGING__________________________________________________________________
        
            
    return

if __name__ == '__main__':
    args = parse_arguments()

    # Read settings from the YAML file
    filepath=os.path.dirname(os.path.realpath(__file__))
    settings = read_settings(filepath+args.config)

    # Access and use the settings as needed
    data_settings = settings.get('data', {})
    model_settings = settings.get('model', {})
    train_settings = settings.get('train', {})
    # print(model_settings)
    train(data_settings, model_settings, train_settings)
# moviehrasesdata = MoviePhrasesData()
# print(len(moviehrasesdata.vocab))
