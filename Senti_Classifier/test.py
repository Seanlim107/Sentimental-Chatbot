import os, re
import torch
import wandb
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import Counter, OrderedDict

from models import SentimentLSTM
from utils import parse_arguments, read_settings, load_checkpoint, save_checkpoint, load_checkpoint_reg, save_checkpoint_reg
from logger import Logger
from dataset import DialogData
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
# torch.cuda.empty_cache()

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

def evaluateLoop(data_settings, model_settings, train_settings):
    voc_init=data_settings['voc_init']
    regression = model_settings['regression']
    
    if regression:
        model_name = 'Baseline_LSTM_Regressor'
    else:
        model_name = 'Baseline_LSTM_Classifier'
    filename = f"{model_name}_ckpt_.pth"
    dialogdata = DialogData(voc_init_cache=voc_init, max_seq=data_settings['max_seq'], regression=regression)
    voc = dialogdata.voc_keys
    my_lstm = SentimentLSTM(vocab_senti_size=len(voc), embedding_dim=model_settings['embedding_dim'],
                            output_size=data_settings['num_output'], lstm_hidden_dim=model_settings['lstm_hidden_dim'], hidden_dim=model_settings['hidden_dim'],
                            hidden_dim2=model_settings['hidden_dim2'],n_layers=model_settings['n_layers'],
                            drop_prob=model_settings['drop_prob'], regression=regression)
    
    my_lstm = my_lstm
    optimizer = torch.optim.Adam(list(my_lstm.parameters()), lr = train_settings['learning_rate'])
    max_test_acc = 0
    max_valid_acc = 0
    min_test_loss = -1
    min_valid_loss = -1
    ckpt_epoch = 0
    
    if os.path.exists(filename):
        if(regression):
            ckpt_epoch, min_test_loss, min_valid_loss, voc = load_checkpoint_reg(my_lstm, optimizer, min_test_loss, min_valid_loss, filename, voc)
        else:
            ckpt_epoch, max_test_acc, max_valid_acc, voc = load_checkpoint(my_lstm, optimizer, max_test_acc, max_valid_acc, filename, voc)
        print(f'Checkpoint detected')
    else:
        raise Exception('No file detected')
    
    input_sentence = ''
    
    while(1):
        input_sentence = input('> ')
        if input_sentence == 'q' or input_sentence == 'quit': break
        input_sentence = dialogdata.preprocess_tokens(input_sentence)
        input_sentence = torch.tensor(input_sentence, dtype=torch.int)
        input_sentence = input_sentence.unsqueeze(0)
        ypred = my_lstm(input_sentence)
        if(not regression):
            ypred = torch.argmax(ypred, axis=1, keepdims=False) -1
        print(ypred)
        
    

# def calculate_metrics(logger, data_settings, model):
    

def main():
    args = parse_arguments()

    # Read settings from the YAML file
    filepath=os.path.dirname(os.path.realpath(__file__))
    settings = read_settings(filepath+args.config)

    # Access and use the settings as needed
    data_settings = settings.get('data_senti', {})
    model_settings = settings.get('model_senti', {})
    train_settings = settings.get('train_senti', {})

    evaluateLoop(data_settings, model_settings, train_settings)
    
    
