import os, re
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import wandb
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import Counter, OrderedDict
import dataset
from models import SentimentLSTM
from utils import parse_arguments, read_settings
from logger import Logger
from dataset import DialogData
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    
def train(data_settings, model_settings, train_settings):
    # dataset = DialogData(voc_init=False, max_seq=10)
    # print(dataset)
    voc_init='False'
    dialogdata = DialogData(voc_init_cache=voc_init, max_seq=data_settings['max_seq'])
    data_len = len(dialogdata.sentiment_sentences_df)
    train_len = int(data_len*data_settings['train_size'])
    test_len = int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len
    train_dataset,test_dataset,val_dataset=random_split(dialogdata, [train_len, test_len, val_len])
    train_dataloader = DataLoader(train_dataset, batch_size=data_settings['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=data_settings['batch_size'], shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=data_settings['batch_size'], shuffle=False)

    # print('HELP ME', dialogdata.len_voc_keys)
    my_lstm = SentimentLSTM(vocab_senti_size=dialogdata.len_voc_keys, embedding_dim=model_settings['embedding_dim'],
                            output_size=dialogdata.output_size, lstm_hidden_dim=model_settings['lstm_hidden_dim'], hidden_dim=model_settings['hidden_dim'],
                            hidden_dim2=model_settings['hidden_dim2'],n_layers=model_settings['n_layers'],
                            drop_prob=model_settings['drop_prob'])
    # print(my_lstm)
    #set train mode
    my_lstm.train()
    my_lstm = my_lstm.to(device)
    optimizer = torch.optim.Adam(list(my_lstm.parameters()), lr = train_settings['learning_rate'])
    
    wandb_logger = Logger(
        f"inm706_sentiment_chatbot",
        project='inm706_CW')
    logger = wandb_logger.get_logger()
    
    for epoch in range(train_settings['epochs']):
        total_loss = 0
        
        for iter,(X,y) in enumerate(train_dataloader):
            # print(X)
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            ypred = my_lstm(X)
            
            # ypred = torch.max(ypred, dim=1)[1]
            # print(torch.max(ypred, dim=1)[1])
            # print('ypred:{}, y:{}'.format(ypred, y.long()))
            loss = F.cross_entropy(ypred, y.long())
            # print(loss)
            loss.backward()
            optimizer.step()
            total_loss+=loss
        # logger.log({'train_loss': total_loss/len(train_dataloader)})
        logger.log({'train_loss': total_loss/len(train_dataloader)})
        print('Epoch:{}, Train Loss:{}'.format(epoch, total_loss/len(train_dataloader)))
    return

# def calculate_metrics(logger, data_settings, model):
    

def main():
    args = parse_arguments()

    # Read settings from the YAML file
    filepath=os.path.dirname(os.path.realpath(__file__))
    settings = read_settings(filepath+args.config)

    # Access and use the settings as needed
    data_settings = settings.get('data', {})
    model_settings = settings.get('model', {})
    train_settings = settings.get('train', {})

    train(data_settings, model_settings, train_settings)
    
    
if __name__ == '__main__':
    main()