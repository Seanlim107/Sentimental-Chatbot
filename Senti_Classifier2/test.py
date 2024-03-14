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
torch.cuda.empty_cache()

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

def evaluate(logger, data_settings, model, dataloader):
    model.eval()
    num_outputs = data_settings['num_output']
    true_labels = []
    predicted_labels = []
    
    for X,y in dataloader:
        X, y = X.to(device), y.to(device)
        ypred = model(X)
        ypred = torch.argmax(ypred, axis=0, keepdims=False)
        
        true_labels.append(y.item())
        predicted_labels.append(ypred.item())
        
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    
    logger.log({"Overall Accuracy": overall_accuracy,
                "Precision": precision,
                "Recall": recall,
    })
    print("Overall accuracy = {0:.4f}, precision = {1:.4f}, recall={2:.4f}".format(overall_accuracy, precision, recall))
    
    return
            
def train(data_settings, model_settings, train_settings):
    # dataset = DialogData(voc_init=False, max_seq=10)
    # print(dataset)
    voc_init='True'
    dialogdata = DialogData(voc_init_cache=voc_init, max_seq=data_settings['max_seq'])
    dialogdata.prepare_dataloader()
    data_len = len(dialogdata.X)
    train_len = int(data_len*data_settings['train_size'])
    test_len = int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len
    train_dataset,test_dataset,val_dataset=random_split(dialogdata, [train_len, test_len, val_len])
    train_dataloader = DataLoader(train_dataset, batch_size=data_settings['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

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
        
        logger.log({'train_loss': total_loss/len(train_dataloader)})
        print('Epoch:{}, Train Loss:{}'.format(epoch, total_loss/len(train_dataloader)))
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        # evaluate(logger,data_settings,my_lstm,train_dataloader)00
        evaluate(logger,data_settings,my_lstm,test_dataloader)
        evaluate(logger,data_settings,my_lstm,val_dataloader)
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