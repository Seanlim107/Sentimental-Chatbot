import os, re
import torch
import wandb
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import Counter, OrderedDict

from models import SentimentLSTM
from utils import parse_arguments, read_settings, load_checkpoint, save_checkpoint
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

def evaluate(optimizer, logger, data_settings, model, dataloader, mode):
    model.eval()
    num_outputs = data_settings['num_output']
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            ypred = model(X)
            
            ypred = torch.argmax(ypred, axis=1, keepdims=False)
            
            true_labels.append(y.item())
            predicted_labels.append(ypred.item())
            
    # Get Recall score and precision score
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    
    logger.log({f"{mode} Overall Accuracy": overall_accuracy,
                f"{mode} Precision": precision,
                f"{mode} Recall": recall,
    })
    print(mode)
    print("Overall accuracy = {0:.4f}, precision = {1:.4f}, recall={2:.4f}".format(overall_accuracy, precision, recall))
    # print(true_labels[:64],'\n',predicted_labels[:64])
    return overall_accuracy,precision
            
def train(data_settings, model_settings, train_settings):
    voc_init='True'
    dialogdata = DialogData(voc_init_cache=voc_init, max_seq=data_settings['max_seq'])
    _,y=dialogdata.prepare_dataloader()
    
    data_len = len(dialogdata.X)
    num_outputs = data_settings['num_output']
    train_len = int(data_len*data_settings['train_size'])
    test_len = int((data_len - train_len)/2)
    val_len = data_len - train_len - test_len
    
    
    
    # Set Class weights
    weightclass = [len(y)/(num_outputs*y.count(a)) for a in range(num_outputs)]
    weightclass = torch.tensor(weightclass)
    
    weightclass = weightclass.to(device)
    
    train_dataset,test_dataset,val_dataset=random_split(dialogdata, [train_len, test_len, val_len])
    train_dataloader = DataLoader(train_dataset, batch_size=data_settings['batch_size'], shuffle=True)
    test_train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    my_lstm = SentimentLSTM(vocab_senti_size=dialogdata.len_voc_keys, embedding_dim=model_settings['embedding_dim'],
                            output_size=data_settings['num_output'], lstm_hidden_dim=model_settings['lstm_hidden_dim'], hidden_dim=model_settings['hidden_dim'],
                            hidden_dim2=model_settings['hidden_dim2'],n_layers=model_settings['n_layers'],
                            drop_prob=model_settings['drop_prob'])
    
    my_lstm = my_lstm.to(device)
    optimizer = torch.optim.Adam(list(my_lstm.parameters()), lr = train_settings['learning_rate'])
    
    wandb_logger = Logger(
        f"inm706_sentiment_chatbot",
        project='inm706_CW')
    logger = wandb_logger.get_logger()
    
    model_name = 'Baseline_LSTM'
    filename = f"{model_name}_ckpt_.pth"
    
    # Variables to compare testing and validation accuracy FOR CHECKPOINTS ONLY
    max_test_acc = 0
    max_valid_acc = 0
    ckpt_epoch = 0
    
    # Load existing model if it exists
    if os.path.exists(filename):
        ckpt_epoch, max_test_acc, max_valid_acc = load_checkpoint(my_lstm, optimizer, max_test_acc, max_valid_acc, filename)
        print(f'Checkpoint detected, starting from epoch {ckpt_epoch}')
    else:
        print('No checkpoint, starting from scratch')
    
    for epoch in range(ckpt_epoch, train_settings['epochs']):
        total_loss = 0
        my_lstm.train()
        for iter,(X,y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            ypred = my_lstm(X)
            
            loss = F.cross_entropy(ypred, y.long(), weight=weightclass)
            loss.backward()
            optimizer.step()
            total_loss+=loss
        
        logger.log({'train_loss': total_loss/len(train_dataloader)})
        print('Epoch:{}, Train Loss:{}'.format(epoch, total_loss/len(train_dataloader)))

        train_acc, train_prec = evaluate(optimizer, logger,data_settings,my_lstm,test_train_dataloader,mode='Train')
        test_acc, test_prec = evaluate(optimizer, logger,data_settings,my_lstm,test_dataloader,mode='Test')
        valid_acc, valid_prec = evaluate(optimizer, logger,data_settings,my_lstm,val_dataloader,mode='Valid')
        
        # Save checkpoint if model outperforms current model
        if((test_acc > max_test_acc) and (valid_acc > max_valid_acc)):
            max_test_acc = test_acc
            max_valid_acc = valid_acc
            save_checkpoint(epoch, my_lstm, 'Baseline_LSTM', optimizer, test_acc, valid_acc)
            print('model saved')
    return

# def calculate_metrics(logger, data_settings, model):
    

def main():
    args = parse_arguments()

    # Read settings from the YAML file
    filepath=os.path.dirname(os.path.realpath(__file__))
    settings = read_settings(filepath+args.config)

    # Access and use the settings as needed
    data_settings = settings.get('data_senti', {})
    model_settings = settings.get('senti_model', {})
    train_settings = settings.get('train_senti', {})

    train(data_settings, model_settings, train_settings)
    
    
if __name__ == '__main__':
    main()