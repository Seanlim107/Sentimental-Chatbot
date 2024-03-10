import torch
import os
from torch.nn import CrossEntropyLoss as CrossEntropyLoss
from torch.utils.data import DataLoader as DataLoader
from torch.utils import data
from logger import Logger
from models import SentimentLSTM
from sklearn.model_selection import train_test_split
from utils import parse_arguments, read_settings
from dataset import DialogData
import torch.nn.functional as F
import numpy as np

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    
def calculate_metrics(logger, data_settings, model, dataloader):
    
    model.eval()
    
    for (x,y) in dataloader:
        return
    return
    

def train(data_settings, model_settings, train_settings):
    # dataset = DialogData(voc_init=False, max_seq=10)
    # print(dataset)
    voc_init='False'
    dialogdata = DialogData(voc_init=voc_init, max_seq=data_settings['max_seq'])
    
    train_dataloader,test_dataloader,val_dataloader=dialogdata.setdata_senti(batch_size=train_settings['batch_size'])
    
    my_lstm = SentimentLSTM(vocab_senti_size=dialogdata.len_voc_keys, embedding_dim=model_settings['embedding_dim'],
                            output_size=dialogdata.output_size, lstm_hidden_dim=model_settings['lstm_hidden_dim'], hidden_dim=model_settings['hidden_dim'],
                            hidden_dim2=model_settings['hidden_dim2'],n_layers=model_settings['n_layers'],
                            drop_prob=model_settings['drop_prob'])
    
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
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            ypred = my_lstm(X)
            # ypred = torch.max(ypred, dim=1)[1]
            # print(ypred)
            print('ypred:{}, y:{}'.format(ypred, y))
            loss = F.cross_entropy(ypred, y)
            # print(loss)
            loss.backward()
            optimizer.step()
            total_loss+=loss
            
        # logger.log({'train_loss': total_loss/len(train_dataloader)})
        print('Epoch {}: Train Loss={}',epoch, total_loss/len(train_dataloader))
    return

    
    
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
    
    
    
    
    
    
    
    
