import os
import torch
import pickle
import torch.utils
import pandas as pd
from datasets import load_dataset
from torch.utils import data
from torch.utils.data import Dataset
from collections import Counter, OrderedDict
import re
import nltk
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, DataLoader, random_split




# chatbot_dataset = load_dataset("daily_dialog")
# dialog_df_train = pd.DataFrame.from_dict(chatbot_dataset['train'])

# temp_dict_train = {'sentence': dialog_df_train['dialog'], 'act': dialog_df_train['act'], 'emotion': dialog_df_train['emotion']}

# nltk.load('english', format='text')

class DialogData(data.Dataset):
    UNK_TOKEN = '<UNK>'
    START_TOKEN = "<S>"
    END_TOKEN = "</S>"
    
    def __init__(self, max_seq=None, voc_init=False, lem=False, stem=False):
        self.max_seq = max_seq
        self.unk_token = self.UNK_TOKEN
        self.end_token = self.END_TOKEN
        self.start_token = self.START_TOKEN
        self.chatbot_dataset = load_dataset("daily_dialog")
        
        self.stopwords = stopwords.words('english')
        self.voc_dict = OrderedDict()
        self.output_size = 1
        
        # For lemmatization or stemming
        self.lem = lem
        self.stem = stem
        self.vocab_senti = []
        self.X = []
        self.y = []
        
        # Overall dataset
        self.dialog_df = pd.DataFrame.from_dict(self.chatbot_dataset['train'])
        self.dialog_df = pd.concat([self.dialog_df, pd.DataFrame.from_dict(self.chatbot_dataset['test'])])
        self.dialog_df = pd.concat([self.dialog_df, pd.DataFrame.from_dict(self.chatbot_dataset['validation'])])
        self.dialoglist = None
        
        # #dataset for sentiment analysis
        self.sentiment_sentences_df = self.dialog_df.copy()
        self.sentiment_sentences_df = self.sentiment_sentences_df.apply(pd.Series.explode)
        
        # Referencing code obtained from INM706 Lab 4
        if voc_init:
            # self.voc = self.create_vocab()
            # with open("vocabulary.pkl", "wb") as file:
            #     pickle.dump(self.voc, file)
                
            self.create_vocab_senti()
            self.vocab_senti = sorted(self.vocab_senti)
            with open("vocabulary_senti.pkl", "wb") as file:
                pickle.dump(self.vocab_senti, file)
        else:
            # self.load_self.vocab()
            self.load_vocab_senti()
            self.vocab_senti = sorted(self.vocab_senti)
            
        for idx, word in enumerate(self.vocab_senti):
            self.voc_dict[word] = idx
            
        self.voc_keys = self.voc_dict.keys()
        self.len_voc_keys = len(self.voc_keys)

    def tokenize(self, text):
        text_tokens = word_tokenize(re.sub('\W+', ' ', text.lower()))
        return text_tokens
        
    def trim_sentence(self, tokenized_k):
        if len((tokenized_k)) > self.max_seq:
            tokenized_k = tokenized_k[:self.max_seq]
            tokenized_k.append(self.end_token)
        elif len(tokenized_k) <= self.max_seq:
            tokenized_k.append(self.end_token)
            seq_pad = (self.max_seq - len(tokenized_k) + 1) * [self.unk_token]
            if len(seq_pad) > 0:
                tokenized_k.extend(seq_pad)
        return tokenized_k

    # sets the input (X) and output (y) data for training
    def setdata_senti(self, train_size=0.8, batch_size=16):
        
        #Initialise variables
        sentimentclean = self.sentiment_sentences_df.dropna()
        
        dialoglist = sentimentclean['dialog'].tolist()
        emotionlist = sentimentclean['emotion'].tolist()
        self.output_size = max(emotionlist)
        
        
        # print(dialoglist)
        # set X and y data for training
        for iter,i in enumerate(dialoglist):
            tokenized_list = self.tokenize(i)
            tokenized_list = self.trim_sentence(tokenized_list)
            tokenized_list = [self.voc_dict[x] for x in tokenized_list]
            emotion = emotionlist[iter]
            self.X.append(tokenized_list)
            self.y.append(emotion)
        

        #Load to dataloader
        
        # print(self.X_tensor)

        dataset_to_dataloader = DataLoaderToken(self.X, self.y)
        data_len = len(dataset_to_dataloader)
        train_len = int(data_len*train_size)
        test_len = int((data_len - train_len)/2)
        val_len = data_len - train_len - test_len
        
        train_dataset, test_dataset, val_dataset = random_split(dataset_to_dataloader, [train_len, test_len, val_len])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        
        
        return train_dataloader, test_dataloader, val_dataloader
    
    def create_vocab_senti(self):
        dialogclean = self.sentiment_sentences_df.copy()
        dialogclean = dialogclean['dialog'].dropna()
        dialoglist = dialogclean.tolist()
        
        
        for sentence in dialoglist:
            # print(sentence)
            try:
                sentence_trimmed = self.tokenize(sentence)
                # print(sentence_trimmed)
                for senti_word in sentence_trimmed:
                    if senti_word not in self.vocab_senti:
                            self.vocab_senti.append(senti_word)
            except:
                continue
                # print(sentence)
        
        self.vocab_senti.append(self.unk_token)
        self.vocab_senti.append(self.start_token)
        self.vocab_senti.append(self.end_token)
        
        return
    
    def load_vocab_senti(self):
        path_vocab = os.path.join(os.getcwd(), 'vocabulary_senti.pkl')
        if os.path.exists(path_vocab):
            # Load the object back from the file
            with open(path_vocab, "rb") as file:
                self.vocab_senti = pickle.load(file)
        return
    
    
class DataLoaderToken(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.long), torch.tensor(self.y[index], dtype=torch.long)
        
