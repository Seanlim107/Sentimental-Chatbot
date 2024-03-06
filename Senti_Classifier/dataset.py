import os
import torch
import pickle
import torch.utils
import pandas as pd
from datasets import load_dataset
from torch.utils import data
from collections import Counter, OrderedDict
import re

# chatbot_dataset = load_dataset("daily_dialog")
# dialog_df_train = pd.DataFrame.from_dict(chatbot_dataset['train'])

# temp_dict_train = {'sentence': dialog_df_train['dialog'], 'act': dialog_df_train['act'], 'emotion': dialog_df_train['emotion']}



class DialogData(data.Dataset):
    UNK_TOKEN = '<UNK>'
    START_TOKEN = "<S>"
    END_TOKEN = "</S>"
    
    def __init__(self, max_seq=None, voc_init=False):
        self.max_seq = max_seq
        self.unk_token = self.UNK_TOKEN
        self.end_token = self.END_TOKEN
        self.start_token = self.START_TOKEN
        self.chatbot_dataset = load_dataset("daily_dialog")
        
        
        # Overall dataset
        self.dialog_df = pd.DataFrame.from_dict(self.chatbot_dataset['train'])
        self.dialog_df = pd.concat([self.dialog_df, pd.DataFrame.from_dict(self.chatbot_dataset['test'])])
        self.dialog_df = pd.concat([self.dialog_df, pd.DataFrame.from_dict(self.chatbot_dataset['validation'])])
        
        
        
        # #dataset for sentiment analysis
        self.sentiment_sentences_df = self.dialog_df.copy()
        self.sentiment_sentences_df = self.sentiment_sentences_df.apply(pd.Series.explode)
        
        # Referencing code obtained from INM706 Lab 4
        if voc_init:
            self.voc = self.create_vocab()
            with open("vocabulary.pkl", "wb") as file:
                pickle.dump(self.voc, file)
        else:
            self.load_vocab()
        
    # ____________ Referencing code from INM706 Lab 4 ___________
    # ___________________________________________________________
    def load_vocab(self):
        path_vocab = os.path.join(os.getcwd(), 'vocabulary.pkl')
        if os.path.exists(path_vocab):
            # Load the object back from the file
            with open(path_vocab, "rb") as file:
                self.voc = pickle.load(file)
        return
    
    def create_vocab(self):
        dialogclean = self.sentiment_sentences_df.copy()
        dialogclean = dialogclean['dialog'].dropna()
        dialoglist = dialogclean.tolist()
        
        vocab = []
        
        for sentence in dialoglist:
            try:
                sentence_trimmed = re.sub('\W+', ' ', sentence).split()
                # print(sentence_trimmed)
                for word in sentence_trimmed:
                    if word not in vocab:
                        vocab.append(word)
            except:
                print(sentence)
        
        vocab.append(self.unk_token)
        vocab.append(self.start_token)
        vocab.append(self.end_token)
        
        return sorted(vocab)
    
    # sets the input (X) and output (y) data for training
    def setdata(self):
        sentimentclean = self.sentiment_sentences_df.dropna()
        
        dialoglist = sentimentclean['dialog'].tolist()
        emotionlist = sentimentclean['emotion'].tolist()
        
        X = []
        y = []
        
        for iter,i in enumerate(dialoglist):
            tokens=[]
            tokenized_list = re.sub('\W+', ' ', i).split()
            emotion = emotionlist[iter]
            tokenized_list = self.trim_sentence(tokenized_list)
            tokens.append(tokenized_list)
            X.append(tokens)
            y.append(emotion)
        return pd.DataFrame({'X':X, 'y':y})
        
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
        
    def load_dialog_text(self, dialog):
        list_lines_text = dialog['lines']
        list_lines = re.findall(r'\b\w+\b', list_lines_text)
        print(list_lines)
        movie_id = dialog['movie']
        all_lines = []
        lines_dict = {}
        prev_line = None
        count = 0
        for line in list_lines:
            line_text = self.lines_data.loc[(self.lines_data['line_id'] == line) & (self.lines_data['movie_id'] == movie_id)]
            if len(line_text) == 1:
                if prev_line is not None:
                    lines_dict[prev_line] = line_text['line'].item()
                    prev_line = line_text['line'].item()
                else:
                    prev_line = line_text['line'].item()
                all_lines.append(line_text['line'].item())


        pairs = [(all_lines[i], all_lines[i + 1]) for i in range(len(all_lines) - 1)]
        # print('pairs ', pairs)
        # print('linesdict ',lines_dict)
        return pairs, lines_dict
    
    # ______________________________________________________________
    # ____________ Referencing code from INM706 Lab 4 ______________


# dataset_dialog = DialogData(voc_init=True, max_seq=10)
# input_df = dataset_dialog.setdata()
# print(input_df.head())