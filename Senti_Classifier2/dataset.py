import os
from zipfile import ZipFile
import datasets
import pandas as pd
import torch.utils
from torch.utils import data
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import Counter, OrderedDict
import re
from torch.utils.data import Dataset
import numpy as np

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')


class DialogData(data.Dataset):
    def __init__(self,max_seq=None, voc_init_cache=False, lem=True, stem=False, remove_stop=True):
        
        # Initialise variables
        self.max_seq = max_seq
        self.lem = lem
        self.lemmatizer = WordNetLemmatizer()
        self.stem = stem
        self.stemmer = PorterStemmer()
        self.remove_stop = remove_stop
        self.unk_token = '<UNK>'
        self.start_token = '<S>'
        self.end_token = '</S>'
        
        
        self.stopwords = stopwords.words('english')
        self.voc_dict = OrderedDict()
        self.vocab_senti = []
        self.X = []
        self.y = []
        
        # Initiallise dataset dataframe
        filepath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'ijcnlp_dailydialog/ijcnlp_dailydialog')

        dialog_path=os.path.join(filepath,'dialogues_text.txt')
        emotion_path=os.path.join(filepath,'dialogues_emotion.txt')
        dialog_list=[]
        emotion_list=[]
        with open(dialog_path, 'rb') as dialog_file, open(emotion_path, 'rb') as emotion_file:
            for (dialog_line, emotion_line) in zip(dialog_file, emotion_file):
                dialog = dialog_line.decode().split('__eou__')[:-1]
                emotion = emotion_line.decode().split(" ")[:-1]
                emotion = [int(x) for x in emotion]
                dialog_list.append(dialog)
                emotion_list.append(emotion)
        
        self.dialog_df = pd.DataFrame(list(zip(dialog_list, emotion_list)),columns=['dialog','emotion'])
        self.output_size = 3

        #dataset for sentiment analysis
        self.sentiment_sentences_df = self.dialog_df.copy()
        # print(self.sentiment_sentences_df[self.sentiment_sentences_df.index.duplicated()])
        self.sentiment_sentences_df = self.sentiment_sentences_df.apply(pd.Series.explode, axis=0, ignore_index=True)
        self.sentiment_sentences_df['length'] = [len(word_tokenize(re.sub('\W+', ' ', self.sentiment_sentences_df['dialog'][x]))) for x in range(len(self.sentiment_sentences_df))]
        self.sentiment_sentences_df.loc[self.sentiment_sentences_df['emotion'].isin([0]), "emotion"] = 7
        self.sentiment_sentences_df.loc[self.sentiment_sentences_df['emotion'].isin([1,2,3,5]), "emotion"] = 0
        self.sentiment_sentences_df.loc[self.sentiment_sentences_df['emotion'].isin([4,6]), "emotion"] = 2
        self.sentiment_sentences_df.loc[self.sentiment_sentences_df['emotion'].isin([7]), "emotion"] = 1
        len_pos=(len(self.sentiment_sentences_df[self.sentiment_sentences_df['emotion']==2]))
        len_neg=(len(self.sentiment_sentences_df[self.sentiment_sentences_df['emotion']==0]))
        len_neu=(len(self.sentiment_sentences_df[self.sentiment_sentences_df['emotion']==1]))
        
        self.sentiment_sentences_df_tofilter = self.sentiment_sentences_df[self.sentiment_sentences_df['emotion'] == 1]
        # print(len(self.sentiment_sentences_df_tofilter))
        np.random.seed(42)
        index_tofilter = np.random.choice(self.sentiment_sentences_df_tofilter.index,(len_neu-len_pos-len_neg), replace=False)
        
        df_modified = self.sentiment_sentences_df.drop(index_tofilter)
        self.sentiment_sentences_df = df_modified
        # print(len(df_modified))
        
        # Referencing code obtained from INM706 Lab 4
        if voc_init_cache:
            self.create_vocab_senti()
            self.vocab_senti = sorted(self.vocab_senti)
            with open("vocabulary_senti.pkl", "wb") as file:
                pickle.dump(self.vocab_senti, file)
        else:
            self.load_vocab_senti()
            self.vocab_senti = sorted(self.vocab_senti)
            
        for idx, word in enumerate(self.vocab_senti):
            self.voc_dict[word] = idx
            
        self.voc_keys = self.voc_dict.keys()
        self.len_voc_keys = len(self.voc_keys)
            
    def preprocess_text(self, text):
        text_tokens = word_tokenize(re.sub('\W+', ' ', text.lower()))
        if(self.remove_stop):
            text_tokens = [word for word in text_tokens if word not in self.stopwords]
        if(self.lem):
            text_tokens = [self.lemmatizer.lemmatize(word) for word in text_tokens]
        if(self.stem):
            text_tokens = [self.stemmer.stem(word) for word in text_tokens]
        # print(text_tokens)
        return text_tokens
    
    def create_vocab_senti(self):
        dialogclean = self.sentiment_sentences_df.copy()
        dialogclean = dialogclean['dialog'].dropna()
        dialoglist = dialogclean.tolist()
        for sentence in dialoglist:
            try:
                sentence_trimmed = self.preprocess_text(sentence)
                # print(sentence_trimmed)
                for senti_word in sentence_trimmed:
                    if senti_word not in self.vocab_senti:
                            self.vocab_senti.append(senti_word)
            except:
                raise Exception('Unknown error occured')
            
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
        
    def transform_tokens_to_voc_idx(self, text):
        ind = [self.voc_dict[w] if w in self.voc_dict else self.voc_dict[self.unk_token] for w in text]
        return ind
    
    def preprocess_tokens(self, text):
        to_put = self.max_seq - 2
        ind_tokens = self.preprocess_text(text)
        ind_tokens = self.transform_tokens_to_voc_idx(ind_tokens)
        if len(ind_tokens) >= to_put:
            ind_tokens = ind_tokens[:to_put]
        else:
            ind_tokens.extend([self.voc_dict[self.unk_token]] * (to_put - len(ind_tokens)))
            
        ind_tokens.insert(0, self.voc_dict[self.start_token])
        ind_tokens.append(self.voc_dict[self.end_token])
        return ind_tokens
        
    
    
    
    def prepare_dataloader(self):
        dialogclean = self.sentiment_sentences_df.copy()
        dialogclean = dialogclean.dropna()
        dialoglist = dialogclean['dialog'].tolist()
        emotionlist = dialogclean['emotion'].tolist()
        ind_list=[]
        for sentence in dialoglist:
            try:
                ind_tokens = self.preprocess_tokens(sentence)
                ind_list.append(ind_tokens)
                # print(ind_tokens)
                
            except:
                raise Exception('Unknown error occured')
        self.X = ind_list
        self.y = emotionlist
        return ind_list, emotionlist
    
        
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = torch.tensor(self.X[index], dtype=torch.int)
        y = torch.tensor(int(self.y[index]), dtype=torch.int)
        return X, y
    
    
# dailydialog = DialogData(max_seq = 10, voc_init_cache=False)
# print(dailydialog.sentiment_sentences_df.describe())
# X,y = dailydialog.prepare_dataloader()
