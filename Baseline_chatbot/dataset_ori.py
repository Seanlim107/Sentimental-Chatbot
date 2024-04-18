import os
import torch
import pickle
import torch.utils
import pandas as pd
from torch.utils import data
from collections import Counter, OrderedDict

import re


class MoviePhrasesData(data.Dataset):
    UNK = '<UNK>'
    START_TOKEN = "<S>"
    END_TOKEN = "</S>"

    # voc: vocabulary, word:idx
    # all dialogues:
    def __init__(self, max_seq_len=None, voc_init=False):

        custom_header = ['user0', 'user1', 'movie', 'lines']
        custom_header_lines = ['line_id', 'user_id', 'movie_id', 'user_name', 'line']
        self.movies_data = pd.read_csv(
            os.path.join('movie_dataset', 'movie_conversations.tsv'), sep='\t', header=None,
            names=custom_header)
        self.lines_data = pd.read_csv(
            os.path.join('movie_dataset', 'movie_lines.tsv'), sep='\t', header=None,
            names=custom_header_lines, on_bad_lines='skip')
            
        self.max_seq_len = max_seq_len
        self.unk_token = self.UNK
        self.end_token = self.END_TOKEN
        self.start_token = self.START_TOKEN

        if voc_init:
            self.voc = self.create_vocab()
            with open("vocabulary.pkl", "wb") as file:
                pickle.dump(self.voc, file)
        else:
            self.load_vocab()

    def create_vocab(self):
        movie_lines = self.lines_data['line']
        cleaned_movies = movie_lines.dropna()
        cleaned_movies_lines = cleaned_movies.tolist()

        vocab = []
        voc_idx = OrderedDict()

        for line in cleaned_movies_lines:
            # get the list of trimmed tokens - get rid of non-letter chars
            try:
                phrase_trimmed = re.sub('\W+', ' ', line).split()
                for word in phrase_trimmed:
                    if word not in vocab:
                        vocab.append(word)
            except:
                print(line)

        vocab.append(self.end_token)
        vocab.append(self.start_token)
        vocab.append(self.unk_token)

        return sorted(vocab)

    def load_vocab(self):
        path_vocab = os.path.join(os.getcwd(), 'vocabulary.pkl')
        if os.path.exists(path_vocab):
            # Load the object back from the file
            with open(path_vocab, "rb") as file:
                self.voc = pickle.load(file)
        return

    def load_dialogue_text(self, dialogue):
        list_lines_text = dialogue['lines']
        list_lines = re.findall(r'\b\w+\b', list_lines_text)
        movie_id = dialogue['movie']
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
        return pairs, lines_dict

    def trim_sentence(self, tokenized_k):
        if len(tokenized_k) > self.max_seq_len:
            tokenized_k = tokenized_k[:self.max_seq_len]
            tokenized_k.append(self.end_token)
        elif len(tokenized_k) <= self.max_seq_len:
            tokenized_k.append(self.end_token)
            seq_pad = (self.max_seq_len - len(tokenized_k) + 1) * [self.unk_token]
            if len(seq_pad) > 0:
                tokenized_k.extend(seq_pad)
        return tokenized_k

    # loads one full dialogue (K phrases in a dialogue), an OrderedDict
    # If there are a total of K phrases, the data point will be with dimensions
    # ((K-1) x (MAX_SEQ + 2), (K-1) x (MAX_SEQ + 2))
    # zero-pad after EOS
    def load_dialogue(self, dialogue):
        dialogue_text, dialogue_dict = self.load_dialogue_text(dialogue)
        try:
            # k: phrase
            all_inputs = []
            all_outputs = []
            all_inputs_len = []
            all_outputs_len = []
            # get keys (first phrase) from the dialogues
            keys = dialogue_dict.keys()
            for k in keys:
                # tokenize here, both key and reply
                tokenized_k = re.sub('\W+', ' ', k).split()
                tokenized_r = re.sub('\W+', ' ', dialogue_dict[k]).split()
                all_inputs_len.append(torch.tensor(min(self.max_seq_len,len(tokenized_k))))
                all_outputs_len.append(torch.tensor([min(self.max_seq_len,len(tokenized_r))]))
                # pad or truncate, both key and reply
                tokenized_k = self.trim_sentence(tokenized_k)
                tokenized_r = self.trim_sentence(tokenized_r)

                # Convert to indices - query
                input_phrase = [self.voc.index(w) for w in tokenized_k]
                input_phrase.insert(0, self.voc.index(self.start_token))

                output_phrase = [self.voc.index(w) for w in tokenized_r]
                output_phrase.insert(0, self.voc.index(self.start_token))

                # append to the inputs and outputs - queries and replies
                all_inputs.append(torch.tensor(input_phrase))
                all_outputs.append(torch.tensor(output_phrase))
                
            all_inputs = torch.stack(all_inputs)
            all_outputs = torch.stack(all_outputs)
            # return a tuple
            output_tuple = (all_inputs, all_outputs)
            output_tuple_len = (all_inputs_len, all_outputs_len)

            return  output_tuple_len, output_tuple
        except:
            return (torch.tensor([]), torch.tensor([])), (torch.tensor([]), torch.tensor([]))



    # number of dialogues, 83097
    def __len__(self):
        return len(self.movies_data)

    # x: query
    # y: reply
    # idx: index of the dialogue
    # output: tuple of two torch tensor stacks dimensions ((K-1)x max_seq_len)
    def __getitem__(self, idx):
        self.dialogue = self.movies_data.iloc[idx]
        self.phrases_len, self.phrases = self.load_dialogue(self.dialogue)
        return self.phrases_len, self.phrases


def run():
    data = MoviePhrasesData(voc_init=False, max_seq_len=10)
    print(data[0])

if __name__ == '__main__':
    run()