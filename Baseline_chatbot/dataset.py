# %%
import os
import torch
import pickle
import torch.utils
import pandas as pd
from torch.utils import data
from collections import Counter, OrderedDict

from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# jaxtype pleasseee
import re
import sentencepiece as spm
import ast


class MoviePhrasesData(data.Dataset):

    # voc: vocabulary, word:idx
    # all dialogues:
    def __init__(self, max_seq_len=15, voc_init=False):

        self.movies_data = pd.read_csv(
            os.path.join("Filtered Datasets", "filtered_movie_conversations.tsv"),
            sep="\t",
        )  # reads the tsv movie conversations file

        # convert the string of line ids to a list of line ids ["L598", "L599", "L600", "L601"]
        self.movies_data['lines'] = self.movies_data['lines'].apply(lambda x: ast.literal_eval(x))

        self.lines_data = pd.read_csv(
            os.path.join("Filtered Datasets", "filtered_movie_lines.tsv"),
            sep="\t",
        )
        # reads the tsv movie_lines file

        self.max_seq_len = max_seq_len

        if voc_init:
            self.voc = self.train_tokenizer()
        else:
            self.load_tokenizer()
        
        self.vocab = self.tokenizer.vocab  #unsorted list

    # def _create_raw_text_file(self):
    #     movie_lines = self.lines_data["line"]
    #     cleaned_movies = movie_lines.dropna()
    #     cleaned_movies_lines = cleaned_movies.tolist()  # list of all interventions

    #     # Create a raw text file from the movie lines
    #     with open("movie_lines.txt", "w", encoding="utf-8") as file:
    #         for line in cleaned_movies_lines:
    #             file.write(f"{line}\n")

    
    def load_tokenizer(self):
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join("GRU_tokenizer"))
    
    def train_tokenizer(
        self, vocab_size=18000
    ):
        """
        Trains a SentencePiece tokenizer on the text of Cornell's dataset.
        Text is processed: each intervention is a string elements of the list.

        :param input_text_file: Path to the file containing text to train the tokenizer.
        :param vocab_size: Size of the vocabulary.
        """
        movie_lines = self.lines_data["line"]
        cleaned_movies = movie_lines.dropna()
        cleaned_movies_lines = cleaned_movies.tolist()  # list of all interventions

        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train_from_iterator(
            cleaned_movies_lines,
            vocab_size=vocab_size,  # Desired vocabulary size
            min_frequency=5,  # Minimum frequency for a token to be included
            show_progress=True,
            limit_alphabet=500,  # Limits the number of initial characters to consider
        )
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        special_tokens_dict = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        } # added later
        tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer = tokenizer

        self.tokenizer.save_pretrained(os.path.join("GRU_tokenizer"))

    def load_dialogue_text(self, dialogue):
        """
        From a dialogue --movie and some lines defining it.
        (Think about it as one line of the movie_conversations.tsv)
        Outputs:
         dict: dictionary mapping a line to the next one
        """
        list_lines = dialogue["lines"]  # string of line ids like "[L598 L599 L600 L601]"
        movie_id = dialogue["movie"]
        lines_dict = {}
        for i in range(len(list_lines) - 1):
            # go over the list except the last element
            #obtain line text from its id
            line_text = self.lines_data.loc[
                (self.lines_data["line_id"] == list_lines[i])
                & (self.lines_data["movie_id"] == movie_id)
            ] # dataframes
            next_line_text = self.lines_data.loc[
                (self.lines_data["line_id"] == list_lines[i+1])
                & (self.lines_data["movie_id"] == movie_id)] # dataframes
            lines_dict[line_text["line"].iloc[0]] = next_line_text["line"].iloc[0]
        return lines_dict
    
    def trim_or_pad_sentence(self, tokenized_sentence):
        """
        Trims or pads a sentence to max_seq_len.
        Tokenized_sentence is a list of token ids (integers).
        Adds bos and eos tokens to the sentence (which tok doesn't do by default)
        -> that's why they're trimmed at max_seq_len - 2
        """
        if len(tokenized_sentence) > self.max_seq_len - 2:
            format_sentence = [self.tokenizer.bos_token_id] + tokenized_sentence[: self.max_seq_len - 2] + [self.tokenizer.eos_token_id]
        else:
            format_sentence = [self.tokenizer.bos_token_id] + tokenized_sentence + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id] * (
                self.max_seq_len - len(tokenized_sentence) - 2
            )
        return torch.Tensor(format_sentence)

    def load_dialogue(self, dialogue):
        """
        For a dialogue with K interventions,
        returns a tuple of 2 tensors each with dimensions ((K-1)x max_seq_len),
        representing the queries and the replies.
        Max_seq_len is the maximum number of tokens in a sentence,
        so each sentences is trimmed or padded beforehand.
        Sentences also need to be tokenized.
        """
        lines_dict = self.load_dialogue_text(dialogue)
        dict_len = len(lines_dict)
        all_inputs = torch.zeros(dict_len, self.max_seq_len, dtype=torch.long)
        all_outputs = torch.zeros(dict_len, self.max_seq_len, dtype=torch.long)
        for idx, (line, next_line) in enumerate(lines_dict.items()):
            tokenized_line = self.tokenizer.encode(line)
            tokenized_next_line = self.tokenizer.encode(next_line)
            all_inputs[idx] = self.trim_or_pad_sentence(tokenized_line)
            all_outputs[idx] = self.trim_or_pad_sentence(tokenized_next_line)
        return all_inputs, all_outputs

    # number of dialogues, 83097
    def __len__(self):
        return len(self.movies_data)
    # the dataset is just the dialogues in movie_conversations.tsv

    # x: query
    # y: reply
    # idx: index of the dialogue
    # output: tuple of two torch tensor stacks dimensions ((K-1)x max_seq_len)
    def __getitem__(self, idx):
        self.dialogue = self.movies_data.iloc[idx]  # data for that row in movie_conversations.tsv
        self.phrases = self.load_dialogue(self.dialogue)  
        return idx, self.phrases
    # for every dialogue in the movie_conversations.tsv,
    # return the dialogue index and the tensors for (all_inputs, all_outputs) 

# %%
    

"""
My TEST:
for i in tqdm(range(len(dataset))):
    dataset.load_dialogue(dataset.movies_data.iloc[i])
-> with the new dataset this gives no error
"""

 """
 EXAMPLE TO ENCODE BATCHES!
 
 encoded_batch = transformer_tokenizer.encode_batch(
    ["Your first sentence.", "Your second, much longer, sentence."],
    padding=True,  # Enable padding
    max_length=10,  # Specify the desired maximum length
    truncation=True,  # Enable truncation to max_length
    return_tensors="pt"  # Return PyTorch tensors
)"""

"""
EXAMPLE TO ENCODE SINGLE SENTENCE!
inputs = tokenizer("Hello, world!", padding=True, truncation=True, return_tensors="pt")

"""