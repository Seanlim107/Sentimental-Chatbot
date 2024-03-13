import os
import torch
import pickle
import torch.utils
import pandas as pd
from torch.utils import data
from collections import Counter, OrderedDict

import re
import sentencepiece as spm


class MoviePhrasesData(data.Dataset):
    UNK = "<UNK>"
    START_TOKEN = "<S>"
    END_TOKEN = "</S>"

    # voc: vocabulary, word:idx
    # all dialogues:
    def __init__(self, max_seq_len=None, voc_init=False):

        custom_header = ["user0", "user1", "movie", "lines"]
        custom_header_lines = ["line_id", "user_id", "movie_id", "user_name", "line"]
        self.movies_data = pd.read_csv(
            os.path.join("movie_dataset", "movie_conversations.tsv"),
            sep="\t",
            header=None,
            names=custom_header,
        )  # reads the tsv movie conversations file
        self.lines_data = pd.read_csv(
            os.path.join("movie_dataset", "movie_lines.tsv"),
            sep="\t",
            header=None,
            names=custom_header_lines,
            on_bad_lines="skip",
        )
        # reads the tsv movie_lines file

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
        movie_lines = self.lines_data["line"]
        cleaned_movies = movie_lines.dropna()
        cleaned_movies_lines = cleaned_movies.tolist()  # list of all interventions

        vocab = []
        voc_idx = OrderedDict()  # not used

        for line in cleaned_movies_lines:
            # get the list of trimmed tokens - get rid of non-letter chars
            try:
                phrase_trimmed = re.sub("\W+", " ", line).split()
                # \W+ matches any non-word character -> they are replaced by a space
                # removes punctuation and other non-letter characters
                # => Chatbot:   are . and , important ?
                # split() splits the string into a list of words
                for word in phrase_trimmed:
                    if word not in vocab:
                        vocab.append(word)
                # adds the word to the vocabulary.
            except:
                print(line)

        vocab.append(self.end_token)
        vocab.append(self.start_token)
        vocab.append(self.unk_token)

        return sorted(vocab)

        # PROBLEMS:

    def load_vocab(self):
        path_vocab = os.path.join(os.getcwd(), "vocabulary.pkl")
        if os.path.exists(path_vocab):
            # Load the object back from the file
            with open(path_vocab, "rb") as file:
                self.voc = pickle.load(file)
        else:
            raise FileNotFoundError("Vocabulary file not found")
        return

    # evaluate whether it works  (properly merge with the rest of text)
    def train_sentencepiece_tokenizer(
        self, input_text_file, model_prefix, vocab_size=32000
    ):
        """
        Trains a SentencePiece tokenizer on the given text file.

        :param input_text_file: Path to the file containing text to train the tokenizer.
        :param model_prefix: Prefix for the output model and vocabulary files.
        :param vocab_size: Size of the vocabulary.
        """
        spm.SentencePieceTrainer.train(
            f"--input={input_text_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=0.9995 --model_type=bpe"
        )
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(f"{model_prefix}.model")

    def new_create_vocab(self):
        # Assuming you've already preprocessed your dataset into a single text file
        # where each line is a movie line.
        input_text_file = "movie_lines.txt"
        model_prefix = "movie_dialogue_tokenizer"
        vocab_size = 60000  # Adjust vocab size as needed

        self.train_sentencepiece_tokenizer(input_text_file, model_prefix, vocab_size)

        # At this point, the tokenizer is trained and loaded. You can use self.tokenizer
        # for tokenizing and detokenizing text.
