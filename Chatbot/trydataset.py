import os
import json
import random
import unicodedata
import re
import torch
import itertools

# Constants
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

corpus_name = "movie-corpus"
corpus = os.path.join('data', corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

# Utility functions
def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Loading and processing data
def loadLinesAndConversations(fileName):
    lines = {}
    conversations = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            lineJson = json.loads(line)
            lineObj = {
                "lineID": lineJson["id"],
                "characterID": lineJson["speaker"],
                "text": lineJson["text"]
            }
            lines[lineObj['lineID']] = lineObj

            if lineJson["conversation_id"] not in conversations:
                convObj = {
                    "conversationID": lineJson["conversation_id"],
                    "movieID": lineJson["meta"]["movie_id"],
                    "lines": [lineObj]
                }
            else:
                convObj = conversations[lineJson["conversation_id"]]
                convObj["lines"].insert(0, lineObj)
            conversations[convObj["conversationID"]] = convObj

    return lines, conversations

def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations.values():
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

# Vocabulary class
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = [k for k, v in self.word2count.items() if v >= min_count]

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

# Functions to save and load Voc object
def saveVoc(voc, file_path):
    with open(file_path, 'w') as f:
        json.dump(voc.__dict__, f)

def loadVoc(file_path):
    voc = Voc(corpus_name)
    with open(file_path, 'r') as f:
        voc.__dict__ = json.load(f)
    return voc

# Data conversion and batching
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

# Function to create train-test split
def createTrainTestSplit(conversations, split_ratio=0.8):
    conversation_ids = list(conversations.keys())
    random.shuffle(conversation_ids)
    split_index = int(len(conversation_ids) * split_ratio)
    train_ids = conversation_ids[:split_index]
    test_ids = conversation_ids[split_index:]

    train_conversations = {id: conversations[id] for id in train_ids}
    test_conversations = {id: conversations[id] for id in test_ids}

    return train_conversations, test_conversations

def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p, MAX_LENGTH):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using the ``filterPair`` condition
def filterPairs(pairs, MAX_LENGTH):
    return [pair for pair in pairs if filterPair(pair, MAX_LENGTH)]

def loadPrepareData(corpus, corpus_name, datafile, save_dir, MAX_LENGTH=10, init=False):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs, MAX_LENGTH)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

# Main script to process and save data
if __name__ == "__main__":
    lines, conversations = loadLinesAndConversations(os.path.join(corpus, "utterances.jsonl"))
    train_conversations, test_conversations = createTrainTestSplit(conversations)
    
    train_pairs = extractSentencePairs(train_conversations)
    test_pairs = extractSentencePairs(test_conversations)

    save_dir = os.path.join("data", "splits")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    voc, train_pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir, MAX_LENGTH=10)
    _, test_pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir, MAX_LENGTH=10)
    
    # Save conversations and pairs to separate files
    with open(os.path.join(save_dir, "train_conversations.json"), 'w') as f:
        json.dump(train_conversations, f)
    with open(os.path.join(save_dir, "test_conversations.json"), 'w') as f:
        json.dump(test_conversations, f)
    with open(os.path.join(save_dir, "train_pairs.json"), 'w') as f:
        json.dump(train_pairs, f)
    with open(os.path.join(save_dir, "test_pairs.json"), 'w') as f:
        json.dump(test_pairs, f)
    
    # Save vocabulary
    saveVoc(voc, os.path.join(save_dir, "voc.json"))

    print(f"Number of conversations in the training set: {len(train_conversations)}")
    print(f"Number of conversations in the test set: {len(test_conversations)}")

