#%%
import torch
from torch import optim
import torch.nn as nn
import os
import json
import random
from trydataset import loadVoc, batch2TrainData
from models import EncoderRNN, LuongAttnDecoderRNN
from train_utils import trainIters

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Device: {device}")

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# Paths to preprocessed data files
save_dir = os.path.join("data", "checkpoints")
train_conversations_file = os.path.join(save_dir, "train_conversations.json")
test_conversations_file = os.path.join(save_dir, "test_conversations.json")
train_pairs_file = os.path.join(save_dir, "train_pairs.json")
test_pairs_file = os.path.join(save_dir, "test_pairs.json")
voc_file = os.path.join(save_dir, "voc.json")

# Load vocabulary and pairs from JSON files
def loadJsonData(file):
    with open(file, 'r') as f:
        return json.load(f)

train_pairs = loadJsonData(train_pairs_file)
test_pairs = loadJsonData(test_pairs_file)
voc = loadVoc(voc_file)

# Model configuration
model_name = 'LSTM_noatt_dot'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = 4000
loadFilename = os.path.join(save_dir, model_name, '{}_checkpoint.tar'.format(checkpoint_iter))

# Load model if a `loadFilename` is provided
if os.path.exists(loadFilename):
    checkpoint = torch.load(loadFilename)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
    # Checkpoint detected

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout, rnn_cell='LSTM')
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='LSTM')
if os.path.exists(loadFilename):
    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # Checkpoint loaded

# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 100
save_every = 10

config = {
    "lr": 0.0001,
    "clip": 50,
    "teacher_forcing_ratio": 1.0,
    "decoder_learning_ratio": 5.0,
    "n_iteration": 100,
    "train_batches_per_val": 80,
    "val_batches_per_epoch": 20,
    "save_every": 10
}

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if os.path.exists(loadFilename):
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have CUDA, configure CUDA to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

# Prepare training and validation data batches
# Prepare training and validation data batches
def prepareDataBatches(pairs, voc, batch_size):
    batches = []
    for _ in range(n_iteration):
        batch = [random.choice(pairs) for _ in range(batch_size)]
        batches.append(batch2TrainData(voc, batch))
    return batches

training_batches = prepareDataBatches(train_pairs, voc, batch_size)
validation_batches = prepareDataBatches(test_pairs, voc, batch_size)


# Run training iterations with validation
print("Starting Training!")


# trainIters(model_name, voc, training_batches, validation_batches, encoder, decoder, encoder_optimizer, decoder_optimizer,
#            embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
#            print_every, save_every, clip, teacher_forcing_ratio, loadFilename)



trainIters(model_name, voc, training_batches, validation_batches, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, config['n_iteration'], batch_size,
           config['train_batches_per_val'], config['val_batches_per_epoch'], config['save_every'],
           config['clip'], config['teacher_forcing_ratio'], loadFilename)

# %%
