# This file loads the traind models and defines a function to conversate with the agent
# Currently it braks: no checkpoint to load, f

#%% 
import torch
import os
from models import EncoderGRU, DecoderGRU
from dataset import MoviePhrasesData
from utils import read_settings


# Read settings from the YAML file
settings = read_settings("config.yaml")
data_settings = settings.get('data', {})
model_settings = settings.get('model', {})

# Load the checkpoint (specify the path accordingly)
checkpoint = torch.load("Baseline_LSTM_ckpt_.pth", map_location=torch.device('cpu'))

moviephrasesdata = MoviePhrasesData(voc_init=False, max_seq_len=data_settings['max_seq'])
voc = moviephrasesdata.vocab


# Initialize the models
encoder = EncoderGRU(vocab_size=len(voc), embedding_size=model_settings['embedding_dim'], hidden_size=model_settings['hidden_dim']) 
decoder = DecoderGRU(vocab_size=len(voc), embedding_size=model_settings['embedding_dim'], hidden_size=model_settings['hidden_dim']) 
# Load the model states
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

# %% A function like this, not necessarily this.
# def chat(input_text):
#     # Preprocess the input_text: tokenize, convert to tensor, etc.
#     # This heavily depends on how your data was processed during training
#     input_tensor = ... 

#     # Encode the input
#     with torch.no_grad():  # No need to track gradients
#         encoder_output, encoder_hidden = encoder(input_tensor)

#     # Create initial decoder input (typically the start-of-sequence token)
#     decoder_input = torch.tensor([[SOS_token]], device=device)  # Adjust SOS_token accordingly
#     decoder_hidden = encoder_hidden  # Depending on your architecture, might need adjustment

#     decoded_words = []
#     for di in range(max_length):  # max_length is a predefined limit to avoid infinite loops
#         with torch.no_grad():
#             decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#             topv, topi = decoder_output.data.topk(1)  # Get the most likely next word
#             if topi.item() == EOS_token:  # Adjust EOS_token accordingly
#                 break
#             else:
#                 decoded_words.append(topi.item())

#             decoder_input = topi.squeeze().detach()

#     # Convert the decoded sequence of IDs back to words
#     output_sentence = ' '.join([voc.index2word[id] for id in decoded_words])
#     return output_sentence