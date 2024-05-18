#%%
# perplexity calculation
import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models import EncoderRNN, Attn, LuongAttnDecoderRNN, GreedySearchDecoder, SimpleDecoderRNN
from utils import parse_arguments, read_settings, load_checkpoint, save_checkpoint
import os
from dataset import loadPrepareData, Voc, indexesFromSentence
import torch.nn as nn
import math

EOS_token = 2
SOS_token = 1
# def calculate_perplexity(model, tokenizer, text):
#     encodings = tokenizer(text, return_tensors='pt')
#     max_length = model.config.n_positions
#     stride = 512

#     lls = []
#     for i in range(0, encodings.input_ids.size(1), stride):
#         begin_loc = max(i + stride - max_length, 0)
#         end_loc = min(i + stride, encodings.input_ids.size(1))
#         trg_len = end_loc - i
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
#         target_ids = input_ids.clone()
#         target_ids[:, :-trg_len] = -100

#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)
#             log_likelihood = outputs.loss * trg_len

#         lls.append(log_likelihood)

#     ppl = torch.exp(torch.stack(lls).sum() / end_loc)
#     return ppl.item()

# Example usage
# model_name = 'gpt2'
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# text = "Your test dataset text here."
# print("Perplexity: ", calculate_perplexity(model, tokenizer, text))
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
args = parse_arguments()
filepath=os.path.dirname(os.path.realpath(__file__))
settings = read_settings(filepath+args.config)
model_settings = settings.get('model_ende', {})
data_settings = settings.get('data_ende', {})
corpus_name = "movie-corpus"
corpus = os.path.join(filepath,'data',corpus_name)
save_dir_corpus = os.path.join("data", "movie-corpus")

datafile = os.path.join(corpus, "formatted_movie_lines.txt")

voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir_corpus, data_settings['max_seq'])

attn_model = 'dot'
#``attn_model = 'general'``
#``attn_model = 'concat'``
hidden_size = model_settings['hidden_dim']
encoder_n_layers = model_settings['encoder_num_layers']
decoder_n_layers = model_settings['decoder_num_layers']
dropout = model_settings['dropout']
batch_size = data_settings['batch_size']
# embedding = nn.Embedding(voc.num_words, hidden_size)
embedding = nn.Embedding(7836, hidden_size)

if(model_settings['lstm']):
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout, rnn_cell='LSTM') #GRU
    if(model_settings['use_attention']):
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='LSTM')
    else:
        decoder = SimpleDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='LSTM')
else:
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout, rnn_cell='GRU') #GRU

    if(model_settings['use_attention']):
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='GRU') 
    else:
        decoder = SimpleDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout, rnn_cell='GRU')
save_dir = os.path.join("data", "checkpoints")
ende_mode = 'LSTM' if model_settings['lstm'] else 'GRU'
attn_mode = 'Att' if model_settings['use_attention'] else 'NoAtt' 
attn_method_mode_list = ['dot', 'general', 'concat']
attn_method_mode = attn_method_mode_list[model_settings['attn_method']]
checkpoint_iter = model_settings['checkpoint_iter']

model_name = f'{ende_mode}_{attn_mode}_{attn_method_mode}'

loadFilename = os.path.join(filepath, save_dir, model_name,
                    '{}_checkpoint.tar'.format(checkpoint_iter))
# loadFilename = "C:\Users\trans\Documents\VICTOR DOCUMENTS\Victor now is doing a master\INM706 DL4Sequences\Sentimental-Chatbot\Chatbot\data\checkpoints\LSTM_Att_dot\200000_checkpoint.tar"

if os.path.exists(loadFilename):
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename, map_location = device)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
    # Checkpoint detected
if os.path.exists(loadFilename):
    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    print('Loaded encoder and decoder!')

# %%
def calculate_perplexity(encoder, decoder, searcher, voc, sentence, max_length=10):
    # Format input sentence as a batch
    encoder.eval()
    decoder.eval()
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")

    # Forward input through encoder model
    encoder_outputs, encoder_hidden = encoder(input_batch, lengths)
    
    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    if isinstance(encoder_hidden, tuple):  # LSTM
        decoder_hidden = (
            torch.mean(encoder_hidden[0].view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1),
            torch.mean(encoder_hidden[1].view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1)
        )
    else:  # GRU
        decoder_hidden = torch.mean(encoder_hidden.view(encoder.n_layers, 2, -1, encoder.hidden_size), dim=1)
        
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
    total_log_prob = 0
    num_tokens = 0

    for _ in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        token = topi.item()
        log_prob = topv.item()
        total_log_prob += log_prob
        num_tokens += 1
        
        if token == EOS_token:
            break

        decoder_input = torch.unsqueeze(topi, 0)

    perplexity = math.exp(-total_log_prob / num_tokens)
    return perplexity

# Example usage
searcher = GreedySearchDecoder(encoder, decoder)

sentence = "hello friend how is your life"
ppl = calculate_perplexity(encoder, decoder, searcher, voc, sentence)
print("Perplexity: ", ppl)


# %%
# from transformers import AutoModel, AutoTokenizer
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def embed_text(model, tokenizer, text):
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze()

# def calculate_similarity(model, tokenizer, text1, text2):
#     embedding1 = embed_text(model, tokenizer, text1)
#     embedding2 = embed_text(model, tokenizer, text2)
#     similarity = cosine_similarity([embedding1], [embedding2])
#     return similarity[0][0]

# # Example usage
# model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# text1 = "Your input text here."
# text2 = "Your output text here."
# similarity = calculate_similarity(model, tokenizer, text1, text2)
# print("Semantic Similarity: ", similarity)
