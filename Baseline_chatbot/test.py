# This file loads the traind models and defines a function to conversate with the agent
# Currently it braks: no checkpoint to load, f

#%% 
import torch
import os
from models import EncoderGRU, DecoderGRU
from dataset import MoviePhrasesData
from utils import read_settings




# %% A function like this, not necessarily this.
def chat(input_text: str):
    # Read settings from the YAML file
    settings = read_settings("config.yaml")
    data_settings = settings.get('data', {})
    model_settings = settings.get('model', {})

    device = (torch.device("cuda") if not torch.cuda.is_available() else torch.device("cpu"))
    # Load the checkpoint (specify the path accordingly)
    checkpoint = torch.load("model_and_optimizer_2.pth", map_location=device)

    dataset = MoviePhrasesData(voc_init=False, max_seq_len=data_settings['max_seq'])
    voc = dataset.vocab

    # Initialize the models
    encoder = EncoderGRU(vocab_size=len(voc), embedding_size=model_settings['embedding_dim'], hidden_size=model_settings['hidden_dim']) 
    decoder = DecoderGRU(vocab_size=len(voc), embedding_size=model_settings['embedding_dim'], hidden_size=model_settings['hidden_dim']) 
    # Load the model states
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder=encoder.to(device)
    decoder=decoder.to(device)

    encoder.eval()
    decoder.eval()
    input_tensor = dataset.tokenizer.encode(input_text)
    input_tensor = dataset.trim_or_pad_sentence(input_tensor)

    # Encode the input
    encoder_input = input_tensor.view(-1, 1, dataset.max_seq_len)[0]
    with torch.no_grad():  # No need to track gradients
        encoder_output, encoder_hidden = encoder(encoder_input)

    # Create initial decoder input (typically the start-of-sequence token)
    decoder_input = torch.tensor(dataset.tokenizer.encode(dataset.bos_token),
                                  device=device)  #  BOS_token
    decoder_hidden = encoder_hidden[0] 

    decoded_words = []
    for di in range(dataset.max_seq_len):  # max_length is a predefined limit to avoid infinite loops
        with torch.no_grad():
            # print(f"decoder_input shape is {decoder_input.shape}, decoder_hidden shape is {decoder_hidden.shape}")
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            # Print input and output
            # print('Prompt:', input_text)
            # print(decoder_output)
            # print(f" {di} decoder_output.data shape is {decoder_output.data.shape} and decoder_hidden shape is {decoder_hidden.shape}")
            topv, topi = decoder_output.topk(1)  # Get the most likely next word
            # print(topi)
            # print('Predicted Response:', [dataset.tokenizer.decode(x) for x in topi])
            #print(f"topv is {topv} and topi is {topi}")
            if topi.item() == 18001:  # EOS_token
                print("yup")
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.view(1, -1)[0]
    # add some padding to the output if necessary (not implemented yet)
    # Convert the decoded sequence of IDs back to words
    output_sentence = ' '.join([dataset.tokenizer.decode(id) for id in decoded_words])
    return output_sentence
# %%

print(chat("How is your mother today"))