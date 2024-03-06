import torch
from torch.nn import CrossEntropyLoss as CrossEntropyLoss
from torch.utils.data import DataLoader as DataLoader
from dataset import DialogData
from torch.utils import data
from logger import Logger
from torch.utils.data import random_split
from utils import parse_arguments, read_settings

def train(model_settings, train_settings):
    dataset = DialogData(voc_init=False, max_seq=10)
    # print(dataset)
    
if __name__ == '__main__':
    args = parse_arguments()

    # Read settings from the YAML file
    settings = read_settings(args.config)

    # Access and use the settings as needed
    model_settings = settings.get('model', {})
    train_settings = settings.get('train', {})
    print(model_settings)
    train(model_settings, train_settings)
