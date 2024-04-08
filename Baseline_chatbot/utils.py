import yaml
import torch
import argparse
import pandas as pd
from scipy.io import arff
from torch.utils.data import DataLoader, random_split
import xml.etree.ElementTree as ET

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

def split_data(dataset, batch_size):
    total_size = len(dataset)
    train_size = int(0.75 * total_size)  # 75% for training
    val_size = int(0.15 * total_size)  # 15% for validation
    test_size = total_size - train_size - val_size  # Remaining 10% for testing

    # Split the dataset into train, validation, and test sets
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def save_checkpoint(epoch, model, model_name, optimizer, test_acc, valid_acc):
    ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 
            'test_acc': test_acc, 'valid_acc': valid_acc}
    torch.save(ckpt, f"{model_name}_ckpt_.pth")


def load_checkpoint(model, optimizer, test_acc, valid_acc, file_name):
    ckpt = torch.load(file_name, map_location=device)
    # model_weights = ckpt['model_state_dict']
    epoch = ckpt['epoch']
    test_acc = ckpt['test_acc']
    valid_acc = ckpt['valid_acc']
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    
    return epoch+1, test_acc, valid_acc

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='/config.yaml', help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings

