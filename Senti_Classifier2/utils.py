import yaml
import torch
import argparse
import pandas as pd
from scipy.io import arff
from torch.utils.data import DataLoader, random_split
import xml.etree.ElementTree as ET


input_arff_path = 'credit_card_data/credit_card_data.arff'
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def arff_file_to_csv(file_path):
    output_csv_path = 'credit_card_data/credit_card_data.csv'
    data, meta = arff.loadarff(input_arff_path)
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    return


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


def print_model_size(model):
    total_pars = 0
    for _n, _par in model.state_dict().items():
        total_pars += _par.numel()
    print(f"Total number of parameters: {total_pars}")
    return


def save_checkpoint(epoch, model, model_name, optimizer):
    ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
    torch.save(ckpt, f"{model_name}_ckpt_{str(epoch)}.pth")


def load_checkpoint(model, file_name):
    ckpt = torch.load(file_name, map_location=device)
    model_weights = ckpt['model_weights']
    model.load_state_dict(model_weights)
    print("Model's pretrained weights loaded!")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings


def parse_xml_to_dict(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    result_dict = {'root': {}}  # Dictionary to store the XML structure

    def parse_element(element):
        element_dict = {}

        for child in element:
            child_dict = parse_element(child)
            if child.tag in element_dict:
                if type(element_dict[child.tag]) is list:
                    element_dict[child.tag].append(child_dict)
                else:
                    element_dict[child.tag] = [element_dict[child.tag], child_dict]
            else:
                element_dict[child.tag] = child_dict

        if element.attrib:
            element_dict.update({'attributes': element.attrib})

        if element.text:
            element_dict['text'] = element.text.strip()

        return element_dict

    result_dict['root'] = parse_element(root)
    return result_dict