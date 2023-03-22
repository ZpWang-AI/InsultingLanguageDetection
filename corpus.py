import csv
import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from typing import *
from pathlib import Path as path

from config import CustomConfig

# warnings.filterwarnings("ignore")


train_data_file_list = [
    './data/ghc_train.tsv'
]
test_data_file_list = [
    './data/ghc_test.tsv'
]


def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        content = list(csv.reader(csvfile, delimiter='\t'))
        # reader = csv.reader(csvfile)
        # for row in reader:
        #     content.append(row)
    return content


def preprocess_train_data(train_data_file=train_data_file_list[0]):
    if train_data_file == train_data_file_list[0]:
        train_content = pd.read_csv(train_data_file, sep='\t')
        train_content = np.array(train_content)
        # train_data = [[str(p[0]), int(p[1]), int(p[2]), int(p[3])]for p in train_data]
        return train_content
    else:
        raise 'Wrong train data file'
    
def preprocess_test_data(test_data_file=test_data_file_list[0]):
    if test_data_file == test_data_file_list[0]:
        test_content = pd.read_csv(test_data_file, sep='\t')
        test_content = np.array(test_content)
        # test_data = [[str(p[0]), int(p[1]), int(p[2]), int(p[3])]for p in test_data]
        return test_content
    else:
        raise 'Wrong test data file'

def get_data_info(data, info=''):
    return ' '.join(map(str, [
        f'{info:5s} data info:',   
        np.sum(data[:,1:], axis=0), 
        len(data),
    ]))

def downsample_data(train_data, ratio=0.1):
    downsampled_data = []
    for piece in train_data:
        if np.sum(piece[1:]) != 0 or np.random.rand() < ratio:
            downsampled_data.append(piece)
    return np.stack(downsampled_data, axis=0)


class CustomDataset(Dataset):
    def __init__(self, data, config) -> None:
        super().__init__()
        self.data = data
        self.config = config
    
    def __len__(self):
        return len(self.data)
    
    def deal_sentence(self, sentence:str):
        return str(sentence)
        sentence = sentence.strip().split()
        ans_sentence = []
        for word in sentence:
            if word[0] != '@':
                ans_sentence.append(word)      
        return ' '.join(ans_sentence)
    
    def __getitem__(self, index):
        sentence, label1, label2, label3 = self.data[index]
        x = self.deal_sentence(sentence)
        y = torch.tensor(list(map(int, (label1, label2, label3))))
        return x, y
        

if __name__ == '__main__':

    sample_train_data = preprocess_train_data(train_data_file_list[0])
    sample_test_data = preprocess_test_data(test_data_file_list[0])
    
    print(get_data_info(sample_train_data, 'init'))
    print(sample_train_data.shape)
    sample_downsampled_train_data = downsample_data(sample_train_data)
    print(sample_downsampled_train_data.shape)
    print(get_data_info(sample_downsampled_train_data, 'downsampled'))
    exit()
    
    # print(sample_train_data[:3])
    # print(sample_test_data[:3])
    # exit()
    
    # for line in sample_train_data[1:3]:
    #     print(line)

    sample_config = CustomConfig()
    sample_train_data = CustomDataset(sample_train_data, sample_config)
    sample_train_data = DataLoader(sample_train_data, batch_size=5, shuffle=False)
    for sample_input in sample_train_data:
        print(sample_input)
        print()
        break
    # exit()
    
    from model.baseline import BertModel
    sample_model = BertModel(sample_config)
    sample_criterion = nn.CrossEntropyLoss(reduction='sum')
    for sample_x, sample_y in sample_train_data:
        sample_output = sample_model(sample_x)
        print(sample_output.shape)
        print(sample_y.shape)
        loss = sample_criterion(sample_output.view(-1, 2), sample_y.to(torch.long).view(-1))
        print(loss)
        loss.backward()
        break
    
    # pass
    