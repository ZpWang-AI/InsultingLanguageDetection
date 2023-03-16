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

from config import get_default_config

# warnings.filterwarnings("ignore")


train_data_path = './data/ghc_train.tsv'
test_data_path = './data/ghc_test.tsv'


def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        content = list(csv.reader(csvfile))
        # reader = csv.reader(csvfile)
        # for row in reader:
        #     content.append(row)
    return content


class CustomDataset(Dataset):
    def __init__(self, data, config) -> None:
        super().__init__()
        self.data = data
        self.config = config
    
    def __len__(self):
        return len(self.data)
    
    def deal_sentence(self, sentence:str):
        return sentence
        sentence = sentence.strip().split()
        ans_sentence = []
        for word in sentence:
            if word[0] != '@':
                ans_sentence.append(word)      
        return ' '.join(ans_sentence)
    
    def __getitem__(self, index):
        sentence1, sentence2, label = self.data[index]
        if self.config.base:
            return f'{sentence1}\n{sentence2}', label
        elif self.config.clip:
            return self.deal_sentence(sentence1), self.deal_sentence(sentence2), label
        else:
            raise 'wrong config'    
        
        

if __name__ == '__main__':

    sample_train_csv = read_csv(train_data_path)
    sample_test_csv = read_csv(test_data_path)
    
    for line in sample_train_csv[:3]:
        print(line)
    print()
    for line in sample_test_csv[:3]:
        print(line)
    
    # # print(deal_train_data())
    # # print(deal_test_data())
    # sample_config = get_default_config()
    # sample_train_data = deal_train_data()
    # sample_train_data = CustomDataset(sample_train_data, sample_config)
    # sample_train_data = DataLoader(sample_train_data, batch_size=3, shuffle=False)
    # for sample_input in sample_train_data:
    #     print(sample_input)
    #     print()
    #     break
    # # exit()
    
    # from model.xlm_roberta import BertModel
    # sample_model = BertModel(sample_config)
    # sample_criterion = nn.CrossEntropyLoss(reduction='sum')
    # for sample_x, sample_y in sample_train_data:
    #     sample_output = sample_model(sample_x)
    #     print(sample_output)
    #     loss = sample_criterion(sample_output, sample_y.to(torch.long))
    #     print(loss)
    #     loss.backward()
    #     break
    
    # pass
    