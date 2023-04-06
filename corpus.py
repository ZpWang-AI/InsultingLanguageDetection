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

cls_list = ['hd', 'cv', 'vo']
# hd: Assaults on Human Dignity 侵犯人类尊严
# cv: Call for Violence 呼吁暴力
# vo: Vulgarity/Ofensive Language directed at an individual 针对个人的粗俗/冒犯性语言


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


class CustomDataset(Dataset):
    def __init__(self, data:np.ndarray, config:CustomConfig) -> None:
        super().__init__()
        self.config = config
        
        if config.share_encoder:
            self.data = []
            for sentence, label1, label2, label3 in data:
                self.data.append([sentence, list(map(int, [label1,label2,label3]))])    
        else:
            sentences = data[:,0]
            label = np.zeros(data.shape[0], dtype=int)
            for p_cls in range(3):
                if cls_list[p_cls] in config.cls_target:
                    label |= np.int64(data[:,p_cls+1])
            self.data = list(zip(sentences, label))
    
    def is_positive(self, piece):
        if self.config.share_encoder:
            return bool(sum(piece[1]))
        else:
            return bool(piece[1])
    
    def downsample(self, positive_ratio):
        positive_cnt = sum(self.is_positive(p)for p in self.data)
        negative_cnt = int(positive_cnt/positive_ratio*(1-positive_ratio))
        
        np.random.shuffle(self.data)
        new_data = []
        for p in self.data:
            if self.is_positive(p):
                new_data.append(p)
            else:
                if negative_cnt:
                    new_data.append(p)
                    negative_cnt -= 1
        self.data = new_data
        np.random.shuffle(self.data)
        pass
    
    def get_data_info(self, info=''):
        if self.config.share_encoder:
            positive_cnts = [0,0,0]
            for _, (l1, l2, l3) in self.data:
                positive_cnts[0] += l1
                positive_cnts[1] += l2
                positive_cnts[2] += l3
            return f'{info:5s} data info: positive{positive_cnts}, total[{len(self.data)}]'
        else:
            positive_cnt = sum(self.is_positive(p)for p in self.data)
            return f'{info:5s} data info: positive[{positive_cnt}], total[{len(self.data)}]'
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x, y = self.data[index]
        return str(x), torch.tensor(y)
        

if __name__ == '__main__':

    sample_train_data = preprocess_train_data(train_data_file_list[0])
    sample_test_data = preprocess_test_data(test_data_file_list[0])

    # print(sample_train_data[:3])
    # print(sample_test_data[:3])
    # exit()
    
    # for line in sample_train_data[1:3]:
    #     print(line)

    sample_config = CustomConfig()
    sample_config.share_encoder = False
    sample_config.cls_target = 'hd+cv'
    sample_config.positive_ratio = 0.2
    
    sample_train_data = CustomDataset(sample_train_data, sample_config)
    print(sample_train_data.get_data_info('init'))
    sample_train_data.downsample(sample_config.positive_ratio)
    print(sample_train_data.get_data_info('down'))
    
    sample_train_data = DataLoader(sample_train_data, batch_size=5, shuffle=False)
    for sample_x, sample_y in sample_train_data:
        print(sample_x)
        print(sample_y)
        print()
        break
    # exit()
    
    from model.baseline import BaselineModel
    sample_model = BaselineModel(sample_config)
    sample_criterion = nn.CrossEntropyLoss(reduction='sum')
    for sample_x, sample_y in sample_train_data:
        sample_output = sample_model(sample_x)
        print(sample_output.shape)
        print(sample_y.shape)
        loss = sample_criterion(sample_output.view(-1, 2), sample_y.view(-1))
        print(loss)
        loss.backward()
        break
    
    # pass
    