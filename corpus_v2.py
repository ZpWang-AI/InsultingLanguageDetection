import csv
import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from typing import *
from pathlib import Path as path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

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


class Datasetv2(Dataset):
    def __init__(self, 
                 data:np.ndarray, 
                 share_encoder=False, 
                 cls_target='hd+cv+vo',
                 tokenizer=None,
                 ) -> None:
        super().__init__()
        self.share_encoder = share_encoder
        self.cls_target = cls_target
        self.tokenizer = tokenizer
        
        self.x_list = data[:,0]
        
        if share_encoder:
            self.y_list = data[:,1:]
        else:
            self.y_list = np.zeros(data.shape[0], dtype=int)
            for p_cls in range(3):
                if cls_list[p_cls] in cls_target:
                    self.y_list |= np.int64(data[:,p_cls+1])
    
    def is_positive(self, y_):
        if self.share_encoder:
            return bool(sum(y_))
        else:
            return bool(y_)
    
    def downsample(self, positive_ratio):
        positive_cnt = sum(self.is_positive(p)for p in self.y_list)
        negative_cnt = int(positive_cnt/positive_ratio*(1-positive_ratio))
        
        data_id_list = list(range(len(self.x_list)))
        np.random.shuffle(data_id_list)
        new_x_list = []
        new_y_list = []
        for data_id in data_id_list:
            if self.is_positive(self.y_list[data_id]):
                new_x_list.append(self.x_list[data_id])
                new_y_list.append(self.y_list[data_id])
            else:
                if negative_cnt:
                    new_x_list.append(self.x_list[data_id])
                    new_y_list.append(self.y_list[data_id])
                    negative_cnt -= 1
        self.x_list = np.array(new_x_list)
        self.y_list = np.array(new_y_list)
    
    def get_data_info(self, info=''):
        if self.share_encoder:
            positive_cnts = np.sum(self.y_list, axis=0)
            return f'{info:5s} data info: positive{positive_cnts}, total[{len(self.x_list)}]'
        else:
            positive_cnt = np.sum(self.y_list)
            return f'{info:5s} data info: positive[{positive_cnt}], total[{len(self.x_list)}]'
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return str(self.x_list[index]), torch.tensor(self.y_list[index])
    
    def collate_fn(self, batch_data):
        xs, ys = zip(*batch_data)
        xs = self.tokenizer(xs, padding=True, truction=True, return_tensors='pt')
        ys = torch.stack(ys, dim=0)
        return xs, ys
        

if __name__ == '__main__':

    sample_train_data = preprocess_train_data(train_data_file_list[0])
    # sample_test_data = preprocess_test_data(test_data_file_list[0])

    # print(sample_train_data[:3])
    # print(sample_test_data[:3])
    # exit()
    
    # for line in sample_train_data[1:3]:
    #     print(line)

    sample_share_encoder = False
    sample_cls_target = 'hd+cv'
    sample_positive_ratio = 0.2
    sample_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./pretrained_model/')
    
    sample_train_data = Datasetv2(sample_train_data, sample_share_encoder, sample_cls_target, sample_tokenizer)
    print(sample_train_data.get_data_info('init'))
    sample_train_data.downsample(sample_positive_ratio)
    print(sample_train_data.get_data_info('down'))
    
    sample_train_data = DataLoader(sample_train_data, batch_size=5, collate_fn=sample_train_data.collate_fn)
    for sample_x, sample_y in sample_train_data:
        print(sample_x)
        print(sample_y)
        print()
        break
    # exit()
    
    # from model.baseline import BaselineModel
    # sample_model = BaselineModel(sample_config)
    # sample_criterion = nn.CrossEntropyLoss(reduction='sum')
    # for sample_x, sample_y in sample_train_data:
    #     sample_output = sample_model(sample_x)
    #     print(sample_output.shape)
    #     print(sample_y.shape)
    #     loss = sample_criterion(sample_output.view(-1, 2), sample_y.view(-1))
    #     print(loss)
    #     loss.backward()
    #     break
    
    # pass
    