import os
import logging
import warnings
import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path as path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torcheval.metrics.functional import (
    multiclass_f1_score, 
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
    multiclass_confusion_matrix,
    binary_f1_score,
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_confusion_matrix
)

from utils import get_cur_time
from config import *
from corpus import deal_train_data, deal_test_data, CustomDataset
from model.baseline import BertModel


def eval_main(model, eval_dataloader, config):
    model.eval()
    pred, groundtruth = [], []
    with torch.no_grad():
        for x, y in eval_dataloader:
            output = model.predict(x)
            pred.append(output)
            groundtruth.append(y)
            if config.just_test:
                break
    pred = torch.cat(pred, dim=0).cpu()
    groundtruth = torch.cat(groundtruth, dim=0).cpu()
    # print(pred, groundtruth)
    n_sample, cls_cnt = groundtruth.shape
    # print(n_sample, cls_cnt)
    
    eval_fn_list = [binary_f1_score, binary_accuracy, binary_precision, binary_recall]
    eval_res = [
        [eval_fn(pred[:, p], groundtruth[:, p]) for p in range(cls_cnt)] for eval_fn in eval_fn_list
    ]
    
    def show_res():
        lines = [[' '*9, 'hd'.rjust(6), 'cv'.rjust(6), 'vo'.rjust(6)]]+[
            [eval_fn.__name__[7:].ljust(9), 
             *map(lambda x: f'{x*100:6.2f}', eval_res[p])]
                for p, eval_fn in enumerate(eval_fn_list)
        ]
        for line in lines:
            print(' '.join(line))
    
    show_res()
    return eval_res
    

def train_main():
    config = get_default_config()
    config.device = 'cuda'
    config.cuda_id = '5'
    config.just_test = False
    if not path(config.saved_model_fold).exists():
        os.mkdir(config.saved_model_fold)
    
    device = config.device
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    
    train_data = deal_train_data()
    train_data, dev_data = train_test_split(train_data, train_size=0.8, shuffle=True)
    train_data = CustomDataset(train_data, config)
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    dev_data = CustomDataset(dev_data, config)
    dev_data = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False)
    
    model = BertModel(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    
    for epoch in range(1, config.epochs+1):
        model.train()
        tot_loss = 0
        for x, y in tqdm(train_data, desc=f'epoch{epoch}'):
            y = y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, 2), y.view(-1))
            tot_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if config.just_test:
                break
        print(f'epoch{epoch} loss: {tot_loss/len(train_data)}')
        eval_res = eval_main(model, dev_data, config)
        if config.just_test:
            break   
        
        torch.save(
            model.parameters(),
            f'{config.saved_model_fold}/{get_cur_time()}_{config.version}_{config.model_name}_epoch{epoch}.pth'
        )


if __name__ == '__main__':
    train_main()
