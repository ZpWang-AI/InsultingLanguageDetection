import os
import logging
import warnings
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from pathlib import Path as path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torcheval.metrics.functional import (
    binary_f1_score,
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_confusion_matrix
)

from utils import *
from config import CustomConfig
from corpus import (
    train_data_file_list,
    test_data_file_list,
    preprocess_train_data,
    preprocess_test_data,
    CustomDataset,
)
from model.baseline import BertModel


@clock
def eval_main(model, eval_dataloader, config, logger):
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
    
    eval_fn_list = [
        binary_f1_score, 
        binary_accuracy,
        binary_precision,
        binary_recall
    ]
    eval_res = [
        [eval_fn(pred[:, p], groundtruth[:, p]) for eval_fn in eval_fn_list]
        for p in range(cls_cnt) 
    ]
    
    def show_res():
        lines = [
            ' '.join(['  ']+[eval_fn.__name__[7:]for eval_fn in eval_fn_list]),
        ]
        for p, cls_name in enumerate(['hd', 'cv', 'vo']):
            cur_line = [cls_name]
            cur_line.extend(
                list(map(lambda x:f'{x:6.2f}' , eval_res[p]))
            )
            lines.append(' '.join(cur_line))
        logger.info(*lines, sep='\n')
    
    show_res()
    return eval_res
    

@clock(sym='-----')
def train_main(config: CustomConfig):
    def check_file_name(fname):
        fname = fname.replace(':', '_')
        fname = fname.replace(' ', '_')
        return fname
    
    config.device = 'cuda'
    config.cuda_id = '5'
    config.just_test = False
    if not path(config.saved_model_fold).exists():
        os.mkdir(config.saved_model_fold)
    
    device = config.device
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    
    start_time = check_file_name(get_cur_time())
    config.version = check_file_name(config.version)
    saved_res_fold = path(config.save_res_fold) / path(f'{start_time}_{config.version}')
    saved_model_fold = saved_res_fold / path('saved_model')
    saved_model_fold.mkdir(parents=True, exist_ok=True)
    
    logger = MyLogger(
        fold=saved_res_fold, 
        file=f'{start_time}_{config.version}.out',
        just_print=config.just_test,
        log_with_time=False,
    )
    
    logger.info(*config.as_list())
    logger.info(get_cur_time())
    
    train_data = preprocess_train_data(config.train_data_file)
    if not config.dev_data_file:
        train_data, dev_data = train_test_split(train_data, train_size=0.8, shuffle=True)
    else:
        dev_data = preprocess_train_data(config.dev_data_file)
    train_data = CustomDataset(train_data, config)
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    dev_data = CustomDataset(dev_data, config)
    dev_data = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False)
    
    model = BertModel(config)
    model.get_pretrained_encoder()
    if config.freeze_encoder:
        model.freeze_encoder()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # scheduler = StepLR(optimizer, step_size=10)
    
    logger.info('=== start training ===')
    for epoch in range(1, config.epochs+1):
        model.train()
        tot_loss = AverageMeter()
        for x, y in train_data:
            y = y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, 2), y.view(-1))
            loss.backward()
            tot_loss += loss
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
    custom_config = CustomConfig()
    
    train_main()
