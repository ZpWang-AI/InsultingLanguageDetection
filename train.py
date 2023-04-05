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
    get_data_info,
    downsample_data,
    CustomDataset,
)
from model.baseline import BaselineModel


cls_list = ['hd', 'cv', 'vo']

def get_gt_by_config(gt, config:CustomConfig):
    # gt: bsz, cls(3)
    # config.share_encoder: True/False
    # config.cls_target_list: hd, cv, vo, hd+vo, hd+cv+vo
    # return: bsz, cls or bsz
    if config.share_encoder:
        return gt
    else:
        res = torch.zeros(gt.shape[0], dtype=gt.dtype)
        for cls_target in config.cls_target.split('+'):
            res |= gt[:,cls_list.index(cls_target)]
        return res
    

@clock
def eval_main(model, eval_dataloader, config:CustomConfig, logger:MyLogger):
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
    groundtruth = get_gt_by_config(groundtruth, config)
    # print(pred, groundtruth)
    
    eval_fn_list = [
        binary_f1_score, 
        binary_precision,
        binary_recall,
        binary_accuracy,
    ]
    
    if config.share_encoder:
        cls_cnt = 3
        eval_res = [
            [eval_fn(pred[:, p], groundtruth[:, p]) for eval_fn in eval_fn_list]
            for p in range(cls_cnt) 
        ]
        eval_res = np.array(eval_res)
        
        def show_res_multi():
            lines = [
                ' '.join(['  ']+[eval_fn.__name__[7:]for eval_fn in eval_fn_list]),
            ]
            for p, cls_name in enumerate(cls_list):
                cur_line = [cls_name]
                cur_line.extend(
                    list(map(lambda x:f'{x:6.4f}' , eval_res[p]))
                )
                lines.append(' '.join(cur_line))
            logger.info(*lines, sep='\n')

            for p, cls_name in enumerate(cls_list):
                confusion_matrix = binary_confusion_matrix(pred[:, p], groundtruth[:, p])
                lines = [
                    f'{cls_name:4s} pred_0 pred_1',
                    f'gt_0 {confusion_matrix[0][0]:5d} {confusion_matrix[0][1]:5d}',
                    f'gt_1 {confusion_matrix[1][0]:5d} {confusion_matrix[1][1]:5d}',
                ]
                logger.info(*lines, sep='\n')
        
        show_res_multi()
    else:
        cls_cnt = 1

        # TODO
    return eval_res
    

@clock(sym='-----')
def train_main(config: CustomConfig):
    def check_file_name(fname):
        fname = fname.replace(':', '_')
        fname = fname.replace(' ', '_')
        return fname
    
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
        just_print=False,
        log_with_time=False,
    )
    
    logger.info(*config.as_list(), sep='\n')
    logger.info(get_cur_time())
    
    train_data = preprocess_train_data(config.train_data_file)
    if not config.dev_data_file:
        train_data, dev_data = train_test_split(train_data, train_size=0.8, shuffle=True)
    else:
        dev_data = preprocess_train_data(config.dev_data_file)
    if config.downsample_data:
        train_data = downsample_data(train_data, config.downsample_ratio)
        
    logger.info(get_data_info(train_data, 'train'))
    logger.info(get_data_info(dev_data, 'dev'))
    
    train_data = CustomDataset(train_data, config)
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    dev_data = CustomDataset(dev_data, config)
    dev_data = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False)
    
    model = BaselineModel(config)
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
        epoch_start_time = time.time()
        for p, (x, y) in enumerate(train_data):
            y = y.to(device)  # bsz, cls
            output = model(x)  # bsz, cls, 2 or bsz, 2
            if config.share_encoder:
                output = output.view(-1,2)  # bsz, cls, 2 -> bsz*cls, 2
                y = y.view(-1)  # bsz, cls -> bsz*cls
            else:
                output  # bsz, 2
                y = get_gt_by_config(y, config)  # bsz, cls -> bsz
            loss = criterion(output, y)  # (output)bsz*cls,2 (y)bsz*cls or (output)bsz,2 (y)bsz
            loss.backward()
            tot_loss += loss
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            if config.just_test:
                break
            p += 1
            if p % config.pb_frequency == 0 or p == len(train_data):
                epoch_running_time = time.time()-epoch_start_time
                epoch_remain_time = epoch_running_time/p*(len(train_data)-p)
                epoch_running_time = datetime.timedelta(seconds=int(epoch_running_time))
                epoch_remain_time = datetime.timedelta(seconds=int(epoch_remain_time))
                epoch_end_time = datetime.datetime.now(datetime.timezone(offset=datetime.timedelta(hours=8))) + epoch_remain_time
                logger.info(
                    f'batch[{p}/{len(train_data)}]',
                    f'time[run:{epoch_running_time}/remain:{epoch_remain_time}/end:{epoch_end_time.strftime("%Y-%m-%d_%H:%M:%S")}]',
                    f'loss: {tot_loss.average:.6f}',
                    sep=', '
                )

        logger.info('\n> evaluate train <')
        eval_main(model, train_data, config, logger)
        logger.info('\n> evaluate dev <')
        eval_res = eval_main(model, dev_data, config, logger)
        average_f1 = np.average(eval_res[:, 0])
        
        logger.info(
            f'epoch{epoch} ends, '
            f'average of f1: {average_f1:.4f}\n'
        )
        if config.just_test:
            break   
        
        if epoch % config.save_model_epoch == 0:
            saved_model_file = (
                f'{start_time}_'
                f'epoch{epoch}_'
                f'{int(average_f1*1000)}'
                '.pth'
            )
            torch.save(
                model.state_dict(),
                saved_model_fold / saved_model_file
            )
            
    logger.info('=== finish training ===')
    logger.info(get_cur_time())
    logger.close()
    del logger


if __name__ == '__main__':
    def get_config_base_test():
        config = CustomConfig()
        config.model_name = 'bert-base-uncased'
        config.device = 'cuda'
        config.cuda_id = '3'

        # config.just_test = True
        config.freeze_encoder = False
        config.downsample_data = True
        config.downsample_ratio = 0.1
        config.share_encoder = True
        config.cls_target = 'hd'
        
        config.save_model_epoch = 1
        config.pb_frequency = 20

        config.epochs = 10
        config.batch_size = 8
        config.lr = 5e-5
        
        config.version = 'baseline'
        config.train_data_file = train_data_file_list[0]
        config.dev_data_file = ''
        return config
    
    train_main(get_config_base_test())
    
    pass

