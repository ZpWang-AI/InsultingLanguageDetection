import os
import logging
import warnings
import torch
import torch.nn as nn
import numpy as np
# import fitlog

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
from corpus import *
from model.baseline import BaselineModel


@clock
def eval_main(model, eval_dataloader, config:CustomConfig, logger:MyLogger):
    eval_fn_list = [
        binary_f1_score, 
        binary_precision,
        binary_recall,
        binary_accuracy,
    ]
    
    def eval_one_cls(pred_, gt_, cls_name):
        logger.info(f'- {cls_name}')
        
        eval_cls_res = []
        for eval_fn in eval_fn_list:
            cur_res = eval_fn(pred_, gt_)
            logger.info(f'{eval_fn.__name__[7:]:9s}: {cur_res:6.4f}')
            eval_cls_res.append(cur_res)
            
        confusion_matrix = binary_confusion_matrix(pred_, gt_)
        lines = [
            f'{cls_name:4s} pred_0 pred_1',
            f'gt_0 {confusion_matrix[0][0]:5d} {confusion_matrix[0][1]:5d}',
            f'gt_1 {confusion_matrix[1][0]:5d} {confusion_matrix[1][1]:5d}',
        ]
        logger.info(*lines, sep='\n')

        return eval_cls_res

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
    
    if config.share_encoder:
        eval_res = []
        for cls_p in range(3):
            eval_res.append(
                eval_one_cls(pred[:,cls_p], groundtruth[:,cls_p], cls_list[cls_p])
            )
    else:
        eval_res = [eval_one_cls(pred, groundtruth, config.cls_target)]

    return np.array(eval_res)
    

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
    
    warnings.filterwarnings('ignore')
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
    train_data = CustomDataset(train_data, config)
    dev_data = CustomDataset(dev_data, config)
    if config.downsample_data:
        train_data.downsample(config.positive_ratio)
        
    logger.info(train_data.get_data_info('train'))
    logger.info(dev_data.get_data_info('dev'))
    
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
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
            y = y.to(device)  # bsz, cls or bsz
            output = model(x)  # bsz, cls, 2 or bsz, 2
            loss = criterion(output.view(-1,2), y.view(-1))  
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
                    f'time[run:{epoch_running_time} / remain:{epoch_remain_time} / end:{epoch_end_time.strftime("%Y-%m-%d_%H:%M:%S")}]',
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
    def process_config_decorator(func):
        def new_func(*args, **kwargs):
            config = func(*args, **kwargs)
            
            if config.share_encoder:
                config.version += '-shareEncoder'
            else:
                config.version += '-'+config.cls_target
            return config
        return new_func
    
    def get_config_base_test():
        config = CustomConfig()
        config.model_name = 'bert-base-uncased'
        # config.model_name = 'roberta-base'
        config.device = 'cuda'
        config.cuda_id = '2'

        # config.just_test = True
        config.freeze_encoder = False
        config.downsample_data = True
        config.positive_ratio = 0.5
        config.share_encoder = False
        config.cls_target = 'hd+cv+vo'
        
        config.save_model_epoch = 1
        config.pb_frequency = 20

        config.epochs = 10
        config.batch_size = 8
        config.lr = 5e-5
        
        config.version = 'bertBase'
        config.train_data_file = train_data_file_list[0]
        config.dev_data_file = ''
        return config
    
    @process_config_decorator
    def config_mixed_three_cls():
        config = get_config_base_test()
        config.positive_ratio = 0.4
        config.share_encoder = False
        config.cls_target = 'hd+cv+vo'
        return config
    
    @process_config_decorator
    def config_share_encoder():
        config = get_config_base_test()
        config.positive_ratio = 2/3
        config.share_encoder = True
        return config
    
    @process_config_decorator
    def config_single_cls(cls):
        config = get_config_base_test()
        config.positive_ratio = 0.4
        config.share_encoder = False
        config.cls_target = cls
        return config
    
    train_main(config_mixed_three_cls())
    train_main(config_share_encoder())
    for cls_ in cls_list:
        train_main(config_single_cls(cls_))
    # train_main(get_config_base_test())

    
    pass

