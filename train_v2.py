import os
import numpy as np
import torch
import torch.nn as nn
import lightning
import gc
import warnings
import yaml
import shutil
import traceback
# import fitlog

from tqdm import tqdm
from pathlib import Path as path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from utils import *
from corpus_v2 import *
from model.model_v2 import Modelv2


class Configv2:
    version = 'bertBase'
    
    just_test = False
    
    # cmp
    model_name = 'bert-base-uncased'

    share_encoder = False
    cls_target = 'hd+cv+vo'  # hd cv vo hd+vo hd+cv+vo..
    
    downsample_data = True
    positive_ratio = 0.4
    
    rdrop = None
    
    early_dropout = None
    
    # ablation
    amp = True
    deepspeed = True
    
    freeze_encoder = False
    
    # simple settings
    train_data_file = ''
    dev_data_file = ''
    test_data_file = ''
    pretrained_model_fold = './pretrained_model'
    log_fold = './logs'
    
    epochs = 10
    batch_size = 16
    log_step = 10
    
    train_ratio = 0.8
    lr = 5e-5
    
    def as_list(self):
        return [[attr, getattr(self, attr)] for attr in dir(self)
                if not callable(getattr(self, attr)) and not attr.startswith("__")]
    
    def as_dict(self):
        return dict(self.as_list())


devices = []
completed_mark_file = 'yes.txt'
error_mark_file = 'error.txt'


def clear_error_log(log_root='./logs/'):
    log_root = path(log_root)
    for project_fold in os.listdir(log_root):
        project_fold = log_root/project_fold
        for log_fold in os.listdir(project_fold):
            log_fold = project_fold/log_fold
            log_fold: path
            if log_fold.is_dir() and error_mark_file in os.listdir(log_fold):
                shutil.rmtree(log_fold)


def main_decorator(main_func):
    def new_main_func(config):
        try:
            if not devices:
                cuda_id = ManageGPUs.get_free_gpu(target_mem_mb=9000)
                print(f'device: cuda {cuda_id}')
                warnings.filterwarnings('ignore')
                devices.append(cuda_id)
                
            log_path = path(config.log_fold)
            log_path /= path(config.version.replace(' ', '_'))
            log_path.mkdir(parents=True, exist_ok=True)
            
            config_dic = config.as_dict()
            for son_dir in os.listdir(log_path):
                son_dir = log_path/path(son_dir)
                son_dir_hparams = son_dir/'hparams.yaml'
                if son_dir_hparams.exists():
                    son_config_dic = load_config_from_yaml(son_dir_hparams)
                    if all(config_dic[k] == son_config_dic[k] for k in config_dic) and completed_mark_file in os.listdir(son_dir):
                        return 
            log_path /= path(get_cur_time().replace(':', '-'))
            log_path.mkdir(parents=True, exist_ok=True)

            gc.collect()
            torch.cuda.empty_cache()
            main_func(config, log_path)
            gc.collect()
            torch.cuda.empty_cache()

            with open(log_path/path(completed_mark_file), 'w')as f:
                f.write('')
                
        except Exception as cur_error:
            with open(log_path/path(error_mark_file), 'w')as f:
                f.write(str(cur_error))
                f.write('\n'+'-'*20+'\n')
                f.write(traceback.format_exc())
    
    return new_main_func


@main_decorator
def main(config: Configv2, log_path):

    train_data_init = preprocess_train_data(train_data_file_list[0])
    test_data_init = preprocess_test_data(test_data_file_list[0])
    train_data_init, val_data_init = train_test_split(train_data_init, train_size=config.train_ratio, shuffle=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.pretrained_model_fold)
    
    def get_dataloader(data_init, phase):
        dataset_ = Datasetv2(
            data=data_init,
            share_encoder=config.share_encoder,
            cls_target=config.cls_target,
            tokenizer=tokenizer,
        )
        if phase == 'train' and config.downsample_data:
            dataset_.downsample(config.positive_ratio)
        dataloader_ = DataLoader(
            dataset_, 
            batch_size=config.batch_size,
            shuffle=(phase=='train'),
            collate_fn=dataset_.collate_fn,
            drop_last=False,
        )
        return dataloader_
        
    train_data = get_dataloader(train_data_init, 'train')
    val_data = get_dataloader(val_data_init, 'val')
    test_data = get_dataloader(test_data_init, 'test')
    
    model = Modelv2(
        model_name=config.model_name,
        pretrained_model_fold=config.pretrained_model_fold,
        share_encoder=config.share_encoder,
        optimizer=torch.optim.AdamW,
        lr=config.lr,
        criterion=nn.CrossEntropyLoss(),
    )
    model.get_pretrained_encoder()
    if config.freeze_encoder:
        model.freeze_encoder()
    
    callbacks = [ModelCheckpoint(
        # dirpath=log_path/'checkpoint',
        dirpath=log_path,
        filename='epoch{epoch}-f1score{val_macro_f1:.2f}',
        monitor='val_macro_f1',
        save_top_k=1,
        mode='max',
        auto_insert_metric_name=False,
        save_weights_only=True,
    )]
    logger = CSVLogger(save_dir=log_path, name='', version='')
    logger.log_hyperparams(config.as_dict())
    
    limit = 10 if config.just_test else None
    trainer = lightning.Trainer(
        accelerator='gpu',
        strategy=('deepspeed_stage_2' if config.deepspeed else 'auto'),
        devices=devices,
        precision=('16-mixed' if config.amp else '32-true'),
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.epochs,
        log_every_n_steps=config.log_step,
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
        # default_root_dir=log_path,
    )
    
    start_time = time.time()
    trainer.fit(
        model=model,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
    )
    running_time = time.time()-start_time
    running_time = datetime.timedelta(seconds=int(running_time))
    logger.log_hyperparams({'running time': str(running_time)})
    
    trainer.test(
        model=model,
        dataloaders=test_data,
        ckpt_path='best'
    )
    
    del trainer, model, callbacks, logger
    

if __name__ == '__main__':
    def just_test_main():
        config = Configv2() 
        config.version = 'just test'
        config.epochs = 5
        config.batch_size = 8
        config.rdrop = 2
        config.early_dropout = 2
        config.just_test = True
        config.share_encoder = True
        main(config)
        config.share_encoder = False
        main(config)
        exit()

    def baseline():
        config = Configv2()
        config.version = 'baseline'
        main(config)    
    
    pass