import os
import numpy as np
import torch
import torch.nn as nn
import lightning
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
    
    model_name = 'bert-base-uncased'
    
    train_data_file = ''
    dev_data_file = ''
    test_data_file = ''
    pretrained_model_fold = './pretrained_model'
    log_fold = './logs'
    
    model_ckpt = ''
    
    just_test = False
    downsample_data = True
    positive_ratio = 0.4
    freeze_encoder = False
    share_encoder = False
    cls_target = 'hd'  # hd cv vo hd+vo hd+cv+vo..
    
    amp = True
    deepspeed = True
    
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


def main(config: Configv2):
    cuda_id = get_free_gpu()
    print(f'device: cuda {cuda_id}')
    
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
    
    log_path = path(config.log_fold)/path(get_cur_time().replace(':', '-'))
    
    callbacks = [ModelCheckpoint(
        dirpath=log_path/'checkpoint',
        filename='epoch{epoch}-f1score{val_macro_f1:.2f}',
        monitor='val_macro_f1',
        save_top_k=3,
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
        devices=[cuda_id],
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
    trainer.fit(
        model=model,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
    )
    trainer.test(
        model=model,
        dataloaders=test_data,
        ckpt_path='best'
    )


if __name__ == '__main__':
    config = Configv2()
    config.just_test = True
    config.share_encoder = True
    main(config)
    config.share_encoder = False
    main(config)
    exit()
    
    config = Configv2()
    config.share_encoder = True
    config.positive_ratio = 2/3
    main(config)
    
    for cls_target in ['hd', 'cv', 'vo', 'hd+vo', 'hd+cv+vo']:
        config = Configv2()
        config.share_encoder = False
        config.cls_target = cls_target
        config.positive_ratio = 0.4
        main(config)
    