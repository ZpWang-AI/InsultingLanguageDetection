import os
import numpy as np
import torch
import torch.nn as nn
import lightning
import gc
import warnings
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
    cls_target = 'hd'  # hd cv vo hd+vo hd+cv+vo..
    
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


cuda_id = get_free_gpu()
print(f'device: cuda {cuda_id}')
warnings.filterwarnings('ignore')


def main(config: Configv2):
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
    
    log_path = path(config.log_fold)/path(get_cur_time().replace(':', '-')+'_'+config.version)
    
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
    
    start_time = time.time()
    trainer.fit(
        model=model,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
    )
    running_time = time.time()-start_time
    logger.log_hyperparams({'running time': running_time})
    
    trainer.test(
        model=model,
        dataloaders=test_data,
        ckpt_path='best'
    )
    
    del trainer, model
    
    gc.collect()
    torch.cuda.empty_cache()
    

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

    def display():
        config = Configv2()
        config.version = 'display'
        config.share_encoder = False
        config.cls_target = 'hd+cv+vo'
        config.rdrop = True
        main(config)

    def best_model():
        config = Configv2()
        config.version = 'best'
        config.model_name = 'roberta-base'
        config.rdrop = 4
        config.early_dropout = 3
        config.amp = True
        config.deepspeed = True
        config.freeze_encoder = False
        main(config)
    
    def model_encoder_cmp():
        model_name_lst = ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'xlm-roberta-base']
        config = Configv2()
        config.version = 'encoder of model cmp'
        for model_name in model_name_lst:
            config.model_name = model_name
            main(config)
    
    def structure_cmp():
        config = Configv2()
        config.version = 'structure cmp'
        config.share_encoder = True
        config.positive_ratio = 2/3
        main(config)
        config.share_encoder = False
        config.positive_ratio = 0.4
        for cls_tar in ['hd', 'cv', 'vo', 'hd+cv+vo']:
            config.cls_target = cls_tar
            main(config)
    
    def downsample_cmp():
        config = Configv2()
        config.version = 'downsample cmp'
        config.downsample_data = False
        main(config)
        config.downsample_data = True
        for rate in range(1, 11):
            config.positive_ratio = rate/10
            main(config)
    
    def rdrop_cmp():
        config = Configv2()
        config.version = 'rdrop cmp'
        config.rdrop = None
        main(config)
        for p_rdrop in range(1, 6):
            config.rdrop = p_rdrop
            main(config)
    
    def early_dropout_cmp():
        config = Configv2()
        config.version = 'early dropout cmp'
        config.early_dropout = None
        main(config)
        for start_epoch in range(1, 10):
            config.early_dropout = start_epoch
            main(config)
    
    def running_time_ablation():
        config = Configv2()
        config.version = 'running time ablation'
        for p in range(4):
            config.amp = p % 2
            config.deepspeed = p // 2 % 2
            main(config)
    
    def freeze_encoder_ablation():
        config = Configv2()
        config.version = 'freeze encoder ablation'
        config.freeze_encoder = True
        main(config)
    
    # just_test_main()
    # baseline()
    # display()
    # best_model()
    # model_encoder_cmp()
    # structure_cmp()
    # downsample_cmp()
    # rdrop_cmp()
    # early_dropout_cmp()
    # running_time_ablation()
    # freeze_encoder_ablation()
    
    pass