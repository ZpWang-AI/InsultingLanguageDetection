import os
import logging
import torch
import torch.nn as nn
import lightning
import torchmetrics
import time

from pathlib import Path as path
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from lightning import Fabric
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
# from config import CustomConfig


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomModel(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained_model_fold='./pretrained_model',
                 share_encoder=False,
                 ):
        super().__init__()
        self.model_name = model_name
        self.pretrained_model_fold = pretrained_model_fold
        self.share_encoder = share_encoder
        
        self.model_config = AutoConfig.from_pretrained(model_name, 
                                                       num_labels=2, 
                                                       cache_dir=pretrained_model_fold)
        self.encoder = AutoModel.from_config(self.model_config)
        if share_encoder:
            self.decoder_list = nn.ModuleList([ClassificationHead(self.model_config)for _ in range(3)])
        else:
            self.decoder = ClassificationHead(self.model_config)
        # self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    
    def get_pretrained_encoder(self):
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)        
        # logging.getLogger("transformers").setLevel(logging.ERROR)
        cache_dir = self.pretrained_model_fold
        path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.encoder = AutoModel.from_pretrained(self.model_name, cache_dir=cache_dir)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, batch_x):
        feature = self.encoder(**batch_x)
        feature = feature['last_hidden_state']
        # feature = feature[0]
        
        if self.share_encoder:
            logits_list = [decoder(feature)for decoder in self.decoder_list]  # cls(3), bsz, 2
            prob_list = [F.softmax(logits, dim=-1)for logits in logits_list]  # cls, bsz, 2
            return torch.stack(prob_list, dim=0)  # cls, bsz, 2
        else:
            logits = self.decoder(feature)  # bsz, 2
            prob = F.softmax(logits, dim=-1)  # bsz, 2
            return prob
    
    def predict(self, batch_x):
        output = self(batch_x)  # cls, bsz, 2 or bsz, 2
        preds = torch.argmax(output, dim=-1)  # cls, bsz or bsz
        return preds


class Modelv2(lightning.LightningModule):
    def __init__(self, 
                 model_name='bert-base-uncased',
                 pretrained_model_fold='./pretrained_model',
                 share_encoder=False,
                 rdrop=True,
                 early_dropout=True,
                 optimizer=torch.optim.AdamW, 
                 lr=5e-5,
                 criterion=nn.CrossEntropyLoss(),
                 ):
        super().__init__()
        
        self.model_name = model_name
        self.pretrained_model_fold = pretrained_model_fold
        self.share_encoder = share_encoder
        self.rdrop = rdrop
        self.early_dropout = early_dropout
        
        self.model_config = AutoConfig.from_pretrained(model_name, 
                                                       num_labels=2, 
                                                       cache_dir=pretrained_model_fold)
        self.encoder = AutoModel.from_config(self.model_config)
        if share_encoder:
            self.decoder_list = nn.ModuleList([ClassificationHead(self.model_config)for _ in range(3)])
        else:
            self.decoder = ClassificationHead(self.model_config)
        
        self.optimizer = optimizer
        self.lr = lr
        self.criterion = criterion
        
        self.metric_name_list = ['accuracy', 'precision', 'recall', 'f1']
        if self.share_encoder:
            self.train_metric_list = [
                [
                    torchmetrics.Accuracy('binary'),
                    torchmetrics.Precision('binary'),
                    torchmetrics.Recall('binary'),
                    torchmetrics.F1Score('binary')
                ]
                for _ in range(3)
            ]
            self.val_metric_list = [
                [
                    torchmetrics.Accuracy('binary'),
                    torchmetrics.Precision('binary'),
                    torchmetrics.Recall('binary'),
                    torchmetrics.F1Score('binary')
                ]
                for _ in range(3)
            ]
            self.test_metric_list = [
                [
                    torchmetrics.Accuracy('binary'),
                    torchmetrics.Precision('binary'),
                    torchmetrics.Recall('binary'),
                    torchmetrics.F1Score('binary')
                ]
                for _ in range(3)
            ]
        else:
            self.train_metric_list = [
                torchmetrics.Accuracy('binary'),
                torchmetrics.Precision('binary'),
                torchmetrics.Recall('binary'),
                torchmetrics.F1Score('binary')
            ]
            self.val_metric_list = [
                torchmetrics.Accuracy('binary'),
                torchmetrics.Precision('binary'),
                torchmetrics.Recall('binary'),
                torchmetrics.F1Score('binary')
            ]
            self.test_metric_list = [
                torchmetrics.Accuracy('binary'),
                torchmetrics.Precision('binary'),
                torchmetrics.Recall('binary'),
                torchmetrics.F1Score('binary')
            ]
            
        def recurse_moduleList(lst): 
            lst = [recurse_moduleList(p) if type(p) == list else p for p in lst]
            return nn.ModuleList(lst)
        
        self.train_metric_list = recurse_moduleList(self.train_metric_list)
        self.val_metric_list = recurse_moduleList(self.val_metric_list)
        self.test_metric_list = recurse_moduleList(self.test_metric_list)
    
    def get_pretrained_encoder(self):
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)        
        # logging.getLogger("transformers").setLevel(logging.ERROR)
        cache_dir = self.pretrained_model_fold
        path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.encoder = AutoModel.from_pretrained(self.model_name, cache_dir=cache_dir)
        
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, batch_x):
        feature = self.encoder(**batch_x)
        feature = feature['last_hidden_state']
        # feature = feature[0]
        
        if self.share_encoder:
            logits_list = [decoder(feature)for decoder in self.decoder_list]  # cls(3), bsz, 2
            prob_list = [F.softmax(logits, dim=-1)for logits in logits_list]  # cls, bsz, 2
            return torch.stack(prob_list, dim=0)  # cls, bsz, 2
        else:
            logits = self.decoder(feature)  # bsz, 2
            prob = F.softmax(logits, dim=-1)  # bsz, 2
            return prob
    
    def predict(self, batch_x):
        output = self(batch_x)  # cls, bsz, 2 or bsz, 2
        preds = torch.argmax(output, dim=-1)  # cls, bsz or bsz
        return preds
    
    def one_step(self, batch, stage):
        xs, ys = batch
        output = self(xs)
        
        loss = self.criterion(output.view(-1,2), ys.view(-1))
        self.log(f'{stage}_loss', loss)
        
        with torch.no_grad():
            preds = torch.argmax(output, -1)
            metric_list = getattr(self, f'{stage}_metric_list')
            if self.share_encoder:
                macro_f1 = 0
                for p in range(3):
                    for metric_name, metric in zip(self.metric_name_list, metric_list[p]):
                        metric(preds[p], ys[p])
                        self.log(f'{stage}_{metric_name}_{p}', metric, on_epoch=True, on_step=False)
                    macro_f1 += metric_list[p][-1].compute()
                macro_f1 /= 3
                self.log(f'{stage}_macro_f1', macro_f1, on_epoch=True, on_step=False)
            else:
                for metric_name, metric in zip(self.metric_name_list, metric_list):
                    metric(preds, ys)
                    self.log(f'{stage}_{metric_name}', metric, on_epoch=True, on_step=False)
                self.log(f'{stage}_macro_f1', metric_list[-1], on_epoch=True, on_step=False)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.one_step(batch, 'train')
            
    def validation_step(self, batch, batch_idx):
        self.one_step(batch, 'val')
        
    def test_step(self, batch, batch_idx):
        self.one_step(batch, 'test')
    
    def configure_optimizers(self):
        params_list = [{'params':self.encoder.parameters()}]
        if self.share_encoder:
            params_list.append({'params': self.decoder_list.parameters()})
        else:
            params_list.append({'params': self.decoder.parameters()})
        return self.optimizer(params_list, self.lr)
    

if __name__ == '__main__':
    class SampleDataset(Dataset):
        def __init__(self, model_name, pretrained_model_fold='./pretrained_model', share_encoder=False) -> None:
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=pretrained_model_fold)
            self.data = [
                'a sample sentence', 
                'two sample sentences',
                'three sample sentences',
                'four sample sentences '*3,
                # '谢谢关注',
            ]*10
            self.share_encoder = share_encoder
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            if self.share_encoder:
                return self.data[index], (index%2,)*3
            else:
                return self.data[index], index%2    
        def collate_fn(self, batch_data):
            xs, ys = zip(*batch_data)
            xs = self.tokenizer(xs, padding=True, truncation=True, return_tensors='pt')
            ys = torch.tensor(ys)
            if self.share_encoder:
                ys = ys.reshape((3,-1))
            return xs, ys
    
    sample_model_name = 'bert-base-uncased'
    # sample_model_name = 'distilBert-base'
    sample_pretrained_model_fold = './pretrained_model'
    sample_share_encoder = True
    devices = [4]
    
    def sample_model_forward():
        print('--- start testing')
        
        start_time = time.time()
        cur_time = time.time()
        sample_data = SampleDataset(sample_model_name, sample_pretrained_model_fold, sample_share_encoder)
        sample_data = DataLoader(sample_data, batch_size=5, collate_fn=sample_data.collate_fn)
        print(f'prepare data cost {time.time()-cur_time:.2f}s')
        
        cur_time = time.time()
        sample_model = Modelv2(
            sample_model_name, 
            sample_pretrained_model_fold, 
            share_encoder=sample_share_encoder,
        )
        print(f'prepare model cost {time.time()-cur_time:.2f}s')
        
        cur_time = time.time()
        sample_model.get_pretrained_encoder()
        # sample_model.freeze_encoder()
        print(f'load model cost {time.time()-cur_time:.2f}s')
        
        cur_time = time.time()
        fab = Fabric(accelerator='cuda',devices=devices,precision='16-mixed')
        fab.launch()
        sample_model_fab = fab.setup_module(sample_model)
        sample_data_fab = fab.setup_dataloaders(sample_data)
        fab.barrier()
        print(f'prepare fabric cost {time.time()-cur_time:.2f}s')
        
        cur_time = time.time()
        for sample_x, sample_y in sample_data_fab:
            print('x')
            # print(sample_x)
            print(sample_x['input_ids'].shape)
            print('y')
            # print(sample_y)
            print(sample_y.shape)
            sample_output = sample_model_fab(sample_x)
            print('output')
            # print(sample_output)
            print(sample_output.shape)
            break
        print(f'deal one item cost {time.time()-cur_time:.2f}s')
        print(f'total cost {time.time()-start_time:.2f}s')
    
    def sample_train_test():
        sample_data = SampleDataset(sample_model_name, sample_pretrained_model_fold, sample_share_encoder)
        sample_data = DataLoader(sample_data, batch_size=5, collate_fn=sample_data.collate_fn)
        sample_model = Modelv2(
            sample_model_name, 
            sample_pretrained_model_fold, 
            share_encoder=sample_share_encoder,
        )
        sample_model.get_pretrained_encoder()
        
        sample_callbacks = [ModelCheckpoint(
            dirpath='logs/sample_ckpt/',
            filename='{epoch}-{val_macro_f1:.2f}',
            monitor='val_macro_f1',
            save_top_k=3,
            mode='max',
        )]
        sample_logger = CSVLogger(save_dir='logs', name='sample-log', version=10)
        sample_logger.log_hyperparams({'lr': 10, 'version': 'sample'})
        
        sample_trainer = lightning.Trainer(
            max_epochs=5,
            callbacks=sample_callbacks,
            accelerator='gpu',
            devices=devices,
            logger=sample_logger,
            log_every_n_steps=10,
            # deterministic=True,
            precision='16-mixed',
            # strategy='deepspeed_stage_2'
        )
        
        sample_trainer.fit(
            model=sample_model,
            train_dataloaders=sample_data,
            val_dataloaders=sample_data,
            
        )
        # sample_ckpt = torch.load('./logs/sample_ckpt/epoch=0-step=8-v1.ckpt')
        # sample_lightning_model.load_state_dict(sample_ckpt['state_dict'])
        sample_trainer.test(
            model=sample_model,
            dataloaders=sample_data,
            ckpt_path='best'
        )
    
    def sample_load_ckpt():
        sample_data = SampleDataset(sample_model_name, sample_pretrained_model_fold, sample_share_encoder)
        sample_data = DataLoader(sample_data, batch_size=5, collate_fn=sample_data.collate_fn)
        sample_model = Modelv2(
            sample_model_name, 
            sample_pretrained_model_fold, 
            share_encoder=sample_share_encoder
        )
        
        sample_ckpt_file = './logs/sample_ckpt/epoch=0-val_macro_f1=1.00.ckpt'
        # sample_model: lightning.LightningModule
        sample_model.load_from_checkpoint(sample_ckpt_file)
        
        fab = Fabric(accelerator='cuda',devices=devices,precision='16')
        fab.launch()
        sample_model_fab = fab.setup_module(sample_model)
        sample_data_fab = fab.setup_dataloaders(sample_data)
        fab.barrier()
        
        for sample_x, sample_y in sample_data_fab:
            print('x')
            # print(sample_x)
            print(sample_x['input_ids'].shape)
            print('y')
            # print(sample_y)
            print(sample_y.shape)
            sample_output = sample_model_fab(sample_x)
            print('output')
            # print(sample_output)
            print(sample_output.shape)
            break
        
    sample_model_forward()
    print('-'*20)
    sample_train_test()
    print('-'*20)
    sample_load_ckpt()
    print('-'*20)
    
        
    pass