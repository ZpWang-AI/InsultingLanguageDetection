import os
import logging
import torch
import torch.nn as nn

from pathlib import Path as path
from torch.nn import functional as F
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    XLMRobertaForSequenceClassification
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


class BaselineModel(nn.Module):
    def __init__(self, 
                 config,
                #  config: CustomConfig,
                 ):
        super().__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config.model_name, 
                                                       num_labels=2, 
                                                       cache_dir=config.pretrained_model_fold)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name,
                                                       cache_dir=config.pretrained_model_fold)
        self.encoder = AutoModel.from_config(self.model_config)
        if config.share_encoder:
            self.decoder_list = nn.ModuleList(ClassificationHead(self.model_config)for _ in range(3))
        else:
            self.decoder = ClassificationHead(self.model_config)
        # self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    
    def get_pretrained_encoder(self):
        # logging.getLogger("transformers").setLevel(logging.ERROR)
        cache_dir = self.config.pretrained_model_fold
        path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.encoder = self.encoder.from_pretrained(self.config.model_name, cache_dir=cache_dir)
        self.to(self.config.device)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, input):
        input = self.tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors='pt')
        input.to(self.config.device)
        feature = self.encoder(**input)
        feature = feature[0]
        # print(input)
        # print(feature[0])
        # print()
        # print(feature[0].shape)
        if self.config.share_encoder:
            logits_list = [decoder(feature)for decoder in self.decoder_list]  # cls(3), bsz, 2
            prob_list = [F.softmax(logits, dim=-1)for logits in logits_list]  # cls, bsz, 2
            output = torch.stack(prob_list, dim=1)  # bsz, cls, 2
            return output
        else:
            logits = self.decoder(feature)  # bsz, 2
            prob = F.softmax(logits, dim=-1)  # bsz, 2
            return prob
    
    def predict(self, input):
        output = self(input)  # bsz, cls, 2 or bsz, 2
        preds = torch.argmax(output, dim=-1)  # bsz, cls(3) or bsz
        return preds


if __name__ == '__main__':
    class SampleConfig:
        model_name = 'xlm-roberta-base'
        device = 'cuda'
        
        pretrained_model_fold = './saved_model'
        share_encoder = True
        # share_encoder = False
        
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    
    sample_sentences = [
        'a sample sentence', 
        'two sample sentences',
        'three sample sentences',
        'four sample sentences '*1000,
        # '谢谢关注',
    ]
    sample_model = BaselineModel(SampleConfig())
    sample_model.get_pretrained_encoder()
    # sample_model.freeze_encoder()
    sample_model.to(SampleConfig.device)
    sample_model.eval()
    sample_output = sample_model(sample_sentences)
    print(sample_output)
    print(sample_output.shape)
    sample_preds = sample_model.predict(sample_sentences)
    print(sample_preds)
    print(sample_preds.shape)
    pass
    
