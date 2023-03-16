# import os
import torch
import torch.nn as nn

from torch.nn import functional as F
from transformers import (XLMRobertaModel, 
                          XLMRobertaTokenizer, 
                          XLMRobertaForSequenceClassification, 
                          XLMRobertaConfig,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          AutoTokenizer
                          )


class BertModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        
    def forward(self, x):
        x = self.tokenizer(x, padding=True, truncation=True, max_length=128, return_tensors='pt')
        x.to(self.config.device)
        output = self.model(**x)
        return output


class LossModel(nn.Module):
    def __init__(self, model) -> None:
        self.model = model


if __name__ == '__main__':
    pass
    
