import os
import torch
import torch.nn as nn

from config import *
from corpus import deal_train_data, deal_test_data, CustomDataset
from model.xlm_roberta import BertModel

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torcheval.metrics.functional import (
    multiclass_f1_score, 
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
    multiclass_confusion_matrix,
)
from torcheval.metrics.functional import multiclass_f1_score


def eval_main(model, eval_dataloader):
    model.eval()
    pred, groundtruth = [], []
    with torch.no_grad():
        for x, y in eval_dataloader:
            output = model(x)
            pred.append(output)
            groundtruth.append(y)
    pred = torch.cat(pred)
    groundtruth = torch.cat(groundtruth)
    eval_res = {
        'macro_f1': multiclass_f1_score(pred, groundtruth, num_classes=2, average='macro'),
        'micro_f1': multiclass_f1_score(pred, groundtruth, num_classes=2, average='micro'),
        'accuracy': multiclass_accuracy(pred, groundtruth, num_classes=2),
        'precision': multiclass_precision(pred, groundtruth, num_classes=2),
        'recall': multiclass_recall(pred, groundtruth, num_classes=2),
    }
    print(eval_res)
    confusion_matrix = multiclass_confusion_matrix(pred, groundtruth, num_classes=2)
    print(confusion_matrix)
    print(pred)
    print(groundtruth)
    

def train_main():
    config = get_default_config()
    device = config.device
    os.environ['CUDA_VISIBLE_DEVICES']=config.cuda_id
    
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
        for x, y in train_data:
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break
        eval_res = eval_main(model, dev_data)
        break   


if __name__ == '__main__':
    train_main()
