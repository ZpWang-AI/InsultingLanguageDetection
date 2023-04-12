import os
import torch
import torch.nn as nn
import gradio as gr

from transformers import AutoTokenizer
from lightning import Fabric
from lightning.pytorch.utilities.deepspeed import get_fp32_state_dict_from_zero_checkpoint

from utils import *
from model.model_v2 import Modelv2
from train_v2 import Configv2


def main():
    config = Configv2()
    cuda_id = get_free_gpu()
    device = torch.device(f'cuda:{cuda_id}')
    
    ckpt_file = './logs/2023-04-12_19-43-18/checkpoint/epoch5-f1score0.77.ckpt/'
    model = Modelv2(
        model_name=config.model_name,
        share_encoder=config.share_encoder,
    )
    model.load_state_dict(get_fp32_state_dict_from_zero_checkpoint(ckpt_file), strict=False)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.pretrained_model_fold)
    
    def interface_fn(sentence):
        with torch.no_grad():
            x_input = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
            x_input = x_input.to(device)
            res = model.predict(x_input)[0]
        res = res.cpu().numpy()
        res = bool(res)
        # res = sentence+'hello'
        if res:
            return '存在谩骂类情感'
        else:
            return '不存在谩骂类情感'
        # return str(res)

    demo = gr.Interface(fn=interface_fn, inputs="text", outputs="text")

    demo.launch()   
    

if __name__ == '__main__':
    main()