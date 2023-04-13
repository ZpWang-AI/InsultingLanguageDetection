import os
import torch
import torch.nn as nn
import gradio as gr
import yaml
# import lightning

from transformers import AutoTokenizer
from lightning.pytorch.utilities.deepspeed import get_fp32_state_dict_from_zero_checkpoint

# from utils import *
from model.model_v2 import Modelv2


def main():
    # cuda_id = get_free_gpu()
    # device = torch.device(f'cuda:{cuda_id}')
    device = torch.device('cpu')
    
    config_file = './logs/2023-04-13_17-14-12_display/hparams.yaml'
    ckpt_file = './logs/2023-04-13_17-14-12_display/checkpoint/epoch1-f1score0.56.ckpt/'

    with open(config_file, 'r', encoding='utf-8')as file:
        config_dic = yaml.load(file, Loader=yaml.FullLoader)
    
    model = Modelv2(
        model_name=config_dic['model_name'],
        share_encoder=config_dic['share_encoder'],
    )
    state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_file)
    new_state_dict = {k.replace('_forward_module.', ''):v for k,v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config_dic['model_name'], cache_dir=config_dic['pretrained_model_fold'])
    
    def interface_fn(sentence):
        with torch.no_grad():
            x_input = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
            x_input = x_input.to(device)
            res = model.predict(x_input)[0]
            # return res
        res = bool(res.cpu().numpy())
        # res = sentence+'hello'
        if res:
            return '存在谩骂类情感'
        else:
            return '不存在谩骂类情感'
        # return str(res)

    demo = gr.Interface(
        fn=interface_fn, 
        inputs="text", 
        outputs="text",
        examples=['Hello world', 'Nice to meet you', 'Fuck you', 'Son of bitch']
    )

    demo.launch(share=True)   
    

if __name__ == '__main__':
    main()