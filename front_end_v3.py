import os
import torch
import torch.nn as nn
import gradio as gr
import yaml
# import lightning

from pathlib import Path as path
from transformers import AutoTokenizer
from lightning.pytorch.utilities.deepspeed import get_fp32_state_dict_from_zero_checkpoint

# from utils import *
from model.model_v2 import Modelv2


def main():
    # cuda_id = get_free_gpu()
    # device = torch.device(f'cuda:{cuda_id}')
    device = torch.device('cpu')
    
    cls_lst = ['Assaults on Human Dignity', 'Call for Violence', 'Vulgarity/Ofensive Language directed at an individual']
    # hd: Assaults on Human Dignity 侵犯人类尊严
    # cv: Call for Violence 呼吁暴力
    # vo: Vulgarity/Ofensive Language directed at an individual 针对个人的粗俗/冒犯性语言
    ckpt_fold_lst = []
    config_lst = []
    model_lst = []

    for ckpt_fold in ckpt_fold_lst:
        ckpt_fold = path(ckpt_fold)
        config_file = ckpt_fold/'hparams.yaml'
        with open(config_file, 'r', encoding='utf-8')as file:
            config_dic = yaml.load(file, Loader=yaml.FullLoader)
        config_lst.append(config_dic)
        
        model = Modelv2(
            model_name=config_dic['model_name'],
            share_encoder=config_dic['share_encoder'],
        )
        for son_file in os.listdir(ckpt_fold):
            son_file = ckpt_fold/son_file
            if son_file.is_dir():
               model_ckpt_fold = son_file
               break
        else:
            raise 'no state dict of model'
        state_dict = get_fp32_state_dict_from_zero_checkpoint(model_ckpt_fold)
        new_state_dict = {k.replace('_forward_module.', ''):v for k,v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        model_lst.append(model)
        
    tokenizer = AutoTokenizer.from_pretrained(config_dic['model_name'], cache_dir=config_dic['pretrained_model_fold'])
    
    def interface_fn(sentences):
        with torch.no_grad():
            x_input = tokenizer(sentences.split('\n'), padding=True, truncation=True, return_tensors='pt')
            x_input = x_input.to(device)
            for cls_name, model in zip(cls_lst, model_lst):
                res = model.predict(x_input)[0]
            # return res
        res = bool(res.cpu().numpy())
        # res = sentence+'hello'
        if res:
            return '存在谩骂类情感'
        else:
            return '不存在谩骂类情感'
        # return str(res)
    
    
    examples = [
        'Hello, Jack! Nice to meet you.\n'
        'Nice to meet you, too!',
        'Fuck you idiot!',
        'Holy shit! You such a bitch just got my shoes dirty.\n'
        'Oh, sorry, I apologize.\n'
        'Well. Watch out next time.'
    ],

    demo = gr.Interface(
        fn=interface_fn, 
        inputs=gr.inputs.Textbox(lines=2, label="Inputs(English)"), 
        outputs=gr.outputs.Textbox(label="Outputs"),
        examples=examples,
    )

    demo.launch(share=True)   
    

if __name__ == '__main__':
    main()