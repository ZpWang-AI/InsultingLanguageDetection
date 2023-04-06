import torch
import torch.nn as nn
import gradio as gr

from config import CustomConfig
from model.baseline import BaselineModel


def main():
    config = CustomConfig()
    config.device = 'cpu'
    config.share_encoder = True
    
    model = BaselineModel(config)
    model_file_mixed = './saved_res/2023-04-05_21_59_16_bertBase-hd+cv+vo/saved_model/2023-04-05_21_59_16_epoch1_518.pth'
    model_file_shareEncoder = './saved_res/2023-04-05_22_12_45_bertBase-shareEncoder/saved_model/2023-04-05_22_12_45_epoch7_292.pth'
    model_file_hd = './saved_res/2023-04-05_22_21_21_bertBase-hd/saved_model/2023-04-05_22_21_21_epoch3_446.pth'
    model_file_cv = './saved_res/2023-04-05_22_31_22_bertBase-cv/saved_model/2023-04-05_22_31_22_epoch3_203.pth'
    model_file_vo = './saved_res/2023-04-05_22_33_53_bertBase-vo/saved_model/2023-04-05_22_33_53_epoch8_469.pth'
    model.load_state_dict(torch.load(model_file_shareEncoder))
    model.eval()
    model.to(config.device)
    
    def interface_fn(sentence):
        with torch.no_grad():
            res = model.predict([sentence])
        # res = sentence+'hello'
        return str(res)
        # return "Hello " + name + "!"

    demo = gr.Interface(fn=interface_fn, inputs="text", outputs="text")

    demo.launch()   
    

if __name__ == '__main__':
    main()