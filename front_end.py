import torch
import torch.nn as nn
import gradio as gr

from config import CustomConfig
from model.baseline import BaselineModel


def main():
    config = CustomConfig()
    config.device = 'cpu'
    model = BaselineModel(config)
    model_file = './saved_res/2023-03-22_11_04_56_baseline/saved_model/2023-03-22_11_04_56_epoch2_304.pth'
    model.load_state_dict(torch.load(model_file))
    model.eval()
    model.to(config.device)
    
    with torch.no_grad():
        a = model.predict(['12312'])
    print(a)
    
    # def interface_fn(sentence):
    #     with torch.no_grad():
    #         res = model.predict([sentence])
    #     return str(res)
    #     # return "Hello " + name + "!"

    # demo = gr.Interface(fn=interface_fn, inputs="text", outputs="text")

    # demo.launch()   
    

if __name__ == '__main__':
    main()