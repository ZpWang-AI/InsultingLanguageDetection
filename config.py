class Config:
    version = 'base'
    
    model_name = 'xlm-roberta-base'
    device = 'cpu'
    cuda_id = '0'
    
    base = True
    just_test = False
    
    epochs = 10
    batch_size = 8
    
    
def get_default_config():
    return Config()

