class Config:
    version = 'base'
    
    model_name = 'xlm-roberta-base'
    device = 'cpu'
    cuda_id = '0'
    
    base = True
    clip = False
    
    epochs = 10
    batch_size = 1
    
    
def get_default_config():
    return Config()


def get_clip_config():
    config = Config()
    config.version = 'clip'
    config.base = False
    config.clip = True
    return config
