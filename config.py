class Config:
    version = 'base'
    
    model_name = 'xlm-roberta-base'
    device = 'cpu'
    cuda_id = '0'
    
    saved_model_fold = 'saved_model'
    saved_result_file = 'saved_res.txt'
    
    base = True
    just_test = False
    
    epochs = 10
    batch_size = 8
    
    
def get_default_config():
    return Config()

