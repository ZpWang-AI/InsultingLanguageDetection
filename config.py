class CustomConfig:
    version = 'bertBase'
    
    model_name = 'bert-base-uncased'
    device = 'cpu'
    cuda_id = '0'
    
    train_data_file = ''
    dev_data_file = ''
    test_data_file = ''
    pretrained_model_fold = './saved_model'
    save_res_fold = './saved_res'
    
    test_model_path = ''
    
    base = True
    freeze_encoder = False
    downsample_data = True
    positive_ratio = 0.5
    share_encoder = False
    cls_target = 'hd'  # hd cv vo hd+vo hd+cv+vo..
    clip = False
    just_test = False
        
    epochs = 10
    batch_size = 8
    save_model_epoch = 1
    pb_frequency = 10
    train_ratio = 0.8
    lr = 5e-5
    
    def as_list(self):
        return [[attr, getattr(self, attr)] for attr in dir(self)
                if not callable(getattr(self, attr)) and not attr.startswith("__")]
    
    def as_dict(self):
        return dict(self.as_list())
   
    
if __name__ == '__main__':
    sample_config = CustomConfig()
    print(sample_config.as_list())
    print(sample_config.as_dict())
    pass

