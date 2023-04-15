from train_v2 import *


if __name__ == '__main__':
    cuda_id = ManageGPUs.get_free_gpu(target_mem_mb=9000)
    print(f'device: cuda {cuda_id}')
    warnings.filterwarnings('ignore')
    

def just_test_main():
    config = Configv2() 
    config.version = 'just test'
    config.epochs = 5
    config.batch_size = 8
    config.rdrop = 2
    config.early_dropout = 2
    config.just_test = True
    config.share_encoder = True
    main(config)
    config.share_encoder = False
    main(config)
    exit()

def baseline():
    config = Configv2()
    config.version = 'baseline'
    main(config)

def display():
    config = Configv2()
    config.version = 'display'
    config.share_encoder = False
    config.cls_target = 'hd+cv+vo'
    config.rdrop = True
    main(config)

def best_model():
    config = Configv2()
    config.version = 'best'
    config.model_name = 'roberta-base'
    config.rdrop = 4
    config.early_dropout = 3
    config.amp = True
    config.deepspeed = True
    config.freeze_encoder = False
    main(config)

def model_encoder_cmp():
    model_name_lst = [
        'bert-base-uncased', 
        'distilbert-base-uncased',
        'roberta-base',
        'xlm-roberta-base'
    ]
    config = Configv2()
    config.version = 'encoder of model cmp'
    for model_name in model_name_lst:
        config.model_name = model_name
        main(config)

def structure_cmp():
    config = Configv2()
    config.version = 'structure cmp'
    config.share_encoder = True
    config.positive_ratio = 2/3
    main(config)
    config.share_encoder = False
    config.positive_ratio = 0.4
    for cls_tar in ['hd', 'cv', 'vo', 'hd+cv+vo']:
        config.cls_target = cls_tar
        main(config)

def downsample_cmp():
    config = Configv2()
    config.version = 'downsample cmp'
    config.downsample_data = False
    main(config)
    config.downsample_data = True
    for rate in range(1, 11):
        config.positive_ratio = rate/10
        main(config)

def rdrop_cmp():
    config = Configv2()
    config.version = 'rdrop cmp'
    config.rdrop = None
    main(config)
    for p_rdrop in range(1, 6):
        config.rdrop = p_rdrop
        main(config)

def early_dropout_cmp():
    config = Configv2()
    config.version = 'early dropout cmp'
    config.early_dropout = None
    main(config)
    for start_epoch in range(1, 10):
        config.early_dropout = start_epoch
        main(config)

def running_time_ablation():
    config = Configv2()
    config.version = 'running time ablation'
    for p in range(4):
        config.amp = p % 2
        config.deepspeed = p // 2 % 2
        main(config)

def freeze_encoder_ablation():
    config = Configv2()
    config.version = 'freeze encoder ablation'
    config.freeze_encoder = True
    config.deepspeed = False
    main(config)

'''
nohup python train_v2.py &
python train_v2.py
'''
# just_test_main()
# baseline() # 1
# display() 
# best_model()

# model_encoder_cmp() # 4
# structure_cmp() # 5
# downsample_cmp() # 11
# rdrop_cmp() # 6
# early_dropout_cmp() # 10
# running_time_ablation() # 4
# freeze_encoder_ablation() # 1

