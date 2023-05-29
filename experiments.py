from train_v2 import *


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
    config.version = 'display_v4'
    config.share_encoder = False
    for cls_t in 'hd cv vo'.split():
        config.cls_target = cls_t
        config.rdrop = True
        main(config)

def best_model():
    config = Configv2()
    config.version = 'best'
    config.downsample_data = True
    config.positive_ratio = 0.3
    config.amp = True
    config.deepspeed = True
    config.freeze_encoder = False
    config.cls_target = 'hd+cv+vo'

    model_name_lst = [
        'bert-base-uncased', 
        'roberta-base',
        'xlnet-base-cased',
    ]
    for model_name in model_name_lst:
        for rdrop in range(1, 6):
            for early_dropout in range(1, 6):
                config.model_name = model_name
                config.rdrop = rdrop
                config.early_dropout = early_dropout
                main(config)

def model_encoder_cmp():
    model_name_lst = [
        'bert-base-uncased', 
        'distilbert-base-uncased',
        'roberta-base',
        'xlm-roberta-base',
        'albert-base-v2',
        # 'microsoft/deberta-v3-base',
        'facebook/bart-base',
        'xlnet-base-cased',
        'google/electra-base-discriminator',
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
    config.downsample_data = False
    main(config)
    config.share_encoder = False
    config.downsample_data = False
    for cls_tar in ['hd', 'cv', 'vo']:
        config.cls_target = cls_tar
        main(config)
    
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
    config.version = 'rdrop cmp v2'
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
    config.epochs = 1
    config.amp = 1
    config.deepspeed = 1
    config.just_test = 1
    main(config)

def freeze_encoder_ablation():
    config = Configv2()
    config.version = 'freeze encoder ablation'
    config.freeze_encoder = True
    config.deepspeed = False
    main(config)


# clear_error_log()

# just_test_main()
# baseline() # 1
# display() 
# best_model()

# model_encoder_cmp() # 4
# structure_cmp() # 5
# downsample_cmp() # 11
rdrop_cmp() # 6
# early_dropout_cmp() # 10
# running_time_ablation() # 4
# freeze_encoder_ablation() # 1

