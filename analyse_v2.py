import pandas as pd
import numpy as np
import fitlog

from collections import defaultdict

from utils import *
from train_v2 import completed_mark_file


def get_metric_line(file_path, metric_name):
    record = pd.read_csv(file_path, sep=',')
    val_macro_f1 = record[metric_name]
    val_macro_f1 = val_macro_f1[~val_macro_f1.isnull()]
    return val_macro_f1


def get_config_from_project(project_dic, filter_same=True):
    project_dic = path(project_dic)
    all_config_dic = defaultdict(set)
    for log_fold in os.listdir(project_dic):
        log_fold = project_dic/log_fold
        if completed_mark_file in os.listdir(log_fold):
            config_dic = load_config_from_yaml(log_fold/'hparams.yaml')
            for k in config_dic:
                all_config_dic[k].add(config_dic[k])
    if filter_same:
        all_config_dic = {k:v for k, v in all_config_dic.items() if len(v)>1}
    return all_config_dic


def main():
    pass


if __name__ == '__main__':
    project_fold = './logs/structure_cmp/'
    # for k, v in get_config_from_project(project_fold).items():
    #     print(k, v)
    # project_dic = path(project_fold)
    # for log_fold in os.listdir(project_dic):
    #     log_fold = project_dic/log_fold
    #     if completed_mark_file in os.listdir(log_fold):
    #         config_dic = load_config_from_yaml(log_fold/'hparams.yaml')
    #         if config_dic['share_encoder'] == False and '+' not in config_dic['cls_target']:
    #             print(log_fold)