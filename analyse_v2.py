import re
import pandas as pd
import numpy as np
import fitlog

from collections import defaultdict, OrderedDict

from utils import *
from corpus_v2 import *
from train_v2 import completed_mark_file


pd.set_option('display.max_columns', 10**9)
pd.set_option('display.max_rows', 10**9)


def filter_null(pd_data):
    return pd_data[~pd_data.isnull()]

class Analyzer:
    @staticmethod
    def get_log_folds_from_project(project_dic):
        project_dic = path(project_dic)
        log_folds = []
        for log_fold in os.listdir(project_dic):
            log_fold = project_dic/log_fold
            if completed_mark_file in os.listdir(log_fold):
                log_folds.append(log_fold)
        return log_folds
    
    @staticmethod
    def _deal_metric(log_fold, metric_data:pd.DataFrame, metric_only_f1):
        for log_file in os.listdir(log_fold):
            if 'ckpt' in log_file:
                nums = re.findall(r'\d+', log_file)
                epoch = int(nums[0])
                break
        else:
            raise f'wrong ckpt file in {log_fold}'
        
        val_macro_f1 = filter_null(metric_data['val_macro_f1']).iloc[[epoch]].to_frame().reset_index()
        if metric_only_f1:
            test_macro_f1 = filter_null(metric_data['test_macro_f1']).reset_index()
            new_metric_data = pd.concat([val_macro_f1, test_macro_f1], axis=1)
        else:
            test_metric = filter_null(metric_data.iloc[-1]).to_frame().transpose().reset_index()
            new_metric_data = pd.concat([val_macro_f1, test_metric], axis=1)
        new_metric_data = new_metric_data.drop(['step', 'index', 'epoch'], axis=1, errors='ignore')
        epoch_data = pd.DataFrame({'epoch':epoch}, index=[0])
        new_metric_data = pd.concat([epoch_data, new_metric_data], axis=1)
        return new_metric_data
    
    @staticmethod
    def get_res_from_project(
        project_dic,
        metric_only_f1=True,
        sort_data=True,
        filter_same_column=True,
        filter_running_time=False,
    ):
        project_dic = path(project_dic)
        all_res = pd.DataFrame()
        for log_fold in Analyzer.get_log_folds_from_project(project_dic):
            config_dic = load_config_from_yaml(log_fold/'hparams.yaml')
            config_dic = pd.DataFrame(config_dic, index=[0])
            metric_data = pd.read_csv(log_fold/'metrics.csv')
            metric_data = Analyzer._deal_metric(log_fold, metric_data, metric_only_f1)
            cur_res = pd.concat([config_dic, metric_data], axis=1)
            all_res = pd.concat([all_res, cur_res])
        
        if filter_same_column:
            nunique = all_res.nunique()
            def filter_fn(column_name):
                if nunique[column_name] > 1:
                    return False
                if 'file' in column_name:
                    return True
                if column_name[:3] == 'val' or column_name[:4] == 'test':
                    return False
                return True
            all_res.drop(filter(filter_fn, all_res.columns), axis=1, inplace=True, errors='ignore')
        if filter_running_time:
            all_res.drop('running time', axis=1, inplace=True, errors='ignore')
        if sort_data == True:
            all_res.sort_values(by=list(all_res.columns), inplace=True, ascending=True)
        elif sort_data != False:
            all_res.sort_values(by=list(sort_data), inplace=True, ascending=True)
        all_res.reset_index(inplace=True)
        return all_res


def get_all_res(output_path=''):
    project_name_lst = [
        'best',

        'encoder_of_model_cmp',
        'structure_cmp',
        'downsample_cmp',
        'rdrop_cmp',
        'early_dropout_cmp',
        'running_time_ablation',
        'freeze_encoder_ablation',
    ]
    all_res = OrderedDict()
    baseline_config = load_config_from_yaml('./logs/baseline/2023-04-14_16-49-30/hparams.yaml')
    baseline_config = pd.Series(baseline_config)
    all_res['baseline_config'] = baseline_config
    print(baseline_config)
    for p, project_name in enumerate(project_name_lst):
        res = Analyzer.get_res_from_project(
            path('./logs/')/project_name,
            metric_only_f1=True,
            sort_data=True,
            filter_same_column=True,
            filter_running_time=(p not in [6, 7]),
        )
        all_res[project_name] = res
        print(project_name)
        print(res)
    if output_path:
        for k in all_res:
            all_res[k].to_excel(excel_writer=output_path, sheet_name=str(k), index=False)
    return all_res
        

def get_data_info():
    def inner(data):
        data_label = data[:,1:]
        print(data_label.shape[0])
        print(np.sum(data_label, axis=0))
        print(np.sum(np.sum(data_label, axis=1) != 0))

    print('train')
    train_data = preprocess_train_data(train_data_file_list[0])
    inner(train_data)
    print('test')
    test_data = preprocess_test_data(test_data_file_list[0])
    inner(test_data)
    

def main():
    pass


if __name__ == '__main__':
    get_all_res('./_result.xlsx')