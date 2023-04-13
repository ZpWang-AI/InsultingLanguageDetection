import pandas as pd
import numpy as np
import fitlog


def filter_nan(line_data):
    return [p for p in line_data if not np.isnan(p)]


def main():
    file = './logs/2023-04-13_14-00-09/metrics.csv'
    record = pd.read_csv(file, sep=',')
    val_macro_f1 = record['val_macro_f1']
    val_macro_f1 = val_macro_f1[~val_macro_f1.isnull()]
    print(val_macro_f1)


if __name__ == '__main__':
    main()