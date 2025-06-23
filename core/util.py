import math
import torch
from torch import nn
import numpy as np
import pandas as pd

def log_skip_zeroes(x, base : int = 10):
    return math.log(1 + x, base)

def get_time_index():
    time_index = [f'{12 + i//60:02d}' for i in range(0,720,120)]
    time_index += [f'{i//60:02d}' for i in range(0,720,120)]
    time_index += ['12']
    return time_index

def data_mean_med_std(data : pd.Series):
    d_mean = data.mean()
    d_median = data.median()
    d_std = data.std()
    return pd.Series([d_mean, d_median, d_std], ['mean', 'median', 'std'])

def subtract_corresponding_minute(data, sub_data):
    row_name = data.name
    data, sub_data = np.array(data), np.array(sub_data)
    difference = pd.Series(np.subtract(data, sub_data))
    difference.name = row_name
    return difference