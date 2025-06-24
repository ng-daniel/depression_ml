import math
import torch
from torch import nn
import numpy as np
import pandas as pd

def log_skip_zeroes(x, base : int = 10):
    '''
    Returns the log of 1 + x (for preprocessing)
    '''
    return math.log(1 + x, base)

def data_mean_med_std(data : pd.Series):
    '''
    Calculates the mean, median, and standard deviation of a series and returns these values as a new series.
    '''
    d_mean = data.mean()
    d_median = data.median()
    d_std = data.std()
    return pd.Series([d_mean, d_median, d_std], ['mean', 'median', 'std'])

def subtract_corresponding_minute(data, sub_data):
    '''
    Subtracts `sub_data` from `data` elementwise, where each 1d arraylike object has the same number of elements.

    Args
    - data - arraylike object of the original data
    - sub_data - the values to subtract from the data

    Returns
    - series containing the result of the operation
    '''
    row_name = data.name
    data, sub_data = np.array(data), np.array(sub_data)
    difference = pd.Series(np.subtract(data, sub_data))
    difference.name = row_name
    return difference