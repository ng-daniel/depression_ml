import math
import torch
from torch import nn

def log_skip_zeroes(x):
    if x == 0:
        return 0
    return math.log10(x)

def print_model_performance_table(model_performance_dicts: list):
    '''
    Each Column is 10 spaces across
    '''
    COL_WIDTH = 8
    NUM_COLS = 7
    
    #calculate max name length
    max_name_length = 0
    for d in model_performance_dicts:
        name_len = len(d['model_name'])
        if name_len > max_name_length:
            max_name_length = name_len

    NAME_COL_WIDTH = max_name_length + 2

    print('-'*COL_WIDTH*(NUM_COLS-1) + '-'*NAME_COL_WIDTH)
    print(f"{'name':{NAME_COL_WIDTH}}{'#samples':>{COL_WIDTH}}{'loss':>{COL_WIDTH}}{'acc':>{COL_WIDTH}}{'prec':>{COL_WIDTH}}{'recall':>{COL_WIDTH}}{'f1':>{COL_WIDTH}}\n")
    for d in model_performance_dicts:
        print(f"{d['model_name']:{NAME_COL_WIDTH}}{d['num_samples']:>{COL_WIDTH}}{d['loss']:>{COL_WIDTH}}{d['accuracy']:>{COL_WIDTH}}{d['precision']:>{COL_WIDTH}}{d['recall']:>{COL_WIDTH}}{d['f1score']:>{COL_WIDTH}}")
    print('-'*COL_WIDTH*(NUM_COLS-1) + '-'*NAME_COL_WIDTH)