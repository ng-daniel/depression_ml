import math
import torch
from torch import nn

def log_skip_zeroes(x, base : int = 10):
    if x == 0:
        return 0
    return math.log(x, base)