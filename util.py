import math
import torch
from torch import nn

def log_skip_zeroes(x, base : int = 10):
    return math.log(1 + x, base)