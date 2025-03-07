import math

def log_skip_zeroes(x):
    if x == 0:
        return 0
    return math.log10(x)