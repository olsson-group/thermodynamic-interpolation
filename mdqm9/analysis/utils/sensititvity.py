import numpy as np


def filter_iqr(x: np.array, k=10):
    if k == None:
        return np.ones(x.shape, dtype=bool)
    
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    lower_bound = q25 - k * iqr
    upper_bound = q75 + k * iqr
    return (x > lower_bound) & (x < upper_bound)
