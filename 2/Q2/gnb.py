import numpy as np

def stats(data):
    mean = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    std = np.std(data, axis=0)
    return (mean, std, variance)

