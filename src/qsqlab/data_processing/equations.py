import numpy as np

def binomial_error(k, N):
    return np.sqrt((k/N**2) - (k**2/N**3))