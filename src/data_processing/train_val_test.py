import numpy as np
import pandas as pd

def train_val_test(data, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    """
    Split the data into training, validation, and test sets.
    Parameters:
    data -- ndarray (# reviews, *).
    train_frac, val_frac, test_frac -- proportion of the data for each set
    Returns: 
    (train, val, test) datasets of shape (# reviews, *)
    """
    assert train_frac + val_frac + test_frac == 1

    timesteps = data.shape[0] # number of reviews
    train_size = int(round(timesteps * train_frac))
    val_size = int(round(timesteps * val_frac))

    train, val, test = data[:train_size], data[train_size:train_size+val_size], data[train_size+val_size:]
    return train, val, test


