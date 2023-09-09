import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from src.models.model_zoo import *


### FUNCTIONS TO EVALUATE MODEL PERFORMANCE ####

def multi_performance(y_true, y_pred, classes):
    """
    Returns a dataframe that summarizes performance metrics (accuracy, f1, precision, recall)
    for each class and the overall score (which returns a weighted average of all the scores 
    based on the number of true instances)

    y_true: np.array(n, #classes); true class labels
    y_pred: np.array(n, #classes); predicted class labels
    classes: list of class labels (from label_encoder.classes_ for example)
    """
    result = np.empty(shape=(3, len(classes)), dtype=np.dtype(object))
    # F1 scores
    result[0] = f1_score(y_true, y_pred, average=None)
    # Precision
    result[1] = precision_score(y_true, y_pred, average=None)
    # Recall
    result[2] = recall_score(y_true, y_pred, average=None)                       
    # Convert result to pandas df
    df = pd.DataFrame(result, columns=classes, index=['f1', 'precision', 'recall'])

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return df, accuracy