import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, criterion, optimizer, dataloader_train):
    '''
    Train one epoch (runs through every training example)
    Param:
    model -- Pytorch model
    criterion -- loss function
    optimizer -- Pytorch optimizer
    dataloader_train -- Pytorch dataloader for training data
    return training loss (avg. for each minibatch) for the epoch
    '''

    # training loop
    total_loss = 0  #initialize total loss
    model.train() # set model to training mode. e.g: dropout layer behaves differently
    for inputs, targets in dataloader_train: # inputs: (mini batch size, timesteps, 300) targets: (mini batch size, 4) the labelled outputs (Actual)
        optimizer.zero_grad() #reset the gradients of all trainable weights to 0 at every backward pass or else gradients accumalate

        # forward pass
        outputs = model(inputs) #predicts the outputs

        # calculate loss using defined loss function
        loss = criterion(outputs,targets)

        # backward pass and optimization step
        loss.backward() # find gradients
        optimizer.step() # updating parameters 

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader_train) # number of minibatches
    return avg_loss

def val_one_epoch(model, criterion, dataloader_val):
    '''
    Calcuate validation loss for one epoch
    Param:
    model -- Pytorch model
    criterion -- loss function
    optimizer -- Pytorch optimizer
    dataloader_val -- Pytorch dataloader for validation data
    return validation loss (avg. for each minibatch) for the epoch
    '''
    model.eval() # set model to evaluation mode. e.g: dropout layer behaves differently
    with torch.no_grad():
        total_val_loss = 0
        for val_inputs, val_targets in dataloader_val:
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(dataloader_val)
    return avg_val_loss