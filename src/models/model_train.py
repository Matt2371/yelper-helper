import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#### LSTM TRAINING FUNCTIONS ####
def train_one_epoch(model, criterion, optimizer, dataloader_train):
    '''
    Train one epoch (runs through every training example)
    Param:
    model -- Pytorch model
    criterion -- loss function
    optimizer -- Pytorch optimizer
    dataloader_train -- Pytorch dataloader for training data
    Return:
    training loss (avg. for each minibatch) for the epoch
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

    avg_loss = total_loss / len(dataloader_train) # average over number of minibatches
    return avg_loss

def val_one_epoch(model, criterion, dataloader_val):
    '''
    Calcuate validation loss for one epoch
    Param:
    model -- Pytorch model
    criterion -- loss function
    optimizer -- Pytorch optimizer
    dataloader_val -- Pytorch dataloader for validation data
    Return:
    validation loss (avg. for each minibatch) for the epoch
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
    
def training_loop(model, criterion, optimizer, patience, dataloader_train, dataloader_val, epochs):
    '''
    Training loop with EarlyStopper
    Params:
    model -- Pytorch model
    criterion -- loss function
    optimizer -- Pytorch optimizer
    patience -- number of patience epochs to call EarlyStopper
    dataloader_train -- Pytorch dataloader for training data
    dataloader_val -- Pytorch dataloader for validation data
    epoch -- defined maximum number off training epochs

    Returns:
    train_losses -- avg training losses for each epoch
    val_losses -- avg validation losses for each epoch
    '''
    num_epochs = epochs
    train_loss_list = []
    val_loss_list = []
    early_stopper = EarlyStopper(patience = patience)
    for epoch in tqdm(range(num_epochs), desc = "Training epochs: "):
        train_loss = train_one_epoch(model, criterion, optimizer, dataloader_train)
        train_loss_list.append(train_loss)
        val_loss = val_one_epoch(model, criterion, dataloader_val)
        val_loss_list.append(val_loss)
        
        if early_stopper.early_stop(validation_loss=val_loss):
            break
    return train_loss_list, val_loss_list

#### FUNCTIONS TO FINE TUNE BERT ####

def bert_train_one_epoch(bert_model, criterion, optimizer, dataloader_train):
    """ 
    Fine tune BERT model for sequence classification for ONE EPOCH
    Params:
    bert_model -- custom torch model to finetune BERT
    criterion -- torch criterion object to evaluate loss
    optimizer -- torch optimizer object
    dataloader_train -- torch dataloader for train set, each minibatch is a dictionary containing 
                        ['ids', 'mask', 'token_type_ids', 'targets']
    """
    bert_model.train()
    total_loss = 0  #initialize total loss
    for data in dataloader_train: # data is a dictionary
        optimizer.zero_grad()

        # cast data types
        ids = data['ids'].to(dtype = torch.long)
        mask = data['mask'].to(dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(dtype = torch.long)
        targets = data['targets'].to(dtype = torch.float)

        # forward pass
        outputs = bert_model(ids, mask, token_type_ids)

        # calculate loss over batch
        loss = criterion(outputs, targets)

        # backward pass and optimization step
        loss.backward() # find gradients
        optimizer.step() # updating parameters

        # update total loss
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader_train) # average over number of minibatches
    return avg_loss

def bert_val_one_epoch(bert_model, criterion, dataloader_test):
    """ 
    Validate fine tuning BERT model for ONE EPOCH
    Params:
    bert_model -- custom torch model to finetune BERT
    criterion -- torch criterion object to evaluate loss
    dataloader_test -- torch dataloader for test set, each minibatch is a dictionary containing 
                        ['ids', 'mask', 'token_type_ids', 'targets']
    """
    bert_model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for data in dataloader_test:
            # cast data types
            ids = data['ids'].to(dtype = torch.long)
            mask = data['mask'].to(dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(dtype = torch.long)
            val_targets = data['targets'].to(dtype = torch.float)

            val_outputs = bert_model(ids, mask, token_type_ids)
            val_loss = criterion(val_outputs, val_targets)
            total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(dataloader_test)
    
    return avg_val_loss


def bert_training_loop(bert_model, criterion, optimizer, patience, dataloader_train, dataloader_test, epochs, early_stop=False):
    '''
    Training loop with EarlyStopper
    Params:
    model -- custom Pytorch BERT model
    criterion -- loss function
    optimizer -- Pytorch optimizer
    patience -- number of patience epochs to call EarlyStopper
    dataloader_train -- Pytorch dataloader for training data
    dataloader_val -- Pytorch dataloader for validation data
    epoch -- defined maximum number off training epochs
    early_stop -- bool, whether or not to implement early stopping using the validation set

    Returns:
    train_losses -- avg training losses for each epoch
    val_losses -- avg validation losses for each epoch
    '''
    num_epochs = epochs
    train_loss_list = []
    val_loss_list = []
    early_stopper = EarlyStopper(patience = patience)
    for epoch in tqdm(range(num_epochs), desc = "Training epochs: "):
        train_loss = bert_train_one_epoch(bert_model=bert_model, criterion=criterion, 
                                          optimizer=optimizer, dataloader_train=dataloader_train)
        train_loss_list.append(train_loss)
        val_loss = bert_val_one_epoch(bert_model=bert_model, criterion=criterion, dataloader_test=dataloader_test)
        val_loss_list.append(val_loss)
        
        if early_stop:
            if early_stopper.early_stop(validation_loss=val_loss):
                break
            else:
                continue
        else:
            continue
        
    return train_loss_list, val_loss_list


#### MISC TRAINING FUNCTIONS ####
def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopper:
    '''Callback class to implement early stopping during training loop'''
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self,validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plot_train_val(train_loss_list, val_loss_list, ax = None):
    '''
    Plot train and validation loss vs. epochs after the training loop
    Params:
    train_loss_list -- list, training losses per epoch
    val_loss_list -- list, validation losses per epoch
    ax -- matplotlib axes, if provided allow subplots
    '''

    assert len(train_loss_list) == len(val_loss_list)
    if ax is not None:
        ax.plot(train_loss_list)
        ax.plot(val_loss_list)
        ax.legend(['training loss', 'validation loss'])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
    else:
        plt.clf()
        plt.plot(train_loss_list)
        plt.plot(val_loss_list)
        plt.legend(['training loss', 'validation loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
