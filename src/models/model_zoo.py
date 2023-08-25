import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class LSTMmodel(nn.Module):
    '''
    LSTM for learning joint distribution to classify: food, service, neither, both.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
        """
        Params:
        input_size -- number of expected features in input x, (here feature is 300)
        hidden_size -- number of features in hidden state h (will be greater than input_size)
        num_layers -- number of recurrent layers
        dropout_prob -- dropout probability (% of nodes to be turned off)
        output_size -- number of classes we are classifying

        Addtional Comments:
        Features are 300 dimension
        Batch is # of reviews in batch
        Timestep/sequence length is L in the docs
        """
        super(LSTMmodel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(output_size)

    def forward(self, x):
        x = self.LSTM(x)    # input shape: (# of reviews in batch, max review length, 300)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.softmax(x) # output shape: (# of reviews in batch, max review length, 4)
        return x