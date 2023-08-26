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
        return out
        Batch is # of reviews in batch
        Timestep/sequence length is L in the docs
        """
        super(LSTMmodel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout = dropout_prob, bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, output_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        # input shape: (# of reviews in batch, max review length, 300)
        x, (hn, cn) = self.LSTM(x)  # x is shape (# of reviews, max review length, 2 * 300)
        # make prediction using final hidden state from last layer
        hn_last = x[:, -1, :] # shape (# of reviews, 2 * 300)
        hn_last = self.linear(hn_last) # (# of reviews, 4)
        out = self.softmax(hn_last) # output probabilities, shape: (# of reviews in batch, 4)
        return out