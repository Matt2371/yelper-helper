import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import transformers

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
        x, (hn, cn) = self.LSTM(x)  # x is shape (# of reviews, max review length, 2 * hidden_size)
        # make prediction using final hidden state from last layer
        hn_last = x[:, -1, :] # shape (# of reviews, 2 * hidden_size)
        hn_last = self.linear(hn_last) # (# of reviews, 4)
        out = self.softmax(hn_last) # output probabilities, shape: (# of reviews in batch, 4)
        return out
    
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_layer = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 4)
        self.softmax = torch.nn.Softmax(dim=1) 
    
    def forward(self, ids, mask, token_type_ids):
        bert_output = self.bert_layer(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=True)
        bert_hidden = bert_output.pooler_output # pooling output, (batch_size, hidden_size)
        out = self.dropout(bert_hidden) # (batch_size, hidden_size)
        out = self.linear(out)  # (batch_size, output_size)
        out = self.softmax(out) # (batch_size, output_size)
        return out