import unittest
import torch
import torch.nn as nn
import numpy as np
from src.models.model_zoo import *

class test_forward(unittest.TestCase):
    '''Test LSTM model's forward function'''
    def test_forward_LSTMmodel(self):
        '''Test the shape of the output of LSTMmodel'''
        torch.manual_seed(1)
        input = torch.rand((10, 20, 30)) # input shape: (# of reviews in batch, max review length, 300)
        
        model = LSTMmodel(input_size = 30, hidden_size = 2, output_size = 4, num_layers = 2, dropout_prob = 0.5)
        expected = (10, 4)
        self.assertEqual(tuple(model(input).shape), expected)


if __name__ == '__main__':
    unittest.main()