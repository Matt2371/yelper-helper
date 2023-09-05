#### TRAIN LSTM MODEL ON ONE SET OF PARAMETERS ###
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler

from src.data_processing.process_labels import *
from src.data_processing.process_reviews import *
from src.data_processing.train_val_test import train_val_test
from src.models.model_zoo import *
from src.models.model_train import *

## PROCESS DATA
# Read data
df = pd.read_csv('data/raw_reviews/reviews_v1.csv')
# Separate reviews and labels
reviews = df.text
food_labels = df.food
service_labels = df.service
# Get target labels
y = label_generator(food_labels=food_labels.values, 
                    service_labels=service_labels.values).trim_and_fetch_labels()
# Trim reviews to size of labels (y)
reviews = reviews[:len(y)].copy()
# Get word2vec embedded reviews
model = KeyedVectors.load('word2vec/word2vec-google-news-300.model') # Load word2vec model
x_all = process_reviews_w2v(reviews=reviews, model=model) # (1000, max review length, 300)
x_all = x_all[:, :200, :] # cut reviews to only keep first 200 words
# Train/Val/Test split
x_train, x_val, x_test = train_val_test(x_all, train_frac=0.6, val_frac=0.2, test_frac=0.2)
y_train, y_val, y_test = train_val_test(y, train_frac=0.6, val_frac=0.2, test_frac=0.2)
# Create torch datasets
dataset_train, dataset_val = (TensorDataset(x_train, y_train), TensorDataset(x_val, y_val))
# Create torch dataloader (also include OVERSAMPLING of minority class)
dataloader_train, dataloader_val = (DataLoader(dataset_train, batch_size=100, shuffle=True),
                                    DataLoader(dataset_val, batch_size=100, shuffle=True))

## INSTANTIATE AND TRAIN MODEL
# instatiate model
input_size = 300
hidden_size = 300
num_layers = 2
dropout_prob = 0.3
output_size = 4
torch.manual_seed(10)
lstm_model = LSTMmodel(input_size=input_size, hidden_size=hidden_size, 
                       output_size=output_size, num_layers=num_layers, dropout_prob=dropout_prob)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.0001)
# run training loop
train_loss_list, val_loss_list = training_loop(model=lstm_model, criterion=criterion, 
                                               optimizer=optimizer, patience=300, 
                                               dataloader_train=dataloader_train, 
                                               dataloader_val=dataloader_val, epochs=100)

print(f'Training Complete! Final val error:{val_loss_list[-1]}, epochs trained: {len(val_loss_list)}')

plot_train_val(train_loss_list=train_loss_list, val_loss_list=val_loss_list)
plt.savefig(f'jobs/out_h{hidden_size}_l{num_layers}_dp0{dropout_prob * 10}.png')


# save trained model
torch.save(lstm_model.state_dict(), f'jobs/out_h{hidden_size}_l{num_layers}_dp0{dropout_prob * 10}.pt')