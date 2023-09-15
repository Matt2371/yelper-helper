import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
from transformers import BertTokenizer

from src.data_processing.process_labels import *
from src.data_processing.process_reviews import *
from src.data_processing.train_val_test import train_val_test
from src.models.model_evalaute import *
from src.models.model_zoo import *
from src.models.model_train import *

### DATA PROCESSING ###
# Read data
df = pd.read_csv('data/raw_reviews/reviews_v1.csv')
# Separate reviews and labels
food_labels = df.food
service_labels = df.service
y = label_generator(food_labels=food_labels.values, 
                    service_labels=service_labels.values).trim_and_fetch_labels()
X = df.text # review text
X = X[:len(y)] # trim X based on number of available reviews

X_train, X_test, _ = train_val_test(data=X, train_frac=0.8, val_frac=0.2, test_frac=0)
y_train, y_test, _ = train_val_test(data=y, train_frac=0.8, val_frac=0.2, test_frac=0)
# Get Bert encodings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Load Bert tokenizer
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')

class CustomDataset(Dataset):
    def __init__(self, encodings):
        """
        Params: 
        encodings -- dictionary, contains 'input_ids', 'token_type_ids', 'attention_mask'
        """
        self.input_ids = encodings['input_ids'] # tensor of shape (# reviews, max review length)
        self.token_type_ids = encodings['token_type_ids'] # tensor of shape (# reviews, max review length)
        self.attention_mask = encodings['attention_mask'] # tensor of shape (# reviews, max review length)
        return
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return {
            'ids' : self.input_ids[index, :],
            'mask' : self.attention_mask[index, :],
            'token_type_ids' : self.token_type_ids[index, :],
        }
    
# Create dataloader and dataset for test data
dataset_test = CustomDataset(encodings=test_encodings)
dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False)

### EVALUATE PERFORMANCE ON TEST SET ###
# instatiate model
bert_model = BERTClass()
# load saved parameters
bert_model.eval()
bert_model.load_state_dict(torch.load('src/models/saved_models/bert_fine_tuned.pt'))


# Initialize list to contain model outputs for each batch
y_scores_list = []

# Evaluate BERT model
with torch.no_grad():
    for data in tqdm(dataloader_test, desc='loading batches: '):
        # cast data types
        ids = data['ids'].to(dtype = torch.long)
        mask = data['mask'].to(dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(dtype = torch.long)
        # get softmax probabilities for current batch
        y_scores_batch = bert_model(ids, mask, token_type_ids) # shape (batch size, timesteps, 4)
        # save results
        y_scores_list.append(y_scores_batch)
    
    # concat batches to get softmax scores for all reviews
    y_scores = torch.cat(y_scores_list, dim=0) # (# test reviews, timesteps, 4)
        

# convert softmax probabilities to one-hot predicted labels
pred_labels = torch.argmax(y_scores, dim=1)
y_pred = torch.nn.functional.one_hot(pred_labels, num_classes=4)
# get performance metrics
classes = ['only food', 'only service', 'both', 'neither']
performance_df, accuracy = multi_performance(y_true=y_test, y_pred=y_pred, classes=classes)
# save results as csv
performance_df.to_csv('results/finetune_bert_performance_metrics.csv')
print(f'Overall accuracy for fine-tuned BERT is: {accuracy}')