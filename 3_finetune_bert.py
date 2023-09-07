import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import optim
import transformers
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_processing.process_labels import *
from src.data_processing.train_val_test import train_val_test
from src.models.model_zoo import *
from src.models.model_train import *

### DATA PROCESSING ###
# Read data
df = pd.read_csv('data/raw_reviews/reviews_v1.csv')
# Separate reviews and labels
X = df.text # review text
food_labels = df.food
service_labels = df.service
y = label_generator(food_labels=food_labels.values, 
                    service_labels=service_labels.values).trim_and_fetch_labels()
# Trim reviews to size of labels (y)
X = X[:len(y)].copy()

X_train, X_test, _ = train_val_test(data=X, train_frac=0.8, val_frac=0.2, test_frac=0)
y_train, y_test, _ = train_val_test(data=y, train_frac=0.8, val_frac=0.2, test_frac=0)

# Load Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Get Bert encodings
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')

# Create custom PyTorch datasets
class CustomDataset(Dataset):
    def __init__(self, encodings, targets):
        """
        Params: 
        encodings -- dictionary, contains 'input_ids', 'token_type_ids', 'attention_mask'
        targets -- Pytorch tensor of shape (# reviews, 4), one-hot labels
        """
        self.input_ids = encodings['input_ids'] # tensor of shape (# reviews, max review length)
        self.token_type_ids = encodings['token_type_ids'] # tensor of shape (# reviews, max review length)
        self.attention_mask = encodings['attention_mask'] # tensor of shape (# reviews, max review length)
        self.targets = targets # tensor of shape (# reviews, 4)
        return
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return {
            'ids' : self.input_ids[index, :],
            'mask' : self.attention_mask[index, :],
            'token_type_ids' : self.token_type_ids[index, :],
            'targets' : self.targets[index, :]
        }
# Create training and testing PyTorch datasets
training_set = CustomDataset(train_encodings, y_train)
testing_set = CustomDataset(test_encodings, y_test)
# Create PyTorch dataloaders
train_params = {'batch_size': 32,
                'shuffle': True}

test_params = {'batch_size': 32,
                'shuffle': True}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

### FINETUNE BERT MODEL ###
# instantiate model
bert_model = BERTClass()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert_model.parameters(), lr=0.00002)
# run training loop
train_loss_list, val_loss_list = bert_training_loop(bert_model=bert_model, criterion=criterion, 
                                                    optimizer=optimizer, patience=1, dataloader_train=training_loader, 
                                                    dataloader_test=testing_loader, epochs=4, early_stop=False)

print(f'Training Complete! Final val error:{val_loss_list[-1]}, epochs trained: {len(val_loss_list)}')
plot_train_val(train_loss_list=train_loss_list, val_loss_list=val_loss_list)
plt.savefig(f'jobs/bert_fine_tuning.png')

# save trained model
torch.save(bert_model.state_dict(), f'jobs/bert_fine_tuned.pt')