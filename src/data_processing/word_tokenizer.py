#### IMPORT PACKAGES ####
import re
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api
from tqdm import tqdm

#### FUNCTIONS TO TOKENIZE WORDS ####
def basic_tokenizer(input):
    """Tokenize input string (after converting to lowercase) into list of words. Ignores punctuation (,.!?)"""
    # lowercase input string
    input = input.lower()
    # tokenize
    tokens = re.split("\\s+", input)
    tokens = [re.sub("[.?,!]", "", word) for word in tokens]

    return tokens

def to_word2vec(token, model, unk=0):
    """
    Convert token to word2vec embedding. If 
    Params: 
    token -- str, the input token
    model -- gensim word2vec model (dimension 300)
    unk -- special value for unknown words
    Return:
    output -- 1darray (300,), word2vec embedding
    """
    try:
        output = model[token]
    except:
        output = unk * np.ones(300)

    return output

def list_embedding(input, model):
    """
    Get embeddings for a list of tokens
    Params: 
    input -- list of tokens
    model -- gensim word2vec model (dimension 300)
    Return:
    output -- PyTorch tensor of shape (timesteps (review length), 300)
    """

    embeddings = []
    for token in input:
        # get embedding
        embedding = to_word2vec(token, model).reshape(1, -1)
        # convert embedding to torch tensor
        embedding = torch.tensor(embedding, dtype=torch.float)
        embeddings.append(embedding)
    
    output = torch.cat(embeddings, dim=0) # (review length, 300)
    return output

def batch_embedding(batch_input, model, pad_value=0):
    """
    Get embeddings for batch of reviews. Pads reviews so that each is of same length
    Params:
    batch_input -- list of reviews where each review is a list of tokens
    model -- gensim word2vec model (dimension 300)
    pad_value -- value to pad review sequences with
    Return:
    output -- PyTorch tensor of shape (#reviews, max review length, 300)
    """
    embedding_tensors = [] # list of embedding tensors for each review
    for review in tqdm(batch_input, desc='Fetching review embeddings: '):
        embedding_tensors.append(list_embedding(review, model))
    
    # pad review lengths and combine reviews into pytorch tensor
    output = pad_sequence(embedding_tensors, batch_first=True, padding_value=pad_value)

    return output # (# of reviews, max review length, 300)
