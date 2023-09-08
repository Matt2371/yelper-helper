import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm

def remove_stop_punc(review, model):
    """
    Remove stop words and punctuation from a single review
    Needed to convert it back to full strings from list of token objects to be compatible with our current implementation
    Params:
    review -- str, single review
    model -- spacy english model
    Return:
    cleaned_review -- str
    """
    # tokenize the review
    doc = model(review)
    # Remove both punctuation and stop words and return list of tokens (str)
    clean_review = [token.text for token in doc if not token.is_punct and not token.is_stop and not token.is_space]
    # convert list into str
    clean_review = " ".join(clean_review)
    return clean_review

def preprocess_reviews(reviews, model):
    """ 
    Data preprocessing pipeline for reviews
    Remove punctuation, stop words, and white spaces. 
    Params:
    reviews -- collection (array/list) of full review strings
    model -- spacy english model
    Returns:
    cleaned_reviews -- pandas series
    """
    reviews_list = [remove_stop_punc(review, model) for review in tqdm(reviews, desc ="Preprocessing Reviews")]
    cleaned_reviews = pd.Series(reviews_list)
    return cleaned_reviews