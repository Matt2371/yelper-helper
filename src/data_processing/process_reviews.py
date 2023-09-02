from src.data_processing.word_tokenizer import *

def process_reviews_w2v(reviews, model):
    """ 
    Data processing pipeline for reviews. Returns tensor of word2vec embeddings for each review.
    Params:
    reviews -- collection (array/list) of full review strings
    model -- gensim word2vec model

    Pipeline:
    1. Tokenize each review in reviews, store as list of list
    2. Get word2vec embeddings for each token in each review, as torch tensor (# reviews, max_length, 300)
    3. RETURN word2vec embedded tensor (# reviews, max_length, 300)
    """
    # 1. Tokenize each review in reviews, store as list of list
    review_list = [basic_tokenizer(review) for review in reviews]


    # 2. Get word2vec embeddings for each token in each review, as torch tensor (# reviews, max_length, 300)
    x_all = batch_embedding(review_list, model)

    return x_all


