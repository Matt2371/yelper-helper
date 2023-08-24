import unittest
import numpy as np
from src.data_processing.word_tokenizer import *
from gensim.models import KeyedVectors

class test_word_tokenizer(unittest.TestCase):    
    """Test word_tokenizer.py"""
    def test_punctuation_basic_tokenizer(self):
        '''Test if punctuations such as ?., will be removed'''
        input = "I like this?? Wow."
        expected = ["i", "like", "this", "wow"]
        self.assertEqual(basic_tokenizer(input), expected)
    
    def test_apost_basic_tokenizer(self):
        '''Test if contractions are still one word'''
        input = "I don't like"
        expected = ["i", "don't", "like"]
        self.assertEqual(basic_tokenizer(input), expected)
    
    def test_unk_to_word2vec(self):
        '''Test if unkown word will return dim 300 vector of zeros'''
        model = KeyedVectors.load('word2vec/word2vec-google-news-300.model')
        input = "wooord"
        expected = np.zeros(300)
        np.testing.assert_almost_equal(to_word2vec(input, model), expected)

    def test_shape_batch_embedding(self):
        '''Test the shape of batch embedding'''
        model = KeyedVectors.load('word2vec/word2vec-google-news-300.model')
        input = [['hot', 'dog'], ['my', 'cat','disappeared'], ['this','chick','is','spicy']]
        expected = (3,4,300)
        self.assertEqual(tuple(batch_embedding(input, model, pad_value=0).shape), expected)

if __name__ == "__main__":
    unittest.main()


