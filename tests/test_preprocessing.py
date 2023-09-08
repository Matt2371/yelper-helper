import unittest
import numpy as np
import spacy
import pandas as pd
from pandas import testing

from src.data_processing.preprocess_reviews import *

class test_preprocessing(unittest.TestCase):
    '''Test preprocessing of reviews to remove all stop words and punctuations'''
    def test_remove_stop_punc(self):
        '''Test to remove stop words and punctuation from one review'''
        nlp = spacy.load('en_core_web_sm')  #load the english model
        review = "the food  was great!!"
        expected = 'food great'
        self.assertEqual(remove_stop_punc(review, model = nlp), expected)

    def test_preprocess_reviews(self):
        '''Test to see if it successfully preprocess the entire data'''
        nlp = spacy.load('en_core_web_sm')  #load the english model
        input = pd.Series(["the food was delicious!", "the  service really suck."])
        expected = pd.Series(["food delicious", "service suck"])
        testing.assert_series_equal(preprocess_reviews(input, model = nlp), expected)


if __name__ == '__main__':
    unittest.main()