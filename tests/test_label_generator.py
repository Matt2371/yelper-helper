import unittest
import numpy as np
from src.data_processing.process_labels import *

class test_label_generator(unittest.TestCase):
    def test_init_label_generator(self):
        '''Test to see if the class object is initialized correctly'''
        input_food = np.array([0,1,0,1,np.nan])
        input_service = np.array([1,1,0,0,np.nan])
        label_gen = label_generator(input_food, input_service)
        np.testing.assert_array_equal(label_gen.food_labels, input_food)    # match the raw food labels
        np.testing.assert_array_equal(label_gen.service_labels, input_service)  # match the raw service labels
        self.assertEqual(label_gen.is_trimmed, False) # always initialize as False
        self.assertIsNone(label_gen.result) # always initilize result as None

    def test_trim_unlabeled_reviews(self):
        '''Test to see if the class object attributes are trimmed correctly'''
        input_food = np.array([0,1,0,1,np.nan,np.nan,np.nan])
        input_service = np.array([1,1,0,0,np.nan,np.nan,np.nan])
        label_gen = label_generator(input_food, input_service)
        label_gen.trim_unlabeled_reviews()      # trimming reviews
        expected_food = np.array([0,1,0,1])
        expected_service = np.array([1,1,0,0])
        np.testing.assert_array_equal(label_gen.food_labels, expected_food) # matched the trimmed output for food
        np.testing.assert_array_equal(label_gen.service_labels, expected_service) # matched the trimmed output for service

    def test_get_labels_raise_exception(self):
        '''Test if get_labels() raises exception correctly with the correct error message when trimmed status is False'''
        input_food = np.array([0,1,0,1,np.nan,np.nan,np.nan])
        input_service = np.array([1,1,0,0,np.nan,np.nan,np.nan])
        label_gen = label_generator(input_food, input_service)
        with self.assertRaises(Exception) as context:
            label_gen.get_labels()
        self.assertEqual(str(context.exception), 'Labels need to be trimmed first')

    def test_get_labels_pass(self):
        '''Test if get_labels() correctly gets the label when the trimmed status is True'''
        input_food = np.array([0,1,0,1])
        input_service = np.array([1,1,0,0])
        label_gen = label_generator(input_food, input_service)
        label_gen.is_trimmed = True # manually set it here 
        label_gen.get_labels()  # get labels
        expected = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
        np.testing.assert_array_equal(label_gen.result, expected)

    def test_trim_and_fetch_labels(self):
        '''Test if the labels are properly computed [only food, only service, food and service, neither]'''
        input_food = np.array([0,1,0,1,np.nan,np.nan,np.nan])
        input_service = np.array([1,1,0,0,np.nan,np.nan,np.nan])
        label_gen = label_generator(input_food, input_service)

        expected = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
        np.testing.assert_array_equal(label_gen.trim_and_fetch_labels(), expected)

if __name__ == '__main__':
    unittest.main()