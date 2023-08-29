import numpy as np
import pandas as pd
import torch

class label_generator():
    """
    Given binary food and service labels, create
    one-hot vectors of joint distribution: [only food, only service, food and service, neither]
    """
    def __init__(self, food_labels, service_labels):
        """
        Params:
        food_labels -- np.array, untrimmed binary food labels with shape (# of reviews, )
        service_labels -- np.array, untrimmed binary service labels with shape (# of reviews, )
        """
        self.food_labels = food_labels
        self.service_labels = service_labels
        self.is_trimmed = False # keep track of whether labels have been trimmed
        self.result = None # initialize resulting array
    
    def trim_unlabeled_reviews(self):
        """
        Trim food and service arrays to only include labeled reviews. Also checks for intermediate NA (unlabelled reviews)
        """
        # Trim reviews
        trim_food_labels = self.food_labels[~np.isnan(self.food_labels)]
        trim_service_labels = self.service_labels[~np.isnan(self.service_labels)]

        # Sanity check for intermediate nan
        assert len(trim_food_labels) == len(trim_service_labels)
        if np.isnan(self.food_labels[:len(trim_food_labels)]).sum() != 0:
            print('Check the following indices')
            print(np.argwhere(np.isnan(self.food_labels[:len(trim_food_labels)])).flatten())
            raise Exception(f'Intermediate nan found')
        if np.isnan(self.service_labels[:len(trim_service_labels)]).sum() != 0:
            print('Check the following indices')
            print(np.argwhere(np.isnan(self.food_labels[:len(trim_food_labels)])).flatten())
            raise Exception(f'Intermediate nan found')

        # Update attributes
        self.is_trimmed = True
        self.food_labels = trim_food_labels
        self.service_labels = trim_service_labels
        return
    
    def get_labels(self):
        """
        Given trimmed binary food and service labels, create one-hot vectors of joint distribution. 
        Input labels must be trimmed first.
        """
        # make sure that labels are trimmed first
        if self.is_trimmed == False:
            raise Exception('Labels need to be trimmed first')
        
        # convert input binary labels to boolean and reshape to column vectors
        food_labels, service_labels = self.food_labels.astype(bool).reshape(-1, 1), self.service_labels.astype(bool).reshape(-1, 1)
        # concat
        binary_labels = np.hstack([food_labels, service_labels]) # (# of reviews, 2)

        # get number of reviews
        num_reviews = len(food_labels)
        # initiate output array with 0
        result = np.zeros(shape=(num_reviews, 4))

        # reviews that are both food and service
        result[:, 2] = binary_labels.all(axis=1)
        # reviews that are neither food or service
        result[:, 3] = ~binary_labels.any(axis=1)

        # keep track of reviews that are either both or neither
        both_or_neither = result[:, 2:].any(axis=1)
        # add only food reviews (not both or neither)
        result[np.ix_(~both_or_neither, [0])] = food_labels[~both_or_neither]
        # add only service reviews (not both or neither)
        result[np.ix_(~both_or_neither, [1])] = service_labels[~both_or_neither]

        # sanity check final result dimensions (make sure vectors are truly one-hot)
        assert result.sum() == len(food_labels) == len(service_labels)
        # update class attribute
        self.result = result
        return
    
    def trim_and_fetch_labels(self):
        """
        Run the trim_unlabeled_reviews() method and get_labels() method. 
        Returns one-hot labels as PyTorch tensor
        """
        self.trim_unlabeled_reviews()
        self.get_labels()
        return torch.tensor(self.result)
