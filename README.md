# Yelper-Helper
This project seeks to gain greater insight from yelp reviews by building a neural text classifier, classifying
yelp reviews as food related, service related, both, or neither. An example use case is to determine if a restaurant
has good food but poor service, etc...

## Data Source
The data used for training and finetuning of our models are sourced from the [Yelp Open Dataset](https://www.yelp.com/dataset). Note that the data was filtered to only include businesses that are food related, elminating most non-restaurant establishments.

## Methedology
#### Data labeling
1000 yelp reviews were manually labelled as food or service related.

#### LSTM Model Training:
The bi-directional LSTM is used as a baseline for our project. We did not have enough labelled data
to train a robust model from scratch, and performance will likely be poor.

1. Obtain pretrained word2vec embeddings from the labelled reviews.
2. Split the data into train/validation/test datasets (60%/20%/20%).
3. Train LSTM models on the training data, monitoring performance and manually hyperparameter tuning using the validation data. The test set is used to evaluate the performance of the final model. 

#### BERT Model Fine-Tuning
Here, we set up a transfer learning environment and fine-tune BERT (base) for our text classification text. HuggingFace Transformers provides both an implementation of the BERT model and its tokenizer.

1. Tokenize labelled reviews using the 'bert-base-uncased' tokenizer.
2. Conduct train/test on a subsample of 1000 labelled reviews (80%/20%).
3. Finetune BERT model using 4 epochs of the training data. Evaluate final performance on the test set.

#### BERT Model (Class Weighted)
We note that there is a class imbalance, particularly towards reviews that are only service related, or neither food or service related. Thus, to try to improve performance on the minority classes, we finetune the BERT model using a class-weighted loss. Otherwise, the methodoloy is the same as fine tuning the BERT model with an unmodified loss.

## Source Code
The source code is organized into two main modules: data_processing and models.

### data_processing
1. The *process_label* submodule transforms the raw labeled reviews into one-hot vectors that represent the joint distribution of the labels (food, service, both, neither), returned as a PyTorch tensor.
2. The *process_review* submodule includes the text data processing pipeline for the LSTM model and returns word2vec embeddings for each token in each review, as PyTorch tensor with shape (# reviews, *).
3. The *train_val_test* submodule houses a function that splits the data into train/validate/test partitions.
4. The *word_tokenizer* submodule contain all the necessary functions to convert review text data into tokens and convert them into word2vec embeddings.

### models
1. The *model_evaluate* submodule contains functions that summarizes performance metrics (accuracy, f1, precision, recall)
    for each class and the overall accuracy score.
2. The *model_train* submodule defines functions to faciliate training the LSTM and finetuning BERT. There is an option to train with an early stopping callback.

    - The complete LSTM training loop can be accessed using the 'training_loop' function. You will need to provide the following arguments: the PyTorch model, criterion (e.g., cross-entropy loss), PyTorch optimizer, desired patience value for the early stopper, PyTorch dataloader objects for both training and validation, and the max number of epochs to run. The function returns avg training and validation losses for each epoch in two separate lists.

    - The complete BERT finetuning loop can be accessed using the 'bert_training_loop' function. You will need to provide the following arguments: custom PyTorch BERT model, criterion (e.g., cross-entropy loss), PyTorch optimizer, desired patience value for the early stopper, PyTorch dataloader objects for both training and validation, the max number of epochs to run, and a boolean flag to enable early stopping. The function returns avg training and validation losses for each epoch in two separate lists.

3. The *model_zoo* contains class definitions for PyTorch models used in the project.
    - LSTMmodel: the last hidden state from n bi-directional LSTM layers with dropout + linear + softmax
        - Takes input_size of dimension 300, consistent with the size of word2vec embedding for each token.
    - BERTClass: pooling output from the pretrained 'bert-base-uncased' model (the pooling output which summarizes the entire input sequence) + dropout + linear + softmax

## Tests
Contains unittests of the experimental process throughout, with a focus on data processing and model shape in the forward pass. To run a unittest, run the command `py -m unittest tests.(name of test script)`

## Archive
The archive directory contains notebooks that have been retired. They may still be useful to show the experimental and data processing setup. 

## Project conclusions
As expected, the LSTM model performed extremely poorly on all metrics due to a lack of training data and resources. The fine-tuned BERT model has high accuracy and exhibits good performance over (only food) and (both food and service) categories. Interestingly, however, the model performs extremely poorly on the (only service) or (neither) categories. We note that (only service) or (neither) categories are sparse in the dataset, compared to the (only food) and (both food and service) labels that take up most of the dataset. To try to combat the class imbalance, we tried finetuning using a class weighted loss function, but performance in the minority class did not improve. There is likely not enough labelled reviews (recall there are only 800 manually labelled reviews in the training set) of the minority classes to capture semantic nuances of these labels, and such reviews are almost always incorrectly classified as one of the other labels. Overall, we conclude that while the model performs well for certain classes, there was not enough labelled data to obtain a robust result. However, given the high performance on (only food) labels, it may be possible to build a strong binary classifier in the future using the same methodology.