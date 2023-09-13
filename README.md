# Yelper-Helper
This project endeavors to enhance the customer's experience when browsing reviews for restaurants by developing a deep-learning language model capable of accurately categorizing reviews as pertaining to food quality, service quality, both aspects, or neither.

## Data Source
The data used for training and finetuning of our language models is sourced from [Yelp Open Dataset](https://www.yelp.com/dataset).
Note: The data was filtered to only include businesses that are food related.

## Methedology
#### LSTM Model Training:
1. Manually categorize Yelp reviews into the 'food' and 'service' categories, based on whether they pertain to food-related aspects, service-related aspects, both, or neither.
2. Process the labelled reviews through word2vec tokenizer.
3. Conduct train/validation/test on a subsample of 1000 labelled reviews (60%/20%/20%).
4. Train LSTM models using the training data, while validating and hyperparameter tuning on the validation set. The test set is used to evaluate model performance. 

#### BERT Model Fine-Tuning
1. Manually categorize Yelp reviews into the 'food' and 'service' categories, based on whether they pertain to food-related aspects, service-related aspects, both, or neither.
2. Process the labelled reviews with 'bert-base-uncased' tokenizer.
3. Conduct train/test on a subsample of 1000 labelled reviews (80%/20%).
4. Finetune BERT base model with and without class weights and evaluate model performance.

## Source Code
The source code is organized into two main modules: data_processing and models.

### data_processing
1. The *process_label* submodule is responsible for transforming the manually labeled reviews' 'food' and 'service' columns into one-hot vectors that represent the joint distribution of the labels. These one-hot vectors encode information about whether a review mentions 'food', 'service', 'both food and service', or 'neither'. The submodule returns one-hot labels as PyTorch tensor
2. The *process_review* submodule is includes the text data processing pipeline for the LSTM model and returns word2vec embeddings for each token in each review, as PyTorch tensor with shape (# reviews, *).
3. The *train_val_test* submodule house a custom function to easily split the data into train/validate/test.
4. The *word_tokenizer* submodule contain all the necessary functions to convert review text data into tokens and also convert those word tokens into word2vec embeddings.

### models
1. The *model_evaluate* submodule contains functions that summarizes performance metrics (accuracy, f1, precision, recall)
    for each class and the overall score (which returns a weighted average of all the scores 
    based on the number of true instances).
2. The *model_train* submodule defines functions to faciliate training LSTM and finetuning BERT. Both have optional early stopper hyperparameter (default patience = 1).

    - The complete LSTM training loop can be accessed using the 'training_loop' function. You will need to provide the following arguments: the PyTorch model, criterion (e.g., cross-entropy loss), PyTorch optimizer, desired patience value for the early stopper, PyTorch dataloader objects for both training and validation, and the max number of epochs to run. The function returns avg training and validation losses for each epoch in two separate lists.

    - The complete BERT finetuning loop can be accessed using the 'bert_training_loop' function. You will need to provide the following arguments: custom PyTorch BERT model, criterion (e.g., cross-entropy loss), PyTorch optimizer, desired patience value for the early stopper, PyTorch dataloader objects for both training and validation, the max number of epochs to run, and a boolean flag to enable early stopping. The function returns avg training and validation losses for each epoch in two separate lists.

3. The *model_zoo* contains class definitions for PyTorch models used in the project.