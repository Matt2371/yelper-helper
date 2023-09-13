# yelper-helper
This project endeavors to enhance the precision of customer reviews analysis for restaurants by developing a deep-learning language model capable of accurately categorizing reviews as pertaining to food quality, service quality, both aspects, or neither.

## Data
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
The source code is organized into two main modules: data and models.

