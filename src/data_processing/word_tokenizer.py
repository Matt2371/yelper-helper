#### IMPORT PACKAGES ####
import re

#### FUNCTIONS TO TOKENIZE WORDS ####
def basic_tokenizer(input):
    """Tokenize input string into list of words. Ignores punctuation (,.!?)"""
    # lowercase input string
    input = input.lower()
    # tokenize
    tokens = re.split("\\s+", input)
    tokens = [re.sub("[.?,!]", "", word) for word in tokens]

    return tokens