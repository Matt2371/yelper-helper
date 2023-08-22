import unittest

from src.data_processing.word_tokenizer import basic_tokenizer
class test_basic_tokenizer(unittest.TestCase):
    """Test basic word tokenizer (ignores punctuation)"""
    def test_punctuation(self):
        input = "I like this?? Wow."
        expected = ["i", "like", "this", "wow"]
        self.assertEqual(basic_tokenizer(input), expected)
    
    def test_apost(self):
        input = "I don't like"
        expected = ["i", "don't", "like"]
        self.assertEqual(basic_tokenizer(input), expected)
    

if __name__ == "__main__":
    unittest.main()