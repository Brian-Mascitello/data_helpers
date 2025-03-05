import unittest

import pandas as pd

from data_helpers.word_frequency import count_words, remove_words_from_column, tokenize


class TestTextProcessing(unittest.TestCase):

    def test_tokenize(self):
        text = "Hello, World! This is a test."
        expected = ["hello", "world", "this", "is", "a", "test"]
        self.assertEqual(tokenize(text), expected)

    def test_count_words(self):
        texts = ["apple banana apple", "banana fruit apple"]
        # apple: 3, banana: 2, fruit: 1
        expected = [("apple", 3), ("banana", 2), ("fruit", 1)]
        self.assertEqual(count_words(texts), expected)

    def test_remove_words_from_column(self):
        data = {"Text": ["Hello world", "Testing world", "hello Universe"]}
        df = pd.DataFrame(data)
        words_to_remove = ["world"]
        # Remove "world" regardless of case
        df_clean = remove_words_from_column(
            df, "Text", words_to_remove, case_sensitive=False
        )
        expected_texts = ["Hello", "Testing", "hello Universe"]
        self.assertListEqual(df_clean["Text"].tolist(), expected_texts)

    def test_remove_words_no_changes(self):
        data = {"Text": ["Hello world", "Testing world"]}
        df = pd.DataFrame(data)
        # If words_to_remove is empty, the returned df should be equal (but a copy)
        df_clean = remove_words_from_column(df, "Text", [])
        self.assertListEqual(df["Text"].tolist(), df_clean["Text"].tolist())
        # They should not be the same object
        self.assertIsNot(df, df_clean)


if __name__ == "__main__":
    unittest.main()
