import re
from collections import Counter
from typing import List, Tuple, Union

import pandas as pd


def tokenize(text: str) -> List[str]:
    """
    Extract words from the input text and convert them to lowercase.

    Args:
        text (str): The text to tokenize.

    Returns:
        List[str]: A list of words extracted from the text.
    """
    # Use regex to find word boundaries and convert to lowercase
    words = re.findall(r"\b\w+\b", text.lower())
    return words


def count_words(text_input: Union[List[str], pd.Series]) -> List[Tuple[str, int]]:
    """
    Count word occurrences in a list or pandas Series of text strings.

    This function accepts both a list of strings and a pandas Series, converting a Series to a list if needed.
    It tokenizes each text string, counts the occurrences of each word, and returns the counts sorted by descending
    frequency. When frequencies are equal, words are sorted alphabetically.

    Args:
        text_input (Union[List[str], pd.Series]): A list or pandas Series of text strings.

    Returns:
        List[Tuple[str, int]]: A list of tuples where each tuple contains a word and its frequency.

    Note:
        This function assumes each element in text_input is a string.
    """
    # Convert pandas Series to list if necessary
    if isinstance(text_input, pd.Series):
        text_list = text_input.tolist()
    else:
        text_list = text_input

    # Tokenize all texts and count the occurrences of each word
    all_words = [word for text in text_list for word in tokenize(text)]
    word_counts = Counter(all_words)
    # Sort primarily by descending frequency, then alphabetically
    return sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))


def print_word_counts(word_counts: List[Tuple[str, int]]) -> None:
    """
    Print word frequency counts in a formatted table.

    Args:
        word_counts (List[Tuple[str, int]]): A list of tuples where each tuple contains a word and its frequency.
    """
    print("\nWord Frequency Count:")
    print("-" * 30)
    for word, count in word_counts:
        print(f"{word:<15} {count}")
    print("-" * 30)


def remove_words_from_column(
    df: pd.DataFrame,
    column: str,
    words_to_remove: List[str],
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Remove specified words from a text column in a DataFrame using regex.

    This function first fills missing values in the target column with empty strings, then removes the words
    specified in words_to_remove. The removal is case-insensitive by default unless case_sensitive is set to True.
    Extra spaces resulting from word removal are cleaned up afterwards.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column containing text.
        words_to_remove (List[str]): A list of words to remove.
        case_sensitive (bool): If True, performs case-sensitive removal. Defaults to False.

    Returns:
        pd.DataFrame: A new DataFrame with the specified words removed from the column.

    Raises:
        ValueError: If the specified column is not found in the DataFrame.
    """
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {df.columns.tolist()}"
        )

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Handle missing values by filling NaNs with empty strings
    df[column] = df[column].fillna("")

    if not words_to_remove:
        return df  # No words to remove; return the DataFrame as is.

    # Construct a regex pattern to match any of the words to remove
    words_pattern = r"\b(?:" + "|".join(map(re.escape, words_to_remove)) + r")\b"

    # Replace the words using the appropriate case sensitivity
    if case_sensitive:
        df[column] = df[column].str.replace(words_pattern, "", regex=True)
    else:
        df[column] = df[column].str.replace(
            words_pattern, "", flags=re.IGNORECASE, regex=True
        )

        df[column] = df[column].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def count_and_save_word_counts(
    text_series: pd.Series, csv_filename: str, label: str
) -> pd.DataFrame:
    """
    Count words in a text series, save the frequency counts to a CSV, print the average frequency,
    and return the DataFrame of word counts.
    """
    word_counts = count_words(text_series)
    df_word_counts = pd.DataFrame(word_counts, columns=["Word", "Frequency"])
    df_word_counts.to_csv(csv_filename, index=False)
    avg_freq = df_word_counts["Frequency"].mean()
    print(f"\nAverage word frequency {label}: {avg_freq:.2f}")
    return df_word_counts


def process_text_df(
    df: pd.DataFrame,
    original_col: str = "Original_Text",
    cleaned_col: str = "Cleaned_Text",
    words_to_remove: List[str] = None,
    before_csv: str = "before_word_freqs.csv",
    after_csv: str = "after_word_freqs.csv",
    cleaned_csv: str = "cleaned_text.csv",
) -> pd.DataFrame:
    """
    Process the text DataFrame by counting word frequencies before and after cleaning,
    removing specified words, and saving the intermediate CSV files.

    Returns the DataFrame with an added column for the cleaned text.
    """
    if words_to_remove is None:
        words_to_remove = []

    # Count words BEFORE cleaning using the original text
    count_and_save_word_counts(df[original_col], before_csv, "BEFORE cleaning")

    # Create a new column for cleaned text and remove specified words
    df[cleaned_col] = df[original_col].copy()
    df = remove_words_from_column(
        df, cleaned_col, words_to_remove, case_sensitive=False
    )

    # Save the DataFrame with both Original and Cleaned Text to CSV
    df.to_csv(cleaned_csv, index=False)

    # Count words AFTER cleaning
    count_and_save_word_counts(df[cleaned_col], after_csv, "AFTER cleaning")

    return df


def main() -> None:
    """
    Demonstrate the usage of process_text_df by:
      - Creating a sample DataFrame.
      - Defining the words to remove.
      - Processing the DataFrame.
      - Printing the final DataFrame.
    """
    data = {
        "Original_Text": [
            "Fuzzy matching is useful for string comparison",
            "String Matching techniques include Levenshtein distance",
            "FUZZY string matching can be applied in NLP",
            "Tokenization helps in text processing and comparison",
            "Distance-based approaches are common in Fuzzy Matching",
        ]
    }
    df = pd.DataFrame(data)
    words_to_remove = ["Fuzzy", "Matching", "comparison"]

    processed_df = process_text_df(
        df,
        original_col="Original_Text",
        cleaned_col="Cleaned_Text",
        words_to_remove=words_to_remove,
        before_csv="before_word_freqs.csv",
        after_csv="after_word_freqs.csv",
        cleaned_csv="cleaned_text.csv",
    )

    print("\nData after processing:")
    print(processed_df)


if __name__ == "__main__":
    main()
