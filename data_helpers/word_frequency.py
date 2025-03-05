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

    # Clean up extra spaces left from removals
    df[column] = df[column].str.replace(r"\s+", " ", regex=True).str.strip()

    return df


def main() -> None:
    """
    Execute the main functionality:
    - Create a sample DataFrame.
    - Count word frequencies before and after removing specified words.
    - Save intermediate results to CSV files.
    - Print summary statistics and the final DataFrame.
    """
    # Sample DataFrame with original text
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

    # Count words BEFORE cleaning (using Original_Text)
    word_counts_before = count_words(df["Original_Text"])

    # Save word counts before cleaning to CSV
    df_word_counts_before = pd.DataFrame(
        word_counts_before, columns=["Word", "Frequency"]
    )
    df_word_counts_before.to_csv("before_word_freqs.csv", index=False)

    # Calculate and print average frequency before cleaning
    avg_freq_before = df_word_counts_before["Frequency"].mean()
    print(f"\nAverage word frequency BEFORE cleaning: {avg_freq_before:.2f}")

    # Define the words to remove from the text
    words_to_remove = ["Fuzzy", "Matching", "comparison"]

    # Create a new column for cleaned text without modifying Original_Text
    df["Cleaned_Text"] = df["Original_Text"].copy()
    df["Cleaned_Text"] = remove_words_from_column(
        df, "Cleaned_Text", words_to_remove, case_sensitive=False
    )["Cleaned_Text"]

    # Save the DataFrame with both Original and Cleaned Text to CSV
    df.to_csv("cleaned_text.csv", index=False)

    # Count words AFTER cleaning (using Cleaned_Text)
    word_counts_after = count_words(df["Cleaned_Text"])

    # Save word counts after cleaning to CSV
    df_word_counts_after = pd.DataFrame(
        word_counts_after, columns=["Word", "Frequency"]
    )
    df_word_counts_after.to_csv("after_word_freqs.csv", index=False)

    # Calculate and print average frequency after cleaning
    avg_freq_after = df_word_counts_after["Frequency"].mean()
    print(f"\nAverage word frequency AFTER cleaning: {avg_freq_after:.2f}")

    # Print the final DataFrame showing the cleaned text
    print("\nData after removing specified words:")
    print(df)


if __name__ == "__main__":
    main()
