import re
from collections import Counter
from typing import List, Tuple

import pandas as pd


def tokenize(text: str) -> List[str]:
    """Extracts words from text and converts them to lowercase."""
    words = re.findall(r"\b\w+\b", text.lower())  # Extract words, ignoring case
    return words


def count_words(text_list: List[str]) -> List[Tuple[str, int]]:
    """Counts word occurrences in a list of text strings."""
    all_words = [word for text in text_list for word in tokenize(text)]
    word_counts = Counter(all_words)
    return sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))


def print_word_counts(word_counts: List[Tuple[str, int]]) -> None:
    """Prints word frequencies in a formatted way."""
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
    """Removes specified words from a text column in a DataFrame using regex.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column containing text.
        words_to_remove (List[str]): List of words to remove.
        case_sensitive (bool): If True, remove words exactly as listed (case-sensitive). Defaults to False.

    Returns:
        pd.DataFrame: Modified DataFrame with words removed.
    """
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {df.columns.tolist()}"
        )

    df = df.copy()

    # Handle missing values by filling NaNs with empty strings
    df[column] = df[column].fillna("")

    if not words_to_remove:
        return df
    else:
        # Construct regex pattern for words to remove
        words_pattern = r"\b(?:" + "|".join(map(re.escape, words_to_remove)) + r")\b"

        # Case-insensitive replacement using `lambda` in `apply()`
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
    """Main function to execute the script."""
    # Sample DataFrame
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

    # Count words BEFORE cleaning (Original Text)
    word_counts_before = count_words(df["Original_Text"].tolist())

    # Save word counts BEFORE cleaning
    df_word_counts_before = pd.DataFrame(
        word_counts_before, columns=["Word", "Frequency"]
    )
    df_word_counts_before.to_csv("before_word_freqs.csv", index=False)

    # Calculate and print average frequency BEFORE cleaning
    avg_freq_before = df_word_counts_before["Frequency"].mean()
    print(f"\nAverage word frequency BEFORE cleaning: {avg_freq_before:.2f}")

    # Define words to remove
    words_to_remove = ["Fuzzy", "Matching", "comparison"]

    # Create Cleaned_Text without modifying Original_Text
    df["Cleaned_Text"] = df["Original_Text"].copy()  # Copy original column
    df["Cleaned_Text"] = remove_words_from_column(
        df, "Cleaned_Text", words_to_remove, case_sensitive=False
    )["Cleaned_Text"]

    # Save the DataFrame with both Original and Cleaned Text
    df.to_csv("cleaned_text.csv", index=False)

    # Count words AFTER cleaning (Cleaned Text)
    word_counts_after = count_words(df["Cleaned_Text"].tolist())

    # Save word counts AFTER cleaning
    df_word_counts_after = pd.DataFrame(
        word_counts_after, columns=["Word", "Frequency"]
    )
    df_word_counts_after.to_csv("after_word_freqs.csv", index=False)

    # Calculate and print average frequency AFTER cleaning
    avg_freq_after = df_word_counts_after["Frequency"].mean()
    print(f"\nAverage word frequency AFTER cleaning: {avg_freq_after:.2f}")

    # Print cleaned DataFrame
    print("\nData after removing specified words:")
    print(df)


if __name__ == "__main__":
    main()
