from typing import Optional
from thefuzz import fuzz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import jellyfish as jf
import numpy as np


def contains_string(s1: str, s2: str) -> bool:
    """Checks if one string is contained in the other.
    This function verifies whether one string is a complete substring of the other.
    Best use case: Exact substring matching."""
    return s1 in s2 or s2 in s1


def cosine_similarity_score(s1: str, s2: str) -> float:
    """Returns the cosine similarity between two text strings.
    This function uses TF-IDF vectorization to compare textual similarity.
    Best use case: Comparing long text passages, documents, or descriptions."""
    vectorizer = TfidfVectorizer().fit_transform([s1, s2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]


def hamming_distance(s1: str, s2: str, pad: bool = False) -> Optional[int]:
    """Returns the Hamming distance between two strings.
    When pad is False, the function returns None if the strings have unequal lengths.
    When pad is True, the shorter string is padded with a null character ('\0')
    to match the length of the longer string.
    Best use case: Comparing fixed-length strings such as DNA sequences or binary strings.
    """
    if pad:
        max_len = max(len(s1), len(s2))
        s1_padded = s1.ljust(max_len, "\0")
        s2_padded = s2.ljust(max_len, "\0")
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1_padded, s2_padded))
    else:
        if len(s1) != len(s2):
            return None  # Hamming distance is only defined for equal-length strings
        return jf.hamming_distance(s1, s2)


def jaccard_similarity(s1: str, s2: str) -> float:
    """Returns Jaccard similarity score between two strings.
    This function calculates the Jaccard index, which measures similarity by comparing the intersection and union of words in two strings.
    Best use case: Word-based similarity, useful for documents or names."""
    set1, set2 = set(s1.split()), set(s2.split())
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0


def jaro_similarity(s1: str, s2: str) -> float:
    """Returns the Jaro similarity score between two strings.
    Best use case: Comparing short strings, such as names."""
    return jf.jaro_similarity(s1, s2)


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Returns the Jaro-Winkler similarity score between two strings.
    Best use case: Comparing short strings with a focus on common prefixes, useful for name matching.
    """
    return jf.jaro_winkler_similarity(s1, s2)


def levenshtein_ratio(s1: str, s2: str) -> int:
    """Returns the Levenshtein similarity ratio (0-100).
    This function computes the Levenshtein distance, which quantifies how different two strings are based on character insertions, deletions, and substitutions.
    Best use case: General similarity, sensitive to small changes."""
    return fuzz.ratio(s1, s2)


def longest_common_subsequence(s1: str, s2: str) -> int:
    """Returns the length of the longest common subsequence.
    This function finds the longest sequence of characters appearing in both strings in order.
    Best use case: Identifying shared character sequences, useful for name matching."""
    return SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2)).size


def metaphone_similarity(s1: str, s2: str) -> bool:
    """Checks if two words have the same Metaphone encoding.
    Best use case: Phonetic matching for English words and names."""
    return jf.metaphone(s1) == jf.metaphone(s2)


def nysiis_similarity(s1: str, s2: str) -> bool:
    """Checks if two words have the same NYSIIS encoding.
    Best use case: Phonetic matching for names, better suited for non-English names."""
    return jf.nysiis(s1) == jf.nysiis(s2)


def partial_ratio(s1: str, s2: str) -> int:
    """Returns the Partial Ratio similarity (0-100).
    This function finds the best matching substring in a longer string and calculates the similarity score based on that subset.
    Best use case: One string is a substring of another."""
    return fuzz.partial_ratio(s1, s2)


def soundex_similarity(s1: str, s2: str) -> bool:
    """Checks if two words have the same Soundex encoding.
    Best use case: Name matching with spelling variations (e.g., Jon vs. John)."""
    return jf.soundex(s1) == jf.soundex(s2)


def token_set_ratio(s1: str, s2: str) -> int:
    """Returns the Token Set Ratio similarity (0-100).
    This function compares word sets, ignoring duplicate words and focusing on essential differences.
    Best use case: Handles duplicate words, useful for long strings."""
    return fuzz.token_set_ratio(s1, s2)


def token_sort_ratio(s1: str, s2: str) -> int:
    """Returns the Token Sort Ratio similarity (0-100).
    This function sorts words alphabetically before comparison, making it useful when word order varies.
    Best use case: When word order differs."""
    return fuzz.token_sort_ratio(s1, s2)


def normalize_text(text) -> str:
    """Cleans text by stripping whitespace and converting to lowercase.
    
    If the input is missing (NaN) or not a string, it returns an empty string.
    """
    if pd.isna(text) or text is None:
        return ""
    # Convert non-string values to string before processing.
    text = str(text)
    return text.strip().lower()


def compare_dataframe_columns(
    df: pd.DataFrame, col1: str, col2: str, normalize: bool = False
) -> pd.DataFrame:
    """
    Compares two DataFrame columns using multiple fuzzy methods.
    
    If normalize is True, temporary columns (col1_normalized and col2_normalized)
    are created to hold cleaned text (lowercase and stripped) before performing comparisons.
    """
    if normalize:
        norm_col1 = col1 + "_normalized"
        norm_col2 = col2 + "_normalized"
        df[norm_col1] = df[col1].apply(normalize_text)
        df[norm_col2] = df[col2].apply(normalize_text)
    else:
        norm_col1 = col1
        norm_col2 = col2

    df["contains_string"] = df.apply(
        lambda x: contains_string(x[norm_col1], x[norm_col2]), axis=1
    )
    df["cosine_similarity"] = df.apply(
        lambda x: cosine_similarity_score(x[norm_col1], x[norm_col2]), axis=1
    )
    df["hamming_distance"] = df.apply(
        lambda x: hamming_distance(x[norm_col1], x[norm_col2], pad=True), axis=1
    )
    df["jaccard_similarity"] = df.apply(
        lambda x: jaccard_similarity(x[norm_col1], x[norm_col2]), axis=1
    )
    df["jaro_similarity"] = df.apply(
        lambda x: jaro_similarity(x[norm_col1], x[norm_col2]), axis=1
    )
    df["jaro_winkler_similarity"] = df.apply(
        lambda x: jaro_winkler_similarity(x[norm_col1], x[norm_col2]), axis=1
    )
    df["levenshtein_ratio"] = df.apply(
        lambda x: levenshtein_ratio(x[norm_col1], x[norm_col2]), axis=1
    )
    df["longest_common_subsequence"] = df.apply(
        lambda x: longest_common_subsequence(x[norm_col1], x[norm_col2]), axis=1
    )
    df["metaphone_match"] = df.apply(
        lambda x: metaphone_similarity(x[norm_col1], x[norm_col2]), axis=1
    )
    df["nysiis_match"] = df.apply(
        lambda x: nysiis_similarity(x[norm_col1], x[norm_col2]), axis=1
    )
    df["partial_ratio"] = df.apply(
        lambda x: partial_ratio(x[norm_col1], x[norm_col2]), axis=1
    )
    df["soundex_match"] = df.apply(
        lambda x: soundex_similarity(x[norm_col1], x[norm_col2]), axis=1
    )
    df["token_set_ratio"] = df.apply(
        lambda x: token_set_ratio(x[norm_col1], x[norm_col2]), axis=1
    )
    df["token_sort_ratio"] = df.apply(
        lambda x: token_sort_ratio(x[norm_col1], x[norm_col2]), axis=1
    )

    return df


def main() -> None:
    """Example usage of the fuzzy comparison functions."""
    data = {
        "col1": [" apple pie", "BANANA smoothie", "chocolate cake", "Vanilla ice cream "],
        "col2": [
            "apple tart",
            "banana shake",
            "chocolate gateau",
            "vanilla frozen yogurt",
        ],
    }
    df = pd.DataFrame(data)
    
    # Toggle normalization on or off as desired.
    df = compare_dataframe_columns(df, "col1", "col2", normalize=True)
    print(df)


if __name__ == "__main__":
    main()
