import re
from difflib import SequenceMatcher
from typing import Callable, Optional, Union

import jellyfish as jf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz


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


def longest_common_subsequence_percent(s1: str, s2: str) -> float:
    """Returns the percentage similarity (0-100) based on the longest common contiguous subsequence.
    Best use case: Comparing the degree of shared contiguous sequences relative to the longer string.
    """
    match_length = longest_common_subsequence(s1, s2)
    longest_length = max(len(s1), len(s2))
    if longest_length == 0:
        return 100.0
    return (match_length / longest_length) * 100


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


def normalize_text(
    text,
    allowed: str = "alnum",  # Options: 'alpha' or 'alnum'
    to_lower: bool = True,
    collapse_spaces: bool = True,
    trim_whitespace: bool = True,
) -> str:
    """
    Cleans text by applying several optional transformations.

    Parameters:
        text: The input value to normalize.
        to_lower (bool): If True, converts the text to lowercase.
        allowed (str): Determines character filtering:
            - 'alpha': keeps only alphabetic characters (a-z, A-Z) and spaces.
            - 'alnum': keeps only alphanumeric characters (a-z, A-Z, 0-9) and spaces.
            Any other value will leave the characters unchanged.
        collapse_spaces (bool): If True, replaces multiple spaces with a single space.
        trim_whitespace (bool): If True, removes leading and trailing whitespace.

    Returns:
        A normalized string. If the input is missing (NaN or None), returns an empty string.
    """
    if not text or pd.isna(text):
        return ""

    # Convert non-string inputs to string
    text = str(text)

    # Apply allowed-character filtering while preserving spaces
    if allowed in ["alnum", "alpha"]:
        if allowed == "alnum":
            # Keep only alphanumeric characters and spaces.
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        elif allowed == "alpha":
            # Keep only alphabetic characters and spaces.
            text = re.sub(r"[^a-zA-Z\s]", " ", text)

    if to_lower:
        text = text.lower()

    if collapse_spaces:
        text = re.sub(r"\s+", " ", text)

    if trim_whitespace:
        text = text.strip()

    return text


def compare_dataframe_columns(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    normalize: Union[bool, dict, Callable[[str], str]] = False,
) -> pd.DataFrame:
    """
    Compares two DataFrame columns using multiple fuzzy methods.

    Parameters:
        df: The input DataFrame.
        col1: The first column name.
        col2: The second column name.
        normalize:
            - If False, no normalization is applied.
            - If True, uses the default normalization (normalize_text with default parameters).
            - If a dict, passes it as keyword arguments to normalize_text.
            - If a callable, uses the provided function for normalization.
    """
    if normalize:
        norm_col1 = col1 + "_normalized"
        norm_col2 = col2 + "_normalized"

        # Determine the normalization function to use
        if callable(normalize):
            norm_func = normalize
        elif isinstance(normalize, dict):
            norm_func = lambda x: normalize_text(x, **normalize)
        else:
            norm_func = normalize_text

        df[norm_col1] = df[col1].apply(norm_func)
        df[norm_col2] = df[col2].apply(norm_func)
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
    df["longest_common_subsequence_percent"] = df.apply(
        lambda x: longest_common_subsequence_percent(x[norm_col1], x[norm_col2]), axis=1
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
        "col1": [
            " apple pie",
            "BANANA smoothie",
            "chocolate cake",
            "Vanilla ice cream ",
        ],
        "col2": [
            "apple tart",
            "banana shake",
            "chocolate gateau",
            "vanilla frozen yogurt",
        ],
    }
    df = pd.DataFrame(data)

    # 1) Original Data (no normalization)
    print("=== Original Data (No Normalization) ===")
    df_no_norm = compare_dataframe_columns(df.copy(), "col1", "col2", normalize=False)
    print(df_no_norm, "\n")

    # 2) Dictionary-based normalization (using default normalize_text with custom parameters)
    print("=== Dictionary-based Normalization ===")
    norm_options = {
        "allowed": "alpha",
        "to_lower": True,
        "collapse_spaces": True,
        "trim_whitespace": True,
    }
    df_dict_norm = compare_dataframe_columns(
        df.copy(), "col1", "col2", normalize=norm_options
    )
    print(df_dict_norm, "\n")


if __name__ == "__main__":
    main()
