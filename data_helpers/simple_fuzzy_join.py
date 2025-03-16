import re
from typing import Callable, Optional, Tuple, Union

import pandas as pd
from thefuzz import fuzz
from tqdm import tqdm


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
    # Python should check if NA first, order matters.
    if pd.isna(text) or not text:
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


def find_best_match(
    text1: str,
    df2: pd.DataFrame,
    col2: str,
    score_func: Callable[[str, str], int],
    norm_func: Optional[Callable[[str], str]] = None,
    first_letter_blocking: bool = False,
    first_word_blocking: bool = False,
) -> Tuple[Optional[pd.Series], int]:
    """
    Finds the best match for text1 in the provided DataFrame df2.

    Parameters:
        text1 (str): The text from df1 (assumed to be already normalized if needed).
        df2 (pd.DataFrame): The second DataFrame with candidate texts.
        col2 (str): Column in df2 containing strings to match.
        score_func (Callable[[str, str], int]): Function to compute a matching score.
        norm_func (Optional[Callable[[str], str]]): Normalization function to apply to df2 text.
        first_letter_blocking (bool): If True, only considers candidates starting with the same letter as text1.
        first_word_blocking (bool): If True, only considers candidates whose first word matches that of text1.

    Returns:
        Tuple[Optional[pd.Series], int]: The best matching row from df2 and its score.
    """
    # Ensure text1 is a string
    text1 = "" if pd.isna(text1) else str(text1)
    best_score = -1
    best_match_row = None

    for _, candidate in df2.iterrows():
        # Use cached normalized value if present.
        if f"__norm_{col2}" in candidate:
            text2 = candidate[f"__norm_{col2}"]
        else:
            text2_orig = candidate[col2]
            text2 = norm_func(text2_orig) if norm_func else text2_orig
        text2 = "" if pd.isna(text2) else str(text2)

        if first_letter_blocking:
            if f"__first_letter_{col2}" in candidate:
                letter2 = candidate[f"__first_letter_{col2}"]
            else:
                letter2 = str(text2)[0] if text2 and len(str(text2)) > 0 else ""
            if not text1 or not text2 or text1[0] != letter2:
                continue

        if first_word_blocking:
            if f"__first_word_{col2}" in candidate:
                word2 = candidate[f"__first_word_{col2}"]
            else:
                word2 = str(text2).split()[0] if str(text2).split() else ""
            word1 = str(text1).split()[0] if str(text1).split() else ""
            if word1 != word2:
                continue

        score = score_func(text1, text2)
        if score > best_score:
            best_score = score
            best_match_row = candidate

    return best_match_row, best_score


def join_best_match(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    col1: str,
    col2: str,
    score_func: Callable[[str, str], int] = None,
    normalize: Union[bool, dict, Callable[[str], str]] = False,
    first_letter_blocking: bool = False,
    first_word_blocking: bool = False,
    match_suffix: str = "_match",
    remove_cache: bool = True,
) -> pd.DataFrame:
    """
    Joins two DataFrames based on the best fuzzy match of the specified string columns,
    with optional normalization, first letter blocking, and first word blocking.

    This function returns a new DataFrame that contains:
      - All original columns from df1,
      - A 'best_match_score' column,
      - All columns from the best matching row in df2, with their names appended by the specified suffix.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        col1 (str): The column name in df1 containing strings to match.
        col2 (str): The column name in df2 containing strings to match.
        score_func (Callable[[str, str], int], optional): A scoring function that takes two strings
            and returns an integer score. Defaults to using fuzz.ratio.
        normalize (Union[bool, dict, Callable[[str], str]]): If provided, normalizes text before matching.
            - If False, no normalization is applied.
            - If True, uses the default normalize_text with its default parameters.
            - If dict, passes it as keyword arguments to normalize_text.
            - If callable, uses the provided function.
        first_letter_blocking (bool): If True, only candidate rows from df2 with the same first letter
            (after normalization if applied) as df1 are considered.
        first_word_blocking (bool): If True, only candidate rows from df2 with the same first word
            (after normalization if applied) as df1 are considered.
        match_suffix (str): Suffix to append to the df2 column names in the output (default: "_match").
        remove_cache (bool): If True, remove caching helper columns from the final output.

    Returns:
        pd.DataFrame: A DataFrame with all df1 rows, the best match from df2 (columns renamed with the suffix),
                    and the best_match_score.
    """
    # Use fuzz.ratio as default scoring function if not provided.
    if score_func is None:
        score_func = lambda a, b: fuzz.ratio(str(a), str(b))

    # Set up normalization function if normalization is enabled.
    norm_func = None
    if normalize:
        if callable(normalize):
            norm_func = normalize
        elif isinstance(normalize, dict):
            norm_func = lambda x: normalize_text(x, **normalize)
        else:
            norm_func = normalize_text

    # Enable caching if any normalization or blocking is enabled.
    cache_enabled = bool(normalize or first_letter_blocking or first_word_blocking)
    if cache_enabled:
        # Create normalized columns.
        df1[f"__norm_{col1}"] = df1[col1].apply(norm_func) if norm_func else df1[col1]
        df2[f"__norm_{col2}"] = df2[col2].apply(norm_func) if norm_func else df2[col2]
        # Create first letter columns if needed.
        if first_letter_blocking:
            df1[f"__first_letter_{col1}"] = df1[f"__norm_{col1}"].apply(
                lambda x: str(x)[0] if pd.notna(x) and len(str(x)) > 0 else ""
            )
            df2[f"__first_letter_{col2}"] = df2[f"__norm_{col2}"].apply(
                lambda x: str(x)[0] if pd.notna(x) and len(str(x)) > 0 else ""
            )
        # Create first word columns if needed.
        if first_word_blocking:
            df1[f"__first_word_{col1}"] = df1[f"__norm_{col1}"].apply(
                lambda x: (
                    str(x).split()[0] if pd.notna(x) and len(str(x).split()) > 0 else ""
                )
            )
            df2[f"__first_word_{col2}"] = df2[f"__norm_{col2}"].apply(
                lambda x: (
                    str(x).split()[0] if pd.notna(x) and len(str(x).split()) > 0 else ""
                )
            )

    result_rows = []
    # Iterate over each row in df1 with a progress bar.
    for _, row in tqdm(df1.iterrows(), total=len(df1), desc="Matching rows"):
        # Use the cached normalized value if available.
        text1 = (
            row[f"__norm_{col1}"]
            if cache_enabled and f"__norm_{col1}" in row
            else row[col1]
        )

        best_match_row, best_score = find_best_match(
            text1,
            df2,
            col2,
            score_func,
            norm_func,
            first_letter_blocking,
            first_word_blocking,
        )

        result = row.to_dict()
        result["best_match_score"] = best_score

        # Add matched df2 columns, with column names appended by match_suffix.
        if best_match_row is not None:
            for key, value in best_match_row.to_dict().items():
                result[f"{key}{match_suffix}"] = value
        else:
            for key in df2.columns:
                result[f"{key}{match_suffix}"] = None

        result_rows.append(result)

    result_df = pd.DataFrame(result_rows)

    # Remove caching columns if requested.
    if remove_cache:
        cache_cols = [col for col in result_df.columns if col.startswith("__")]
        result_df.drop(columns=cache_cols, inplace=True)

    return result_df


def main() -> None:
    """Simple test case for join_best_match."""
    # Sample data for df1 and df2.
    data1 = {
        "col1": ["apple pie", "banana smoothie", "chocolate cake", "vanilla ice cream"]
    }
    data2 = {
        "col2": [
            "apple tart",
            "banana shake",
            "chocolate gateau",
            "vanilla frozen yogurt",
        ],
        "other_info": ["delicious", "refreshing", "decadent", "creamy"],
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    result = join_best_match(
        df1,
        df2,
        col1="col1",
        col2="col2",
        first_letter_blocking=True,
        first_word_blocking=True,
        match_suffix="_match",
        remove_cache=True,
    )

    print("Joined DataFrame:")
    print(result)
    result.to_csv("result.csv", index=False)


if __name__ == "__main__":
    main()
