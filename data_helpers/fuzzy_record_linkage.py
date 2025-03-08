import itertools
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from thefuzz import fuzz
from tqdm import tqdm

# Constant used to indicate a "no match" in fuzzy matching.
NO_MATCH_PLACEHOLDER = "NO_MATCH_PLACEHOLDER"


def validate_dataframe_columns(
    df: pd.DataFrame, expected_columns: List[str], df_name: str
) -> None:
    """
    Validates that the DataFrame contains all expected columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to validate.
        expected_columns (List[str]): List of column names expected in the DataFrame.
        df_name (str): Name of the DataFrame (for error messaging).

    Raises:
        ValueError: If any expected column is missing from the DataFrame.
    """
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"DataFrame '{df_name}' is missing required columns: {missing_columns}"
        )


def generate_flexible_combos(flex_keys: List[str]) -> List[Tuple[str, ...]]:
    """
    Generates all possible subsets (combinations) of flexible columns.

    The function starts with the largest combination and works its way down to the empty set.
    This ordering allows the algorithm to try the most strict match first before relaxing constraints.

    Parameters:
        flex_keys (List[str]): List of flexible column names.

    Returns:
        List[Tuple[str, ...]]: List of all combinations of the flexible columns.
    """
    combos: List[Tuple[str, ...]] = []
    for r in range(len(flex_keys), -1, -1):
        for combo in itertools.combinations(flex_keys, r):
            combos.append(combo)
    return combos


def passes_blocking_optimized(
    row: pd.Series,
    candidate_row: pd.Series,
    required_columns: Dict[str, str],
    fuzzy_case_sensitive: bool,
) -> bool:
    """
    Optimized blocking mechanism using set intersections with robust edge case handling.

    For each required column, this function:
      - Checks that both the row (df1) and candidate_row (df2) have a non-empty, non-NA value.
      - Extracts tokens (words) from both sides, applying case normalization if required.
      - Uses a set intersection to quickly determine if any token in df1 exists in df2.
      - Returns False immediately if a valid token is missing on either side, preventing a match.

    Parameters:
        row (pd.Series): A row from df1.
        candidate_row (pd.Series): A row from df2.
        required_columns (Dict[str, str]): Mapping of required column names in df1 to df2.
        fuzzy_case_sensitive (bool): Whether the matching should be case sensitive.

    Returns:
        bool: True if the blocking check passes; otherwise, False.
    """
    for req_col in required_columns:
        # Retrieve values from both dataframes for the required column.
        row_value = row[req_col]
        candidate_value = candidate_row[required_columns[req_col]]

        # Fail the blocking check if either value is missing (NA or empty after stripping).
        if pd.isna(row_value) or pd.isna(candidate_value):
            return False
        row_str = str(row_value).strip()
        candidate_str = str(candidate_value).strip()
        if not row_str or not candidate_str:
            return False

        # Normalize case if fuzzy matching is not case sensitive.
        if not fuzzy_case_sensitive:
            row_str = row_str.lower()
            candidate_str = candidate_str.lower()

        # Tokenize the strings and check for any common tokens using set intersection.
        row_tokens = set(row_str.split())
        candidate_tokens = set(candidate_str.split())
        if not row_tokens & candidate_tokens:
            return False
    return True


def compute_fuzzy_scores(
    row: pd.Series,
    candidate_row: pd.Series,
    required_columns: Dict[str, str],
    fuzzy_case_sensitive: bool,
) -> List[int]:
    """
    Computes fuzzy matching scores for each required column using thefuzz's ratio.

    The function converts values to lowercase if fuzzy_case_sensitive is False.

    Parameters:
        row (pd.Series): A row from df1.
        candidate_row (pd.Series): A row from df2.
        required_columns (Dict[str, str]): Mapping of required column names in df1 to df2.
        fuzzy_case_sensitive (bool): Whether the matching should be case sensitive.

    Returns:
        List[int]: A list of fuzzy matching scores (one per required column).
    """
    scores = []
    for req_col in required_columns:
        val1 = str(row[req_col])
        val2 = str(candidate_row[required_columns[req_col]])
        if not fuzzy_case_sensitive:
            val1 = val1.lower()
            val2 = val2.lower()
        scores.append(fuzz.ratio(val1, val2))
    return scores


def compute_effective_threshold(candidate_min: int, fuzzy_threshold: int) -> int:
    """
    Rounds candidate_min down to the nearest multiple of 5, ensuring it does not drop below fuzzy_threshold.

    Parameters:
        candidate_min (int): The minimum fuzzy matching score for a candidate.
        fuzzy_threshold (int): The baseline threshold for a fuzzy match.

    Returns:
        int: The adjusted effective threshold.
    """
    return max(fuzzy_threshold, candidate_min // 5 * 5)


def perform_exact_matching(
    unmatched_df1: pd.DataFrame,
    df2_renamed: pd.DataFrame,
    required_columns: Dict[str, str],
    flexible_columns: Dict[str, str],
    combos: List[Tuple[str, ...]],
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Performs exact matching between df1 and df2 for various combinations of flexible columns.

    For each combination:
      - Merges df1 with df2 based on required and flexible columns.
      - If an exact match is found, assigns a raw match pattern (without a letter prefix)
        indicating which columns were used in the match.
      - Updates df1 to remove matched records.

    Parameters:
        unmatched_df1 (pd.DataFrame): The DataFrame of unmatched records from df1.
        df2_renamed (pd.DataFrame): df2 with renamed columns (suffixed with '_df2').
        required_columns (Dict[str, str]): Mapping of required columns between df1 and df2.
        flexible_columns (Dict[str, str]): Mapping of flexible columns between df1 and df2.
        combos (List[Tuple[str, ...]]): List of flexible column combinations to try.

    Returns:
        Tuple[List[pd.DataFrame], pd.DataFrame]:
            - A list of DataFrames containing the exact matches.
            - The updated unmatched df1 DataFrame.
    """
    results: List[pd.DataFrame] = []
    for combo in combos:
        if unmatched_df1.empty:
            break
        # Define the merge keys: required columns always plus the current flexible combo.
        left_keys = list(required_columns.keys()) + list(combo)
        right_keys = [f"{required_columns[k]}_df2" for k in required_columns] + [
            f"{flexible_columns[k]}_df2" for k in combo
        ]
        merged = pd.merge(
            unmatched_df1,
            df2_renamed,
            how="left",
            left_on=left_keys,
            right_on=right_keys,
            indicator=True,
        )
        exact_matches = merged[merged["_merge"] == "both"].copy()
        if exact_matches.empty:
            continue
        # Record the raw match pattern (e.g., "Name, Date, ID_number")
        match_pattern = ", ".join(list(required_columns.keys()) + list(combo))
        exact_matches["Match_Pattern"] = match_pattern
        exact_matches.drop(columns=["_merge"], inplace=True)
        results.append(exact_matches)
        # Remove matched records from unmatched_df1
        matched_indices = exact_matches["orig_index"].unique()
        unmatched_df1 = unmatched_df1[
            ~unmatched_df1["orig_index"].isin(matched_indices)
        ]
    return results, unmatched_df1


def perform_fuzzy_matching(
    unmatched_df1: pd.DataFrame,
    df2: pd.DataFrame,
    df2_renamed: pd.DataFrame,
    required_columns: Dict[str, str],
    flexible_columns: Dict[str, str],
    combos: List[Tuple[str, ...]],
    fuzzy_threshold: int,
    fuzzy_case_sensitive: bool,
    blocking: bool,
) -> List[dict]:
    """
    Performs fuzzy matching for records in df1 that were not exactly matched.

    For each unmatched record:
      - Iterates through the combinations of flexible columns.
      - For each candidate in df2, applies an optional blocking mechanism to reduce comparisons.
      - Uses fuzzy matching (fuzz.ratio) on required columns.
      - Records the raw match pattern (e.g., "Name (90% Fuzzy), Date") if a candidate meets the threshold.
      - If no candidate meets the threshold, assigns a "No match" pattern.

    Parameters:
        unmatched_df1 (pd.DataFrame): Unmatched records from df1.
        df2 (pd.DataFrame): The original df2 DataFrame.
        df2_renamed (pd.DataFrame): df2 with renamed columns.
        required_columns (Dict[str, str]): Mapping of required columns between df1 and df2.
        flexible_columns (Dict[str, str]): Mapping of flexible columns between df1 and df2.
        combos (List[Tuple[str, ...]]): List of flexible column combinations.
        fuzzy_threshold (int): Minimum fuzzy matching score for a candidate.
        fuzzy_case_sensitive (bool): Whether fuzzy matching is case sensitive.
        blocking (bool): Whether to apply the blocking mechanism.

    Returns:
        List[dict]: A list of dictionaries representing matched records with a raw match pattern.
    """
    fuzzy_results: List[dict] = []
    # Wrapped the DataFrame iterator in tqdm for progress monitoring.
    for _, row in tqdm(
        unmatched_df1.iterrows(), total=len(unmatched_df1), desc="Fuzzy matching rows"
    ):
        candidate_for_row = None
        best_combo_used: Tuple[str, ...] = ()
        best_effective_threshold: Any = None
        found_candidate = False

        # Try each flexible combo
        for combo in combos:
            local_candidate = None
            local_best_score = -1
            local_effective_threshold = None
            for _, row2 in df2.iterrows():
                # Ensure that flexible columns match exactly for the current combo.
                flexible_match = True
                for flex_col in combo:
                    if str(row[flex_col]) != str(row2[flexible_columns[flex_col]]):
                        flexible_match = False
                        break
                if not flexible_match:
                    continue

                # Optionally apply blocking to quickly filter out unlikely candidates.
                if blocking and not passes_blocking_optimized(
                    row, row2, required_columns, fuzzy_case_sensitive
                ):
                    continue

                # Compute fuzzy scores for required columns.
                scores = compute_fuzzy_scores(
                    row, row2, required_columns, fuzzy_case_sensitive
                )
                candidate_min = min(scores)
                if candidate_min >= fuzzy_threshold:
                    effective = compute_effective_threshold(
                        candidate_min, fuzzy_threshold
                    )
                    sum_score = sum(scores)
                    if sum_score > local_best_score:
                        local_best_score = sum_score
                        local_candidate = row2
                        local_effective_threshold = effective
            if local_candidate is not None:
                candidate_for_row = local_candidate
                best_combo_used = combo
                best_effective_threshold = local_effective_threshold
                found_candidate = True
                break

        result_row = row.to_dict()
        if found_candidate and candidate_for_row is not None:
            # Build the raw match pattern with fuzzy percentage details.
            req_label = ", ".join(
                [
                    f"{req_col} ({best_effective_threshold}% Fuzzy)"
                    for req_col in required_columns.keys()
                ]
            )
            if best_combo_used:
                match_pattern = req_label + ", " + ", ".join(best_combo_used)
            else:
                match_pattern = req_label
            # Retrieve matching candidate info from df2_renamed.
            candidate_renamed = df2_renamed.loc[candidate_for_row.name]
            for col in candidate_renamed.index:
                result_row[col] = candidate_renamed[col]
            result_row["Match_Pattern"] = match_pattern
        else:
            # Fallback: try an exact merge on the required columns only.
            fallback = pd.merge(
                pd.DataFrame([row]),
                df2_renamed,
                how="left",
                left_on=list(required_columns.keys()),
                right_on=[f"{required_columns[k]}_df2" for k in required_columns],
                indicator=True,
            )
            if not fallback.empty and fallback.iloc[0]["_merge"] == "both":
                candidate_renamed = fallback.iloc[0]
                for col in df2_renamed.columns:
                    result_row[col] = candidate_renamed[col]
            else:
                for col in df2_renamed.columns:
                    result_row[col] = pd.NA
            result_row["Match_Pattern"] = "No match"
        fuzzy_results.append(result_row)
    return fuzzy_results


def reorder_final_columns(
    final_df: pd.DataFrame, df1: pd.DataFrame, df2_renamed: pd.DataFrame
) -> pd.DataFrame:
    """
    Reorders the columns of the final merged DataFrame so that:
      - Original df1 columns come first.
      - Followed by 'orig_index' and 'Match_Category'.
      - Then the df2 columns.

    Parameters:
        final_df (pd.DataFrame): The merged DataFrame.
        df1 (pd.DataFrame): Original df1.
        df2_renamed (pd.DataFrame): df2 with renamed columns.

    Returns:
        pd.DataFrame: The reordered DataFrame.
    """
    df1_cols = list(df1.columns)
    df2_cols = list(df2_renamed.columns)
    final_cols = df1_cols + ["orig_index", "Match_Category"] + df2_cols
    for col in final_cols:
        if col not in final_df.columns:
            final_df[col] = pd.NA
    return final_df[final_cols]


def assign_consistent_categories(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-processes the final DataFrame to assign consistent letter labels
    to each unique match pattern. The sorting prioritizes:
      1. Exact matches (those without the term "Fuzzy").
      2. Fuzzy matches sorted in descending order by the fuzzy percentage.
      3. "No match" entries are placed last.

    Additionally, this function removes any "_lower" substrings from the match pattern,
    so that the final categories are cleaner (e.g., "Name_lower" becomes "Name").

    Parameters:
        final_df (pd.DataFrame): The merged DataFrame with a 'Match_Pattern' column.

    Returns:
        pd.DataFrame: The DataFrame with a consistent 'Match_Category' assigned.
    """
    # Remove "_lower" from the Match_Pattern column.
    final_df["Match_Pattern"] = final_df["Match_Pattern"].str.replace(
        "_lower", "", regex=False
    )

    def sort_key(pattern: str):
        # "No match" goes last.
        if pattern.lower() == "no match":
            return (3, 0)
        # For fuzzy patterns, extract the fuzzy percentage.
        if "fuzzy" in pattern.lower():
            m = re.search(r"\((\d+)% fuzzy\)", pattern.lower())
            percent = int(m.group(1)) if m else 0
            return (2, -percent)
        # Exact matches come first.
        else:
            return (1, 0)

    unique_patterns = final_df["Match_Pattern"].unique().tolist()
    sorted_patterns = sorted(unique_patterns, key=sort_key)
    pattern_to_label = {
        pattern: f"{chr(65+i)}) {pattern}" for i, pattern in enumerate(sorted_patterns)
    }
    final_df["Match_Category"] = final_df["Match_Pattern"].map(pattern_to_label)
    return final_df


def merge_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    required_columns: Dict[str, str],
    flexible_columns: Dict[str, str],
    fuzzy_threshold: int,
    fuzzy_case_sensitive: bool = True,
    blocking: bool = True,
) -> pd.DataFrame:
    """
    Main function to merge df1 and df2 using a combination of exact and fuzzy matching.

    The process is as follows:
      1. Validate that both DataFrames contain the expected columns.
      2. Rename df2 columns to avoid conflicts and record original indices.
      3. Attempt exact matching using various combinations of flexible columns.
      4. Apply fuzzy matching on the remaining unmatched df1 records.
      5. Combine the results and post-process to assign consistent letter labels to match patterns.
      6. Reorder the final columns for readability.

    Parameters:
        df1 (pd.DataFrame): The primary DataFrame to match from.
        df2 (pd.DataFrame): The secondary DataFrame to match against.
        required_columns (Dict[str, str]): Mapping of required columns (df1 key -> df2 value).
        flexible_columns (Dict[str, str]): Mapping of flexible columns (df1 key -> df2 value).
        fuzzy_threshold (int): Minimum fuzzy score to consider a match.
        fuzzy_case_sensitive (bool): Whether fuzzy matching is case sensitive.
        blocking (bool): Whether to apply blocking to optimize fuzzy matching.

    Returns:
        pd.DataFrame: The final merged DataFrame with consistent match categories.
    """
    # Data Integrity Checks: Validate that both DataFrames have the necessary columns.
    expected_df1_cols = list(required_columns.keys()) + list(flexible_columns.keys())
    validate_dataframe_columns(df1, expected_df1_cols, "df1")
    expected_df2_cols = list(required_columns.values()) + list(
        flexible_columns.values()
    )
    validate_dataframe_columns(df2, expected_df2_cols, "df2")

    # If fuzzy matching is not case sensitive, precompute lowercased versions
    # of the required and flexible columns to avoid repeated .lower() calls.
    if not fuzzy_case_sensitive:
        # Process df1: for each required and flexible column, create a lowercased version.
        for col in list(required_columns.keys()) + list(flexible_columns.keys()):
            df1[f"{col}_lower"] = df1[col].astype(str).str.lower()
        # Process df2 similarly.
        for col in list(required_columns.values()) + list(flexible_columns.values()):
            df2[f"{col}_lower"] = df2[col].astype(str).str.lower()

        # Update the mapping dictionaries to point to the cached lowercased columns.
        required_columns = {
            f"{k}_lower": f"{v}_lower" for k, v in required_columns.items()
        }
        flexible_columns = {
            f"{k}_lower": f"{v}_lower" for k, v in flexible_columns.items()
        }

    # Rename df2 columns to avoid conflicts and record the original index.
    df2_renamed = df2.rename(columns={col: f"{col}_df2" for col in df2.columns})
    df2_renamed["orig_index_df2"] = df2.index

    # Prepare df1 by adding an 'orig_index' column.
    unmatched_df1 = df1.copy()
    unmatched_df1["orig_index"] = unmatched_df1.index

    # Generate all combinations of flexible columns.
    flex_keys: List[str] = list(flexible_columns.keys())
    combos = generate_flexible_combos(flex_keys)
    results: List[pd.DataFrame] = []

    # Perform exact matching on df1.
    exact_results, unmatched_df1 = perform_exact_matching(
        unmatched_df1, df2_renamed, required_columns, flexible_columns, combos
    )
    results.extend(exact_results)

    # Perform fuzzy matching on the remaining unmatched df1 records.
    fuzzy_results = perform_fuzzy_matching(
        unmatched_df1,
        df2,
        df2_renamed,
        required_columns,
        flexible_columns,
        combos,
        fuzzy_threshold,
        fuzzy_case_sensitive,
        blocking,
    )
    if fuzzy_results:
        fuzzy_df = pd.DataFrame(fuzzy_results)
        results.append(fuzzy_df)

    # Combine all match results; if no matches, mark all as "No match".
    if results:
        final_df = pd.concat(results, ignore_index=True)
    else:
        final_df = df1.copy()
        final_df["Match_Pattern"] = "No match"
        for col in df2_renamed.columns:
            final_df[col] = pd.NA

    # Clean up the lower columns if they exist.
    if not fuzzy_case_sensitive:
        # Remove any cached lowercased columns (both _lower and _lower_df2) from final_df.
        final_df = final_df.loc[:, ~final_df.columns.str.contains("_lower")]

        # Also clean up the original DataFrame copies used for ordering.
        df1 = df1.loc[:, ~df1.columns.str.contains("_lower")]
        df2_renamed = df2_renamed.loc[:, ~df2_renamed.columns.str.contains("_lower")]

    # Post-process to assign consistent letter labels to match patterns.
    final_df = assign_consistent_categories(final_df)
    final_df = reorder_final_columns(final_df, df1, df2_renamed)
    return final_df


def main() -> None:
    """
    Example main function to test the merge process with sample data.

    This function creates sample DataFrames for df1 and df2, defines the matching criteria,
    runs the merge_dataframes function, and prints the final merged DataFrame along with
    separate outputs for matched and unmatched records.
    """
    # Define the columns that must match (required) and those that can be flexible.
    required_columns: Dict[str, str] = {"Name": "Name"}
    flexible_columns: Dict[str, str] = {"Date": "Date", "ID_number": "ID_number"}
    fuzzy_threshold: int = 75

    # Sample test data for df1 (with extra fuzzy cases).
    df1 = pd.DataFrame(
        [
            {"ID_number": "001", "Name": "Alice Smith", "Date": "2021-01-01"},
            {"ID_number": "002", "Name": "Bob Johnson", "Date": "2021-02-01"},
            {"ID_number": "003", "Name": "Charlie Brown", "Date": "2021-03-01"},
            {"ID_number": "004", "Name": "David Lee", "Date": "2021-04-01"},
            {"ID_number": "005", "Name": "Eve Adams", "Date": "2021-05-01"},
            {"ID_number": "006", "Name": "Frank Miller", "Date": "2021-06-01"},
            {"ID_number": "007", "Name": "Grace Hopper", "Date": "2021-07-01"},
            {"ID_number": "008", "Name": "Hank Aaron", "Date": "2021-08-01"},
            {"ID_number": "009", "Name": "Ivy Clark", "Date": "2021-09-01"},
            {"ID_number": "010", "Name": "Evelin Adams", "Date": "2021-05-01"},
            {"ID_number": "011", "Name": "Bobby Johnson", "Date": "2021-02-01"},
            {"ID_number": "012", "Name": "Charlee Brown", "Date": "2021-03-01"},
            {"ID_number": "013", "Name": "Daniell Lee", "Date": "2021-04-01"},
            {"ID_number": "014", "Name": "Zachary Taylor", "Date": "2021-10-01"},
        ]
    )

    # Sample test data for df2 (with extra candidates).
    df2 = pd.DataFrame(
        [
            {"ID_number": "001", "Name": "Alice Smith", "Date": "2021-01-01"},
            {"ID_number": "002", "Name": "Bob Johnson", "Date": "2021-02-15"},
            {"ID_number": "003", "Name": "Charles Brown", "Date": "2021-03-01"},
            {"ID_number": "004", "Name": "Daniel Lee", "Date": "2022-04-01"},
            {"ID_number": "005A", "Name": "Evelyn Adams", "Date": "2021-05-01"},
            {"ID_number": "006B", "Name": "Frank Miller", "Date": "X"},
            {"ID_number": "007C", "Name": "Grace Hopper", "Date": "2021-07-01"},
            {"ID_number": "008D", "Name": "Hank Aaron", "Date": "2021-08-01"},
            {"ID_number": "015", "Name": "Robert Johnson", "Date": "2021-02-15"},
            {"ID_number": "016", "Name": "David Le", "Date": "2021-04-01"},
        ]
    )

    # Run the merge function with the defined criteria.
    final_df = merge_dataframes(
        df1=df1,
        df2=df2,
        required_columns=required_columns,
        flexible_columns=flexible_columns,
        fuzzy_threshold=fuzzy_threshold,
        fuzzy_case_sensitive=False,
        blocking=True,
    )

    # Display the final merged DataFrame.
    print("Final Merged DataFrame:")
    print(final_df)

    # Optionally, split into matched and unmatched DataFrames for further analysis.
    matched_df1 = final_df[final_df["orig_index_df2"].notna()].copy()
    unmatched_df1 = final_df[final_df["orig_index_df2"].isna()].copy()

    # Clean the original df2:
    df2 = df2.loc[:, ~df2.columns.str.contains("_lower")]

    matched_df2_indices = matched_df1["orig_index_df2"].dropna().unique().tolist()
    matched_df2 = df2.loc[matched_df2_indices].copy()
    unmatched_df2 = df2.drop(matched_df2_indices, errors="ignore").copy()

    print("\nMatched df1:")
    print(matched_df1)
    print("\nUnmatched df1:")
    print(unmatched_df1)
    print("\nMatched df2:")
    print(matched_df2)
    print("\nUnmatched df2:")
    print(unmatched_df2)


if __name__ == "__main__":
    main()
