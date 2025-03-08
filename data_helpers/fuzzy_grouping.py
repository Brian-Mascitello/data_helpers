import itertools

import pandas as pd
from column_utils import get_unique_column_name
from thefuzz import fuzz


def assign_similarity_groups(
    df: pd.DataFrame,
    name_column: str,
    group_column: str = "Group_Number",
    start_group: int = 1,
    threshold: int = 70,
    case_insensitive: bool = True,
    presorted: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Assigns a group number to rows based on similarity of a specified column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        name_column (str): Column name to compare for similarity.
        group_column (str): Name of the new group column (ensuring uniqueness).
        start_group (int): Initial group number (default: 1).
        threshold (int): Similarity threshold (default: 70).
        case_insensitive (bool): Whether to compare names in a case-insensitive manner (default: True).
        presorted (bool): Whether the DataFrame is already sorted (default: False).
        verbose (bool): Whether to print debug information about similarity comparisons (default: False).

    Returns:
        pd.DataFrame: DataFrame with the new group column added.
    """
    if name_column not in df.columns:
        raise ValueError(f"Column '{name_column}' not found in DataFrame.")

    if df.empty:
        return df.copy()

    # Ensure unique column name
    group_column = get_unique_column_name(df, group_column)

    # If only one row, return immediately
    if len(df) == 1:
        df[group_column] = start_group
        return df

    # Sort alphabetically if not presorted
    if not presorted:
        df = df.sort_values(
            by=name_column, key=lambda x: x.str.lower() if case_insensitive else x
        ).reset_index(drop=True)

    # Normalize names for comparison
    names = df[name_column].fillna("")
    if case_insensitive:
        names = names.str.lower()
    names = names.values  # Convert to NumPy array for efficiency

    # Initialize grouping
    group_number = start_group
    group_numbers = [group_number]

    # Compare similarity efficiently using itertools.pairwise()
    for prev, curr in itertools.pairwise(names):
        similarity = fuzz.ratio(prev, curr)
        if verbose:
            print(
                f"Comparing: '{prev}' ↔ '{curr}' | Similarity: {similarity} | Threshold: {threshold} → {'New Group' if similarity < threshold else 'Same Group'}"
            )

        if similarity < threshold:
            group_number += 1  # New group if similarity is low

        group_numbers.append(group_number)

    df[group_column] = group_numbers
    return df


def main():
    # Sample DataFrame for testing
    data = {
        "Name": [
            "John Doe",
            "Jon Doe",
            "Jane Doe",
            "Janet Doe",
            "Jack Daniels",
            "Jake Daniels",
        ],
    }
    df = pd.DataFrame(data)

    # Run the function with test parameters
    result_df = assign_similarity_groups(
        df,
        name_column="Name",
        threshold=80,  # Adjust similarity threshold
        verbose=True,  # Show comparison details
    )

    # Display results
    print(result_df)


if __name__ == "__main__":
    main()
