import itertools

import pandas as pd


def get_unique_column_name(df: pd.DataFrame, column_name: str) -> str:
    """
    Ensures column name is unique by appending a number if needed.

    Parameters:
        df (pd.DataFrame): The DataFrame to check for column name conflicts.
        column_name (str): The desired column name.

    Returns:
        str: A unique column name that doesn't conflict with existing columns.
    """
    if column_name not in df.columns:
        return column_name

    for counter in itertools.count(1):  # Infinite counter until a free name is found
        new_name = f"{column_name}_{counter}"
        if new_name not in df.columns:
            return new_name
