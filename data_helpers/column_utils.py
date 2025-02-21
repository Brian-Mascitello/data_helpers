import itertools
import re

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

        import pandas as pd


def standardize_column_names(
    df: pd.DataFrame, separator: str = "_", case: str = "lower"
) -> pd.DataFrame:
    """
    Standardizes column names by converting case, replacing spaces with a chosen separator,
    and removing special characters.

    Parameters:
        df (pd.DataFrame): The DataFrame whose column names need standardization.
        separator (str): The character to replace spaces with ("_" or " "). Default is "_".
        case (str): Desired case format: "lower", "upper", "title", or "capitalize". Default is "lower".

    Returns:
        pd.DataFrame: A DataFrame with standardized column names.
    """
    if separator not in ["_", " "]:
        raise ValueError("separator must be '_' or ' ' (single space)")

    if case not in ["lower", "upper", "title", "capitalize"]:
        raise ValueError("case must be 'lower', 'upper', 'title', or 'capitalize'")

    def clean_column(column: str) -> str:
        column = re.sub(
            r"\W+", " ", column
        ).strip()  # Remove special characters, keep spaces
        column = column.replace(
            " ", separator
        )  # Replace spaces with the chosen separator

        # Convert case based on the specified format
        if case == "lower":
            column = column.lower()  # All lowercase
        elif case == "upper":
            column = column.upper()  # ALL UPPERCASE
        elif case == "title":
            if separator == "_":
                column = "_".join(
                    word.capitalize() for word in column.split("_")
                )  # Capitalize Each Word, Preserve Underscores
            else:
                column = column.title()  # Capitalize Each Word with Spaces
        elif case == "capitalize":
            column = column.capitalize()  # Only First Letter Capitalized

        return column

    df = df.rename(columns=lambda x: clean_column(x))
    return df
