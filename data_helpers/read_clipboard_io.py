import datetime
from typing import Any, List, Optional, Union

import pandas as pd


def generate_filename() -> str:
    """Generates a timestamped filename in the format YYYYMMDD_HHMMSS_rci.csv."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_rci.csv")


def save_dataframe(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Saves the DataFrame to a CSV file. If no path is provided, generates a unique filename.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        save_path (str, optional): Custom file path. Defaults to a timestamped filename.
    """
    if save_path is None:
        save_path = generate_filename()
        print(f"\nNo filename provided. Using auto-generated filename: {save_path}")

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\nData successfully saved to: {save_path}")


def extract_and_copy_list(
    df: pd.DataFrame,
    convert_to_str: bool = True,
    clean_copy: bool = False,
    save: bool = False,
    save_path: Optional[str] = None,
) -> Union[List[Any], List[List[Any]]]:
    """
    Extracts data from a DataFrame as either a single list (if one column) or a list of lists (if multiple columns).
    Copies the formatted data to the clipboard and optionally saves it to a CSV file.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        convert_to_str (bool): If True, converts all values to strings. If False, keeps numeric types.
        clean_copy (bool): If True, copies a clean version without brackets.
        save (bool): If True, saves the extracted data to a CSV file.
        save_path (str, optional): Custom path for saving the file. If None, a timestamped filename is used.

    Returns:
        List[Union[str, int, float]] if one column, otherwise List[List[Union[str, int, float]]].
    """
    if df is None or df.empty:  # Now handles both cases safely
        print("\nWarning: Clipboard data is empty. Returning an empty list.")
        return []

    if df.shape[1] == 1:  # Single column case
        python_list = df.iloc[:, 0].tolist()
        if clean_copy:
            list_str = " ".join(map(str, python_list))
        else:
            list_str = str(python_list)
    else:  # Multiple columns case
        python_list = (
            df.astype(str).values.tolist() if convert_to_str else df.values.tolist()
        )
        list_str = "\n".join(" ".join(map(str, row)) for row in python_list)

    # Copy formatted list string to clipboard
    pd.DataFrame([list_str]).to_clipboard(index=False, header=False)

    print("\nExtracted Data and copied to clipboard:")
    print(list_str)

    # Save to file if enabled
    if save:
        save_dataframe(df, save_path)

    return python_list


def read_from_clipboard(
    print_df: bool = True,
    save_to_clipboard: bool = True,
    sep: str = "\t",
    sort: bool = False,
    ascending: bool = True,
    return_list: bool = False,
    convert_to_str: bool = False,
    save: bool = False,
    save_path: Optional[str] = None,
    header: Union[int, List[int], None] = None,
) -> Union[pd.DataFrame, List[Any], List[List[Any]]]:
    """
    Reads tabular data from clipboard into a Pandas DataFrame and processes it.

    Parameters:
        print_df (bool): Whether to print the DataFrame after reading.
        save_to_clipboard (bool): Whether to copy the DataFrame back to the clipboard.
        sep (str): Separator used when copying data back to clipboard (default is tab '\t').
        sort (bool): Whether to sort the DataFrame.
        ascending (bool): Sort order (True for ascending, False for descending). Default is True.
        return_list (bool): If True, extracts data as a Python list and copies it to clipboard.
        convert_to_str (bool): If True, converts all values to strings. If False, keeps numeric types.
        save (bool): If True, saves the data to a CSV file.
        save_path (str, optional): Custom path for saving the file. If None, a timestamped filename is used.
        header (int, list of int, None): Row number(s) to use as column names.

    Returns:
        pd.DataFrame, List, or List[List]: The processed DataFrame or extracted data.
    """
    try:
        df = pd.read_clipboard(header=header)
        print("Data successfully read from clipboard.")

        if convert_to_str:
            df = df.astype(str)

        if sort and len(df) > 1:
            df = df.sort_values(by=list(df.columns), ascending=ascending)
            sort_order = "ascending" if ascending else "descending"
            print(f"\nData sorted by columns: {list(df.columns)} ({sort_order}).")

        if return_list:
            if df.empty:
                print(
                    "\nWarning: No valid data found in clipboard. Returning an empty list."
                )
                return []
            return extract_and_copy_list(
                df, convert_to_str, save=save, save_path=save_path
            )

        if print_df:
            print("\n", df)

        if save_to_clipboard:
            df.to_clipboard(index=False, sep=sep)
            print(f"\nData copied back to clipboard using separator: '{sep}'")

        return df
    except Exception as e:
        print(f"Error reading clipboard: {e}")
        return None


if __name__ == "__main__":
    print("Testing read_clipboard_io.py...\n")

    # df = read_from_clipboard()  # Default Test

    # df = read_from_clipboard(print_df=False)  # Printing Disabled

    # df = read_from_clipboard(save_to_clipboard=False)  # Copying Disabled

    # df = read_from_clipboard(sep=',')  # Custom Separator

    # df = read_from_clipboard(sort=True)  # Sort Ascending

    # df = read_from_clipboard(sort=True, ascending=False)  # Sort Descending

    # python_list = read_from_clipboard(return_list=True, convert_to_str=True)  # Return List or List of Lists (Auto)

    # python_list = read_from_clipboard(return_list=True, convert_to_str=False)  # Return List or List of Lists (Auto, No String Conversion)

    # read_from_clipboard(return_list=True, save=True)  # Save with Auto Filename

    # read_from_clipboard(return_list=True, save=True, save_path="my_data.csv")  # Save with Custom Filename

    read_from_clipboard(return_list=True, save=True, sort=True)  # Save sorted list with auto filename

    print(
        "\nTesting complete. Copy different data into your clipboard and rerun the script for more testing."
    )
