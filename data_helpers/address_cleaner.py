import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Compile regex patterns at module level
CITY_STATE_ZIP_PATTERN = re.compile(r",\s[A-Z]{2}\s\d{5}(-\d{4})?$")
ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")
PHONE_PATTERN = re.compile(r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")
EXTRACT_CITY_STATE_ZIP_PATTERN = re.compile(r"^(.*),\s([A-Z]{2})\s(\d{5}(-\d{4})?)$")
EMAIL_PATTERN = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
WEBSITE_PATTERN = re.compile(
    r"^(https?://)?(www\.)?[\w\.-]+\.[a-z]{2,}$", re.IGNORECASE
)


def detect_address_component(value: str) -> Optional[str]:
    """
    Detects and classifies a value into an address component category:
    address, city, state, ZIP, phone, email, or website.
    The input string is stripped before processing.
    """
    if value is None or pd.isna(value):
        return None
    value = value.strip()
    if not value:
        return None
    if EMAIL_PATTERN.match(value):
        return "email"
    if WEBSITE_PATTERN.match(value):
        return "website"
    if CITY_STATE_ZIP_PATTERN.search(value):  # Detect "City, ST ZIP"
        return "city_state_zip"
    elif ZIP_PATTERN.match(value):  # ZIP code (5 or 9 digits)
        return "zip"
    elif PHONE_PATTERN.match(value):  # Phone number
        return "phone"
    elif len(value) == 2 and value.isalpha():  # State code
        return "state"
    elif any(char.isdigit() for char in value) and any(
        char.isalpha() for char in value
    ):
        return "address"  # Likely an address if both digits & letters appear
    else:
        return "city"  # Default to city if it’s just text


def extract_city_state_zip(value: str) -> Tuple[str, str, str]:
    """
    Extracts city, state, and ZIP from a 'City, ST ZIP' formatted string.
    """
    match = EXTRACT_CITY_STATE_ZIP_PATTERN.search(value)
    if match:
        city, state, zip_code = match.group(1), match.group(2), match.group(3)
        return city.strip(), state.strip(), zip_code.strip()
    return value.strip(), "", ""  # If no match, assume it's just a city


def handle_city(existing: str, new: str, strategy: str = "first") -> str:
    """
    Handles the assignment of city names based on the chosen strategy.

    Parameters:
        existing (str): The current city value.
        new (str): The new city value to incorporate.
        strategy (str): "first" (default), "last", or "concatenate".

    Returns:
        str: The resulting city value.
    """
    new = new.strip()
    if not existing:
        return new
    if strategy == "concatenate":
        return f"{existing}, {new}"
    elif strategy == "last":
        return new
    # Default ("first") strategy: keep the existing city
    return existing


def calculate_max_counts(
    df: pd.DataFrame, columns_to_check: List[str]
) -> Dict[str, int]:
    """
    Calculates the maximum number of occurrences for address, phone, ZIP, email, and website.
    """
    max_counts: Dict[str, int] = {
        "address": 1,
        "phone": 1,
        "zip": 1,
        "email": 1,
        "website": 1,
    }
    for _, row in df.iterrows():
        for col in columns_to_check:
            value: str = str(row[col]) if col in row else ""
            # The value is stripped inside detect_address_component
            category: Optional[str] = detect_address_component(value)
            if category in max_counts:
                max_counts[category] += 1
    return max_counts


def generate_dynamic_columns(max_counts: Dict[str, int]) -> List[str]:
    """
    Generates dynamic column names based on maximum counts.
    """
    dynamic_columns: List[str] = ["city", "state"]
    for category in sorted(max_counts.keys()):
        count = max_counts[category]
        dynamic_columns.extend([f"{category}{i}" for i in range(1, count + 1)])
    return dynamic_columns


def fix_row(
    row: pd.Series,
    columns_to_check: List[str],
    dynamic_columns: List[str],
    city_strategy: str = "first",
) -> pd.Series:
    """
    Processes a single row, classifies address components, and assigns data dynamically.

    Returns:
        pd.Series: A Series with dictionary keys sorted alphabetically.
    """
    # Initialize the dictionary with dynamic column keys
    data_dict: Dict[str, str] = {col: "" for col in dynamic_columns}
    found: Dict[str, int] = {
        "address": 1,
        "phone": 1,
        "zip": 1,
        "email": 1,
        "website": 1,
    }  # Track duplicate occurrences

    for col in columns_to_check:
        raw_value: str = str(row[col]) if col in row else ""
        value: str = raw_value.strip()
        category: Optional[str] = detect_address_component(value)

        if category == "city_state_zip":
            city, state, zip_code = extract_city_state_zip(value)
            data_dict["city"] = handle_city(data_dict["city"], city, city_strategy)
            data_dict["state"] = state  # Overwrite state regardless
            zip_key = f"zip{found['zip']}"
            data_dict[zip_key] = zip_code
            found["zip"] += 1
        elif category in ["zip", "phone", "address", "email", "website"]:
            key = f"{category}{found[category]}"
            data_dict[key] = value
            found[category] += 1
        elif category == "state":
            data_dict["state"] = value
        elif category == "city":
            data_dict["city"] = handle_city(data_dict["city"], value, city_strategy)

    # Return a Series with dictionary keys sorted in alphabetical order
    return pd.Series(dict(sorted(data_dict.items())))


def clean_addresses(
    input_csv: str,
    output_csv: str,
    columns_to_check: List[str],
    city_strategy: str = "first",
) -> pd.DataFrame:
    """
    Loads a CSV, cleans only the specified columns, and saves the cleaned data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns sorted alphabetically.
    """
    df: pd.DataFrame = pd.read_csv(input_csv, dtype=str)  # Read all data as strings

    # Calculate maximum counts and generate dynamic column names
    max_counts = calculate_max_counts(df, columns_to_check)
    dynamic_columns = generate_dynamic_columns(max_counts)

    # Process each row using the dynamic columns and city strategy
    df_fixed: pd.DataFrame = df.apply(
        lambda row: fix_row(row, columns_to_check, dynamic_columns, city_strategy),
        axis=1,
    )

    # Save the cleaned DataFrame
    df_fixed.to_csv(output_csv, index=False)
    return df_fixed


if __name__ == "__main__":
    # Example usage - update file paths and column names as needed
    input_file: str = "your_file.csv"  # Change this to your actual file path
    output_file: str = "cleaned_file.csv"
    columns_to_check: List[str] = ["Column1", "Column2", "Column3"]

    print(f"Cleaning addresses using columns: {columns_to_check}...")
    cleaned_df: pd.DataFrame = clean_addresses(
        input_file, output_file, columns_to_check, city_strategy="first"
    )
    print(f"Cleaned data saved to {output_file}")
