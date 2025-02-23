import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Improved regex patterns
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
WEBSITE_PATTERN = re.compile(
    r"^(https?://)?(www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(/.*)?$", re.IGNORECASE
)
# Updated city, state, and ZIP patterns to be more flexible
CITY_STATE_ZIP_PATTERN = re.compile(r",\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\s*$")
ZIP_PATTERN = re.compile(r"^\d{5}(?:-\d{4})?$")
# Updated phone pattern:
# Matches standard 10-digit phone numbers with optional extension.
# If an extension is present (e.g., "ext", "Ext", "x"), it requires at least one digit.
PHONE_PATTERN = re.compile(
    r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:ext[:\.]?\s*[0-9]+(?:[a-zA-Z0-9-]*)?))?$",
    re.IGNORECASE,
)
EXTRACT_CITY_STATE_ZIP_PATTERN = re.compile(
    r"^(.*?),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)\s*$"
)

# Set of valid US state abbreviations
US_STATES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}

# Allowed maximum occurrences per row (except names)
ALLOWED_MAX = {
    "address": 4,
    "phone": 4,
    "zip": 1,
    "email": 1,
    "website": 1,
    "company": 1,
}


def detect_address_component(value: str) -> Optional[str]:
    """
    Detects and classifies a value into an address component category.
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
    if CITY_STATE_ZIP_PATTERN.search(value):
        return "city_state_zip"
    elif ZIP_PATTERN.match(value):
        return "zip"
    elif PHONE_PATTERN.match(value):
        return "phone"
    elif value in US_STATES:
        return "state"
    elif any(char.isdigit() for char in value) and any(
        char.isalpha() for char in value
    ):
        return "address"
    else:
        return "city"


def extract_city_state_zip(value: str) -> Tuple[str, str, str]:
    """
    Extracts city, state, and ZIP from a 'City, ST ZIP' formatted string.
    """
    match = EXTRACT_CITY_STATE_ZIP_PATTERN.search(value)
    if match:
        city, state, zip_code = match.group(1), match.group(2), match.group(3)
        return city.strip(), state.strip(), zip_code.strip()
    return value.strip(), "", ""


def calculate_max_counts(
    df: pd.DataFrame, columns_to_check: List[str], leading_name: int = 0
) -> Dict[str, int]:
    """
    Calculates the maximum number of occurrences per row for each component category,
    then caps them using ALLOWED_MAX. Also accounts for a fixed number of leading name fields.
    Note: If a value is identified as 'city_state_zip', we increment the zip count so that a zip column is created.
    """
    categories = list(ALLOWED_MAX.keys())
    max_counts = {cat: 0 for cat in categories}

    for _, row in df.iterrows():
        local_counts = {cat: 0 for cat in categories}
        # Skip the leading columns that are designated as names
        for idx, col in enumerate(columns_to_check):
            if idx < leading_name:
                continue
            raw_value: str = str(row.get(col, "")).strip()
            category: Optional[str] = detect_address_component(raw_value)
            if category in local_counts:
                local_counts[category] += 1
            # Count city_state_zip occurrences as zip occurrences too.
            if category == "city_state_zip":
                local_counts["zip"] += 1
        for cat in categories:
            max_counts[cat] = max(max_counts[cat], local_counts[cat])

    # Cap counts to ALLOWED_MAX values
    for cat, cap in ALLOWED_MAX.items():
        max_counts[cat] = min(max_counts[cat], cap)
    # Add name count if applicable
    if leading_name > 0:
        max_counts["name"] = leading_name
    return max_counts


def generate_dynamic_columns(max_counts: Dict[str, int]) -> List[str]:
    """
    Generates dynamic column names based on maximum counts.
    Names come first (if present), then city and state, followed by other categories.
    """
    dynamic_columns = []
    if "name" in max_counts and max_counts["name"] > 0:
        dynamic_columns.extend([f"name{i}" for i in range(1, max_counts["name"] + 1)])
    dynamic_columns.extend(["city", "state"])
    for category in sorted(max_counts.keys()):
        if category == "name":
            continue
        dynamic_columns.extend(
            [f"{category}{i}" for i in range(1, max_counts[category] + 1)]
        )
    return dynamic_columns


def fix_row(
    row: pd.Series,
    columns_to_check: List[str],
    dynamic_columns: List[str],
    city_strategy: str = "first",
    leading_name: int = 0,
) -> pd.Series:
    """
    Processes a single row, classifies address components, and assigns data dynamically.
    The first `leading_name` columns are treated as names.
    """
    data_dict: Dict[str, str] = {col: "" for col in dynamic_columns}
    # Initialize counters for each component
    found = {cat: 1 for cat in ALLOWED_MAX.keys()}
    found["name"] = 1

    for idx, col in enumerate(columns_to_check):
        raw_value: str = str(row.get(col, "")).strip()

        # Process leading columns as names
        if idx < leading_name:
            key = f"name{idx+1}"
            if raw_value:
                data_dict[key] = raw_value
            continue

        category: Optional[str] = detect_address_component(raw_value)

        if category == "city_state_zip":
            city, state, zip_code = extract_city_state_zip(raw_value)
            data_dict["city"] = (
                city
                if not data_dict["city"]
                else (
                    f"{data_dict['city']}, {city}"
                    if city_strategy == "concatenate"
                    else city
                )
            )
            data_dict["state"] = state

            allowed_zip = sum(1 for c in dynamic_columns if c.startswith("zip"))
            if found["zip"] <= allowed_zip:
                data_dict[f"zip{found['zip']}"] = zip_code
                found["zip"] += 1
        elif category in ["zip", "phone", "address", "email", "website", "company"]:
            allowed = sum(1 for c in dynamic_columns if c.startswith(category))
            if found[category] <= allowed:
                data_dict[f"{category}{found[category]}"] = raw_value
                found[category] += 1
        elif category == "state":
            data_dict["state"] = raw_value
        elif category == "city":
            data_dict["city"] = (
                raw_value
                if not data_dict["city"]
                else (
                    f"{data_dict['city']}, {raw_value}"
                    if city_strategy == "concatenate"
                    else raw_value
                )
            )

    return pd.Series(dict(sorted(data_dict.items())))


def clean_addresses_df(
    df: pd.DataFrame,
    columns_to_check: List[str],
    city_strategy: str = "first",
    append_columns: bool = False,
    leading_name: int = 0,
) -> pd.DataFrame:
    """
    Cleans address components in a pandas DataFrame and returns the modified DataFrame.
    If append_columns is True, the new fixed columns are appended to the original DataFrame.
    The parameter leading_name indicates how many leading columns to treat as names.
    """
    max_counts = calculate_max_counts(df, columns_to_check, leading_name)
    dynamic_columns = generate_dynamic_columns(max_counts)
    fixed_df = df.apply(
        lambda row: fix_row(
            row, columns_to_check, dynamic_columns, city_strategy, leading_name
        ),
        axis=1,
    )
    if append_columns:
        result_df = pd.concat(
            [df.reset_index(drop=True), fixed_df.reset_index(drop=True)], axis=1
        )
        return result_df
    else:
        return fixed_df


def clean_addresses(
    input_csv: str,
    output_csv: str,
    columns_to_check: List[str],
    city_strategy: str = "first",
    append_columns: bool = False,
    leading_name: int = 0,
) -> pd.DataFrame:
    """
    Loads a CSV, cleans only the specified columns, and saves the cleaned data to a new CSV.
    If append_columns is True, the new fixed columns are appended to the original data.
    The parameter leading_name indicates how many leading columns to treat as names.
    """
    df: pd.DataFrame = pd.read_csv(input_csv, dtype=str)
    fixed_df = clean_addresses_df(
        df, columns_to_check, city_strategy, append_columns, leading_name
    )
    fixed_df.to_csv(output_csv, index=False)
    return fixed_df


if __name__ == "__main__":
    # Example usage: adjust file paths, column names, and parameters as needed.
    input_file: str = "your_file.csv"  # Replace with your CSV file path
    output_file: str = "cleaned_file.csv"
    columns_to_check: List[str] = ["Column1", "Column2", "Column3"]  # Columns to clean
    # For example, if the first column should be treated as a name, set leading_name to 1.
    cleaned_df: pd.DataFrame = clean_addresses(
        input_file,
        output_file,
        columns_to_check,
        city_strategy="first",
        append_columns=True,
        leading_name=1,
    )
    print(f"Cleaned data saved to {output_file}")
