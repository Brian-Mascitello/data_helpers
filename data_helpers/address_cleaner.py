import re
from typing import Dict, List, Optional, Pattern, Tuple

import pandas as pd

# Improved regex patterns
CITY_PATTERN = re.compile(r"^[A-Za-z\s\.]+$")

CITY_STATE_ZIP_PATTERN = re.compile(r",\s[A-Z]{2}\s\d{5}(-\d{4})?$")

EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

EXTRACT_CITY_STATE_ZIP_PATTERN = re.compile(r"^(.*),\s([A-Z]{2})\s(\d{5}(-\d{4})?)$")

PHONE_PATTERN = re.compile(
    r"^(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}"
    r"(?:\s*(?:ext[:\.]?\s*[0-9]+(?:[a-zA-Z0-9-]*)?))?$",
    re.IGNORECASE,
)

SUITE_PATTERN: Pattern[str] = re.compile(
    r"(?i)\b#?(?:apartment|apt|building|bldg|floor|fl|room|ste|suite|unit)[\s#\-]*([0-9A-Za-z]+)\b",
    re.IGNORECASE,
)

WEBSITE_PATTERN = re.compile(
    r"^(https?://)?(www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(/.*)?$", re.IGNORECASE
)

ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")

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

# Allowed maximum occurrences per row (except for leading names)
ALLOWED_MAX = {
    "address": 4,
    "city": 1,
    "email": 1,
    "name": 1,
    "phone": 4,
    "state": 1,
    "website": 1,
    "zip": 1,
}


def move_suite_to_next_address(
    df: pd.DataFrame,
    address1_col: str = "address1_fix",
    address2_col: str = "address2_fix",
    pattern: Pattern[str] = SUITE_PATTERN,
) -> pd.DataFrame:
    """
    Moves suite/unit information from address1_fix to address2_fix if address2_fix is blank.

    Parameters:
    df (pd.DataFrame): The dataframe containing address columns.
    address1_col (str): The column name for the first address (default: 'address1_fix').
    address2_col (str): The column name for the second address (default: 'address2_fix').
    pattern (Pattern[str]): Regex pattern to identify suite/unit info (default: SUITE_PATTERN).

    Returns:
    pd.DataFrame: Updated dataframe with suite/unit info moved to address2_fix.
    """

    def extract_and_move(row: pd.Series) -> pd.Series:
        if pd.isna(row[address2_col]) or row[address2_col] == "":
            match = pattern.search(str(row[address1_col]))
            if match:
                row[address2_col] = match.group(0)  # Capture full suite/unit phrase
                row[address1_col] = pattern.sub(
                    "", str(row[address1_col])
                ).strip()  # Remove from address1_fix
        return row

    return df.apply(extract_and_move, axis=1)


def detect_address_component(value: str) -> Optional[str]:
    """
    Detects and classifies a value into an address component category.
    Returns None for empty or "nan" values.
    Uses CITY_PATTERN to classify values composed solely of letters, spaces, and periods as cities.
    """
    if value is None or pd.isna(value):
        return None
    value = value.strip()
    if not value or value.lower() == "nan":
        return None
    elif EMAIL_PATTERN.match(value):
        return "email"
    elif WEBSITE_PATTERN.match(value):
        return "website"
    elif CITY_STATE_ZIP_PATTERN.search(value):
        return "city_state_zip"
    elif ZIP_PATTERN.match(value):
        return "zip"
    elif PHONE_PATTERN.match(value):
        return "phone"
    elif value in US_STATES:
        return "state"
    # alpha and digit
    # elif any(char.isalpha() for char in value) and any(char.isdigit() for char in value):
    # digit
    # elif any(char.isdigit() for char in value):
    # Suite pattern or (alpha and digit) or (# and digit)
    elif (
        SUITE_PATTERN.match(value)
        or (
            any(char.isalpha() for char in value)
            and any(char.isdigit() for char in value)
        )
        or ("#" in value and any(char.isdigit() for char in value))
    ):
        return "address"
    elif CITY_PATTERN.match(value):
        return "city"
    else:
        return "misc"


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
    Note: city_state_zip occurrences are also counted as ZIP occurrences.
    """
    categories = list(ALLOWED_MAX.keys())
    max_counts = {cat: 0 for cat in categories}

    for _, row in df.iterrows():
        local_counts = {cat: 0 for cat in categories}
        # Skip the leading columns designated as names
        for idx, col in enumerate(columns_to_check):
            if idx < leading_name:
                continue
            raw_value: str = str(row.get(col, "")).strip()
            category: Optional[str] = detect_address_component(raw_value)
            if category == "city_state_zip":
                # Count as a ZIP occurrence
                local_counts["zip"] += 1
            elif category in local_counts:
                local_counts[category] += 1
        for cat in categories:
            max_counts[cat] = max(max_counts[cat], local_counts[cat])

    # Cap counts to ALLOWED_MAX values
    for cat, cap in ALLOWED_MAX.items():
        max_counts[cat] = min(max_counts[cat], cap)
    # Override name count with leading_name if provided
    if leading_name > 0:
        max_counts["name"] = leading_name
    return max_counts


def generate_dynamic_columns(
    max_counts: Dict[str, int], suffix_str: str = "_fix"
) -> List[str]:
    """
    Generates dynamic column names in the fixed order:
    name, address, city, state, zip, phone, email, website.
    For each category, even if the maximum count is 1, the column name ends with a trailing "1" plus suffix.
    """
    order = ["name", "address", "city", "state", "zip", "phone", "email", "website"]
    dynamic_columns = []
    for cat in order:
        count = max_counts.get(cat, 1)
        for i in range(1, count + 1):
            dynamic_columns.append(f"{cat}{i}{suffix_str}")
    return dynamic_columns


def fix_row(
    row: pd.Series,
    columns_to_check: List[str],
    dynamic_columns: List[str],
    city_strategy: str = "first",
    leading_name: int = 0,
    suffix_str: str = "_fix",
) -> pd.Series:
    """
    Processes a single row, classifies address components, and assigns data dynamically.
    The first `leading_name` columns are treated as names.
    If a city is extracted from a city_state_zip value, it is preserved as the first city when using the "first" strategy.
    """
    # Initialize dictionary with keys in the order defined by dynamic_columns.
    data_dict: Dict[str, str] = {col: "" for col in dynamic_columns}
    # Initialize counters for each component.
    found = {cat: 1 for cat in ALLOWED_MAX.keys()}
    found["name"] = 1
    city_from_csz = (
        False  # flag indicating that a city was found from a city_state_zip value
    )

    # Define key names using the suffix.
    city_key = f"city1{suffix_str}"
    state_key = f"state1{suffix_str}"

    for idx, col in enumerate(columns_to_check):
        raw_value: str = str(row.get(col, "")).strip()
        if idx < leading_name:
            key = f"name{idx+1}{suffix_str}"
            if raw_value:
                data_dict[key] = raw_value
            continue

        category: Optional[str] = detect_address_component(raw_value)

        if category == "city_state_zip":
            city, state, zip_code = extract_city_state_zip(raw_value)
            # Always treat the city from city_state_zip as primary.
            if not data_dict[city_key]:
                data_dict[city_key] = city
            else:
                if city_strategy == "concatenate":
                    data_dict[city_key] = f"{city}, {data_dict[city_key]}"
                elif city_strategy in ["first", "last"]:
                    data_dict[city_key] = city
            data_dict[state_key] = state

            allowed_zip = sum(1 for c in dynamic_columns if c.startswith("zip"))
            if found["zip"] <= allowed_zip:
                data_dict[f"zip{found['zip']}{suffix_str}"] = zip_code
                found["zip"] += 1
            city_from_csz = True

        elif category == "city":
            if not data_dict[city_key]:
                data_dict[city_key] = raw_value
            else:
                if city_from_csz:
                    if city_strategy == "concatenate":
                        data_dict[city_key] = f"{data_dict[city_key]}, {raw_value}"
                    # For "first", do nothing.
                else:
                    if city_strategy == "concatenate":
                        data_dict[city_key] = f"{data_dict[city_key]}, {raw_value}"
                    elif city_strategy == "last":
                        data_dict[city_key] = raw_value

        elif category == "misc":
            if not data_dict[city_key]:
                data_dict[city_key] = raw_value
            else:
                if city_from_csz:
                    if city_strategy == "concatenate":
                        data_dict[city_key] = f"{data_dict[city_key]}, {raw_value}"
                else:
                    if city_strategy == "concatenate":
                        data_dict[city_key] = f"{data_dict[city_key]}, {raw_value}"
                    elif city_strategy == "last":
                        data_dict[city_key] = raw_value

        elif category == "state":
            data_dict[state_key] = raw_value

        elif category in ["zip", "phone", "address", "email", "website", "name"]:
            allowed = sum(1 for c in dynamic_columns if c.startswith(category))
            if found[category] <= allowed:
                key = f"{category}{found[category]}{suffix_str}"
                data_dict[key] = raw_value
                found[category] += 1

    ordered_data = {col: data_dict[col] for col in dynamic_columns}
    return pd.Series(ordered_data)


def clean_addresses_df(
    df: pd.DataFrame,
    columns_to_check: List[str],
    city_strategy: str = "first",
    append_columns: bool = False,
    leading_name: int = 0,
    suffix_str: str = "_fix",
) -> pd.DataFrame:
    """
    Cleans address components in a pandas DataFrame and returns the modified DataFrame.
    If append_columns is True, the new fixed columns are appended to the original DataFrame.
    The parameter leading_name indicates how many leading columns to treat as names.
    The parameter suffix_str is appended to each generated column name.
    """
    max_counts = calculate_max_counts(df, columns_to_check, leading_name)
    dynamic_columns = generate_dynamic_columns(max_counts, suffix_str)
    fix_df = df.apply(
        lambda row: fix_row(
            row,
            columns_to_check,
            dynamic_columns,
            city_strategy,
            leading_name,
            suffix_str,
        ),
        axis=1,
    )
    if append_columns:
        result_df = pd.concat(
            [df.reset_index(drop=True), fix_df.reset_index(drop=True)], axis=1
        )
        return result_df
    else:
        return fix_df


def clean_addresses(
    input_csv: str,
    output_csv: str,
    columns_to_check: List[str],
    city_strategy: str = "first",
    append_columns: bool = False,
    leading_name: int = 0,
    suffix_str: str = "_fix",
    move_suite: bool = False,
) -> pd.DataFrame:
    """
    Loads a CSV, cleans only the specified columns, and saves the cleaned data to a new CSV.
    If append_columns is True, the new fixed columns are appended to the original data.
    The parameter leading_name indicates how many leading columns to treat as names.
    The parameter suffix_str is appended to each generated column name.
    If move_suite is True, the new suites are attemped to be separated from the 'address1_fix' and 'address2_fix' columns.
    """
    df: pd.DataFrame = pd.read_csv(input_csv, dtype=str)
    fix_df = clean_addresses_df(
        df, columns_to_check, city_strategy, append_columns, leading_name, suffix_str
    )

    if move_suite:
        fix_df = move_suite_to_next_address(fix_df)

    fix_df.to_csv(output_csv, index=False)
    return fix_df


if __name__ == "__main__":
    # Example usage: adjust file paths, column names, and parameters as needed.
    input_file: str = "test_file.csv"  # Replace with your CSV file path
    output_file: str = "cleaned_test_file.csv"
    columns_to_check: List[str] = [
        "COMPANY",
        "ADDRESS1",
        "ADDRESS2",
        "CITY",
        "PHONE",
        "FAX",
        "PHONE2",
    ]  # Columns to clean
    # For example, if the first column should be treated as a name, set leading_name to 1.
    cleaned_df: pd.DataFrame = clean_addresses(
        input_file,
        output_file,
        columns_to_check,
        city_strategy="first",
        append_columns=True,
        leading_name=1,
        suffix_str="_fix",
        move_suite=True,
    )
    print(f"Cleaned data saved to {output_file}")
