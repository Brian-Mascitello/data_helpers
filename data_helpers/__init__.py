"""
data_helpers package

This module provides various utilities for data processing, including:
- Column utilities
- Address cleaning
- Fuzzy grouping
- Country/state name conversion
- Email validation and extraction
"""

from .address_cleaner import (
    calculate_max_counts,
    clean_addresses,
    detect_address_component,
    extract_city_state_zip,
    fix_row,
    generate_dynamic_columns,
    handle_city,
)
from .column_utils import (
    get_unique_column_name,
    standardize_column_names,
)
from .country_name_converter import (
    abbreviation_to_country_name,
    country_to_abbreviation,
)
from .email_checker import (
    extract_emails,
    get_email_components,
    has_valid_mx,
    is_disposable_email,
    is_valid_email,
    is_valid_email_lib,
    normalize_email,
    suggest_email_fix,
)
from .fuzzy_grouping import assign_similarity_groups
from .state_name_converter import (
    abbreviation_to_state_name,
    state_to_abbreviation,
)

__all__ = [
    "calculate_max_counts",
    "clean_addresses",
    "detect_address_component",
    "extract_city_state_zip",
    "fix_row",
    "generate_dynamic_columns",
    "handle_city",
    "get_unique_column_name",
    "standardize_column_names",
    "abbreviation_to_country_name",
    "country_to_abbreviation",
    "extract_emails",
    "get_email_components",
    "has_valid_mx",
    "is_disposable_email",
    "is_valid_email",
    "is_valid_email_lib",
    "normalize_email",
    "suggest_email_fix",
    "assign_similarity_groups",
    "abbreviation_to_state_name",
    "state_to_abbreviation",
]
