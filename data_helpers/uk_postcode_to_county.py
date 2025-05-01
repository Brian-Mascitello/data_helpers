"""
UK Postcode to County and Region Mapping Script
------------------------------------------------

This script uses two datasets downloaded from the ONS Geoportal:

1. ONS Postcode Directory (February 2025):
   https://geoportal.statistics.gov.uk/datasets/6fb8941d58e54d949f521c92dfb92f2a/about
   - Contains postcode units and administrative geographies (e.g., LAD codes).

2. Postcode to OA to LSOA to MSOA to LAD (February 2025) Best Fit Lookup:
   https://geoportal.statistics.gov.uk/datasets/80592949bebd4390b2cbe29159a75ef4/about
   - Maps postcodes to local authority names (LAD), and small statistical areas (OA, LSOA, MSOA).
"""

import pandas as pd

# Load ONSPD data (postcodes and LAD codes)
onspd_df = pd.read_csv(
    r"data_helpers\CSV_Files\ONSPD_FEB_2025_UK.csv", usecols=["pcds", "oslaua"]
)

lad_lookup_df = pd.read_csv(
    r"data_helpers\CSV_Files\PCD_OA21_LSOA21_MSOA21_LAD_FEB25_UK_LU.csv",
    usecols=["ladcd", "ladnm"],
)

# Merge postcodes with local authority names
postcode_to_county = onspd_df.merge(
    lad_lookup_df[["ladcd", "ladnm"]].drop_duplicates(),
    left_on="oslaua",
    right_on="ladcd",
    how="left",
)

# Define postcode area to region mapping based on the visual map
area_to_region = {
    "CB": "East Anglia",
    "CO": "East Anglia",
    "IP": "East Anglia",
    "NR": "East Anglia",
    "SS": "East Anglia",
    "BR": "London",
    "E": "London",
    "EC": "London",
    "EN": "London",
    "HA": "London",
    "IG": "London",
    "N": "London",
    "NW": "London",
    "RM": "London",
    "SE": "London",
    "SW": "London",
    "UB": "London",
    "W": "London",
    "WC": "London",
    "WD": "London",
    "B": "Midlands",
    "CV": "Midlands",
    "DE": "Midlands",
    "DY": "Midlands",
    "LE": "Midlands",
    "LN": "Midlands",
    "NG": "Midlands",
    "NN": "Midlands",
    "PE": "Midlands",
    "ST": "Midlands",
    "SY": "Midlands",
    "TF": "Midlands",
    "WR": "Midlands",
    "WS": "Midlands",
    "WV": "Midlands",
    "BD": "North",
    "CA": "North",
    "DH": "North",
    "DL": "North",
    "DN": "North",
    "HD": "North",
    "HG": "North",
    "HU": "North",
    "HX": "North",
    "LS": "North",
    "NE": "North",
    "S": "North",
    "SR": "North",
    "TS": "North",
    "WF": "North",
    "YO": "North",
    "BB": "North West",
    "BL": "North West",
    "CH": "North West",
    "CW": "North West",
    "FY": "North West",
    "L": "North West",
    "LA": "North West",
    "M": "North West",
    "OL": "North West",
    "PR": "North West",
    "SK": "North West",
    "WA": "North West",
    "WN": "North West",
    "BT": "Northern Ireland",
    "AB": "Scotland",
    "DD": "Scotland",
    "DG": "Scotland",
    "EH": "Scotland",
    "FK": "Scotland",
    "G": "Scotland",
    "HS": "Scotland",
    "IV": "Scotland",
    "KA": "Scotland",
    "KW": "Scotland",
    "KY": "Scotland",
    "ML": "Scotland",
    "PA": "Scotland",
    "PH": "Scotland",
    "TD": "Scotland",
    "ZE": "Scotland",
    "AL": "South East",
    "BN": "South East",
    "CM": "South East",
    "CR": "South East",
    "CT": "South East",
    "DA": "South East",
    "GU": "South East",
    "HP": "South East",
    "KT": "South East",
    "LU": "South East",
    "ME": "South East",
    "MK": "South East",
    "OX": "South East",
    "PO": "South East",
    "RG": "South East",
    "RH": "South East",
    "SG": "South East",
    "SL": "South East",
    "SM": "South East",
    "SO": "South East",
    "TN": "South East",
    "TW": "South East",
    "BA": "South West",
    "BH": "South West",
    "BS": "South West",
    "DT": "South West",
    "EX": "South West",
    "GL": "South West",
    "PL": "South West",
    "SN": "South West",
    "SP": "South West",
    "TA": "South West",
    "TQ": "South West",
    "TR": "South West",
    "CF": "Wales",
    "HR": "Wales",
    "LD": "Wales",
    "LL": "Wales",
    "NP": "Wales",
    "SA": "Wales",
}

# Optional: Reverse lookup dictionary (region to list of postcode area prefixes)
region_to_area = {
    "East Anglia": [
        "CB",
        "CO",
        "IP",
        "NR",
        "SS",
    ],
    "London": [
        "BR",
        "E",
        "EC",
        "EN",
        "HA",
        "IG",
        "N",
        "NW",
        "RM",
        "SE",
        "SW",
        "UB",
        "W",
        "WC",
        "WD",
    ],
    "Midlands": [
        "B",
        "CV",
        "DE",
        "DY",
        "LE",
        "LN",
        "NG",
        "NN",
        "PE",
        "ST",
        "SY",
        "TF",
        "WR",
        "WS",
        "WV",
    ],
    "North": [
        "BD",
        "CA",
        "DH",
        "DL",
        "DN",
        "HD",
        "HG",
        "HU",
        "HX",
        "LS",
        "NE",
        "S",
        "SR",
        "TS",
        "WF",
        "YO",
    ],
    "North West": [
        "BB",
        "BL",
        "CH",
        "CW",
        "FY",
        "L",
        "LA",
        "M",
        "OL",
        "PR",
        "SK",
        "WA",
        "WN",
    ],
    "Northern Ireland": ["BT"],
    "Scotland": [
        "AB",
        "DD",
        "DG",
        "EH",
        "FK",
        "G",
        "HS",
        "IV",
        "KA",
        "KW",
        "KY",
        "ML",
        "PA",
        "PH",
        "TD",
        "ZE",
    ],
    "South East": [
        "AL",
        "BN",
        "CM",
        "CR",
        "CT",
        "DA",
        "GU",
        "HP",
        "KT",
        "LU",
        "ME",
        "MK",
        "OX",
        "PO",
        "RG",
        "RH",
        "SG",
        "SL",
        "SM",
        "SO",
        "TN",
        "TW",
    ],
    "South West": [
        "BA",
        "BH",
        "BS",
        "DT",
        "EX",
        "GL",
        "PL",
        "SN",
        "SP",
        "TA",
        "TQ",
        "TR",
    ],
    "Wales": [
        "CF",
        "HR",
        "LD",
        "LL",
        "NP",
        "SA",
    ],
}

# Sort area_to_region
# sorted_items = sorted(area_to_region.items(), key=lambda item: (item[1], item[0]))
# sorted_dict = dict(sorted_items)
# for key, value in sorted_dict.items():
#     print(f"'{key}': '{value}',")

# Swap keys and values to create region_to_area in case it is useful.
# from collections import defaultdict
# swapped = defaultdict(list)
# for k, v in area_to_region.items():
#     swapped[v].append(k)
# for key in swapped:
#     swapped[key].sort()
# sorted_swapped_items = sorted(swapped.items(), key=lambda item: (item[0], item[1]))
# sorted_swapped_dict = dict(sorted_swapped_items)
# for key, value_list in sorted_swapped_dict.items():
#     print(f"'{key}': {value_list},")

# Extract postcode area prefix (first 1–2 letters)
postcode_to_county["area"] = postcode_to_county["pcds"].str.extract(r"^([A-Z]{1,2})")

# Map area to region
postcode_to_county["region"] = (
    postcode_to_county["area"].map(area_to_region).fillna("Unknown")
)

# Final output: postcode, county, region
final_df = postcode_to_county[["pcds", "ladnm", "region"]]
final_df = final_df.sort_values(by="pcds")
final_df.to_csv(r"data_helpers\CSV_Files\uk_postcode_to_county_region.csv", index=False)
print(f"Total postcodes processed: {len(final_df)}")

# Print the head of the final_df.
print(final_df.head())

# Catch Unknown Regions.
unknown_df = final_df[final_df["region"] == "Unknown"].copy()
unknown_df["code"] = unknown_df["pcds"].str[:2]
unknown_df = unknown_df[["region", "code"]]
unknown_df = unknown_df.drop_duplicates()
unknown_df = unknown_df.sort_values(by="code")
unknown_df.to_csv(r"data_helpers\CSV_Files\uk_unknown_regions.csv", index=False)
print(f"Unmapped postcode areas: {len(unknown_df)}")
