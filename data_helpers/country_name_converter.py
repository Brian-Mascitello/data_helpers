import pandas as pd

country_abbreviations = {
    "Albania": "AL",
    "Antigua and Barbuda": "AG",
    "Argentina": "AR",
    "Austria": "AT",
    "Bahamas": "BS",
    "Bangladesh": "BD",
    "Barbados": "BB",
    "Belarus": "BY",
    "Belgium": "BE",
    "Belize": "BZ",
    "Bosnia and Herzegovina": "BA",
    "Brazil": "BR",
    "Bulgaria": "BG",
    "Canada": "CA",
    "China": "CN",
    "Costa Rica": "CR",
    "Croatia": "HR",
    "Cuba": "CU",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Dominican Republic": "DO",
    "Egypt": "EG",
    "El Salvador": "SV",
    "Ethiopia": "ET",
    "Estonia": "EE",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Greece": "GR",
    "Guatemala": "GT",
    "Haiti": "HT",
    "Honduras": "HN",
    "Hungary": "HU",
    "Iceland": "IS",
    "India": "IN",
    "Indonesia": "ID",
    "Iran": "IR",
    "Ireland": "IE",
    "Italy": "IT",
    "Jamaica": "JM",
    "Japan": "JP",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Malta": "MT",
    "Mexico": "MX",
    "Mongolia": "MN",
    "Montenegro": "ME",
    "Myanmar": "MM",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nicaragua": "NI",
    "Nigeria": "NG",
    "North Korea": "KP",
    "North Macedonia": "MK",
    "Norway": "NO",
    "Pakistan": "PK",
    "Panama": "PA",
    "Philippines": "PH",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Russia": "RU",
    "Saint Kitts and Nevis": "KN",
    "Saint Lucia": "LC",
    "Saint Vincent and the Grenadines": "VC",
    "Serbia": "RS",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Taiwan": "TW",
    "Thailand": "TH",
    "Trinidad and Tobago": "TT",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United Kingdom": "GB",
    "United States": "US",
    "Vietnam": "VN",
}

# Reverse dictionary for abbreviation to country name lookup
abbreviation_to_country = {v: k for k, v in country_abbreviations.items()}


def country_to_abbreviation(country_name: str) -> str:
    """Converts a country name to its two-letter abbreviation (case insensitive)."""
    return country_abbreviations.get(country_name.title(), "Invalid Country Name")


def abbreviation_to_country_name(abbreviation: str) -> str:
    """Converts a two-letter abbreviation to the full country name."""
    return abbreviation_to_country.get(abbreviation.upper(), "Invalid Abbreviation")


if __name__ == "__main__":
    # Example usage
    print(country_to_abbreviation("Germany"))  # Output: DE
    print(abbreviation_to_country_name("JP"))  # Output: Japan

    # Example usage with pandas DataFrame
    data = {
        "Country": [
            "United States",
            "Canada",
            "Mexico",
            "France",
            "Japan",
            "South Korea",
            "Germany",
            "India",
            "Brazil",
            "Nigeria",
        ]
    }
    df = pd.DataFrame(data)

    # Apply country_to_abbreviation function
    df["Country_Abbreviation"] = df["Country"].map(country_to_abbreviation)
    print(df)

    # If the column contains abbreviations and you want full country names
    df["Full_Country_Name"] = df["Country_Abbreviation"].map(
        abbreviation_to_country_name
    )
    print(df)
