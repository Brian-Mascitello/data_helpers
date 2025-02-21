import pandas as pd

state_abbreviations = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

# Reverse dictionary for abbreviation to state name lookup
abbreviation_to_state = {v: k for k, v in state_abbreviations.items()}


def state_to_abbreviation(state_name: str) -> str:
    """Converts a state name to its two-letter abbreviation (case insensitive)."""
    return state_abbreviations.get(state_name.title(), "Invalid State Name")


def abbreviation_to_state_name(abbreviation: str) -> str:
    """Converts a two-letter abbreviation to the full state name."""
    return abbreviation_to_state.get(abbreviation.upper(), "Invalid Abbreviation")


if __name__ == "__main__":
    # Example usage
    print(state_to_abbreviation("California"))  # Output: CA
    print(abbreviation_to_state_name("TX"))  # Output: Texas

    # Example usage with pandas DataFrame
    data = {"State": ["California", "Texas", "New York", "Florida", "Ohio"]}
    df = pd.DataFrame(data)

    # Apply state_to_abbreviation function
    df["State_Abbreviation"] = df["State"].map(state_to_abbreviation)
    print(df)

    # If the column contains abbreviations and you want full state names
    df["Full_State_Name"] = df["State_Abbreviation"].map(abbreviation_to_state_name)
    print(df)
