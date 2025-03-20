from uszipcode import SearchEngine
import pandas as pd

# NOTE: Ensure you have installed compatible versions of uszipcode and sqlalchemy-mate:
# Run the following command if you encounter issues:
# pip install "uszipcode<1.0.2" "sqlalchemy-mate<1.4"

# Initialize ZIP code search engine.
search = SearchEngine(simple_zipcode=True)  # Use simple=True for faster queries.

# Get all ZIP codes in California.
california_zips = search.by_state("CA", returns=0)  # returns=0 gets all ZIP codes.

# Extract ZIP codes and counties into a DataFrame.
zip_county_data = [(z.zipcode, z.county) for z in california_zips]

# Convert to pandas DataFrame.
df = pd.DataFrame(zip_county_data, columns=["ZIP Code", "County"])

# Define county region mapping.
northern_california = {
    "Alameda County", "Alpine County", "Amador County", "Butte County",
    "Calaveras County", "Colusa County", "Contra Costa County", "Del Norte County",
    "El Dorado County", "Glenn County", "Humboldt County", "Lake County",
    "Lassen County", "Marin County", "Mendocino County", "Modoc County",
    "Napa County", "Nevada County", "Placer County", "Plumas County",
    "Sacramento County", "San Francisco County", "San Mateo County",
    "Santa Clara County", "Santa Cruz County", "Shasta County",
    "Sierra County", "Siskiyou County", "Solano County", "Sonoma County",
    "Sutter County", "Tehama County", "Trinity County", "Yolo County",
    "Yuba County"
}

central_california = {
    "Fresno County", "Inyo County", "Kern County", "Kings County",
    "Madera County", "Mariposa County", "Merced County", "Mono County",
    "Monterey County", "San Benito County", "San Joaquin County",
    "San Luis Obispo County", "Santa Barbara County", "Stanislaus County",
    "Tulare County", "Tuolumne County"
}

southern_california = {
    "Imperial County", "Los Angeles County", "Orange County",
    "Riverside County", "San Bernardino County", "San Diego County",
    "Ventura County"
}

# Function to map county to region.
def assign_region(county):
    if county in northern_california:
        return "Northern California"
    elif county in central_california:
        return "Central California"
    elif county in southern_california:
        return "Southern California"
    else:
        return "Unknown"  # In case a county isn't classified.

# Apply region mapping to DataFrame.
df["Region"] = df["County"].apply(assign_region)

# Print the final DataFrame.
print(df.head())  # Show the first 5 rows.

# If you want to see all ZIP codes, print the entire DataFrame.
# print(df.to_string())  # Uncomment this to print the full DataFrame.

# Save DataFrame to CSV if needed.
df.to_csv("california_zip_county_region.csv", index=False)
print("Data saved to california_zip_county_region.csv")
