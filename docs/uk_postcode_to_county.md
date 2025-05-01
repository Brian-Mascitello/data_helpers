# UK Postcode to County and Region Mapping

This script (`uk_postcode_to_county_region.py`) maps full UK postcodes to their corresponding local authority (county/district) and broad UK region (such as "Scotland", "North West", or "Wales").

It is part of the `data_helpers` module and uses official datasets provided by the UK Office for National Statistics (ONS).

## Data Sources

This script uses the following datasets, provided under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/):

1. **ONS Postcode Directory (February 2025)**  
   [View dataset](https://geoportal.statistics.gov.uk/datasets/6fb8941d58e54d949f521c92dfb92f2a/about)  
   - Contains postcode units and administrative geographies such as Local Authority Districts (LADs).

2. **Postcode to OA to LSOA to MSOA to LAD (February 2025) Best Fit Lookup**  
   [View dataset](https://geoportal.statistics.gov.uk/datasets/80592949bebd4390b2cbe29159a75ef4/about)  
   - Maps postcodes to LAD names and statistical areas including Output Areas (OA), Lower Layer Super Output Areas (LSOA), and Middle Layer Super Output Areas (MSOA).

## Region Classification

Broad UK regions (e.g., "South East", "Wales") are assigned based on the postcode area prefix (first one or two letters), using a custom lookup dictionary. This mapping is based on visual interpretation of a publicly available UK postcode map.

## Note on Data Files

This repository does not include the ONS CSV datasets due to their size and licensing considerations.

To run the script:

1. Download the required datasets from the links provided above.
2. Place them in the `data_helpers/CSV_Files/` directory.

To ensure these large files are not committed to the repository, a `.gitignore` rule is in place:

```
data_helpers/CSV_Files/*.csv
```

## Disclaimer

This script uses public data for general informational or analytical purposes. For official or regulated use, refer directly to the datasets provided by the Office for National Statistics (ONS).
