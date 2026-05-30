# City Selection Basis

Cities were selected based on population concentration from the `2020 Census of Population and Housing` and `Highly Urbanized City (HUC)` classification from the `Philippine Standard Geographic Code` of the `Philippine Statistics Authority (2020)`. Urban areas were prioritized because higher population density and human activity influence air pollution levels (`World Health Organization, 2024`). Cities were distributed across `Luzon`, `Visayas`, and `Mindanao` to reduce regional bias, and only locations available in the AQI dataset on `Kaggle (2026)` were included.

## Basis for Selection

### 1. Population Concentration

Cities with high population concentration were prioritized based on official `Philippine Statistics Authority (PSA)` population data from the `2020 Census of Population and Housing`.

Official source:

- PSA, `2020 Census of Population and Housing (2020 CPH)`
- National highlights: https://psa.gov.ph/content/highlights-philippine-population-2020-census-population-and-housing-2020-cph
- NCR highlights: https://psa.gov.ph/content/highlights-national-capital-region-ncr-population-2020-census-population-and-housing-2020

Population concentration was used because densely populated urban areas are more likely to experience sustained human activity and traffic conditions associated with changing air quality.

### 2. Highly Urbanized City Classification

Cities were also selected based on `Highly Urbanized City (HUC)` classification from the `Philippine Standard Geographic Code (PSGC)` of the `Philippine Statistics Authority`.

Official source:

- PSA, `Philippine Standard Geographic Code (PSGC) - Highly Urbanized Cities`
- https://psa.gov.ph/classification/psgc/hucs
- PSA, `Philippine Standard Geographic Code (PSGC)`
- https://psa.gov.ph/classification/psgc

HUC classification was used because highly urbanized cities are more likely to show the urban activity patterns, traffic flow, and land-use intensity that can influence AQI behavior.

### 3. Urban Relevance to AQI

Urban areas were prioritized because higher population density and human activity influence air pollution levels.

Reference:

- World Health Organization (WHO), `2024`

### 4. Geographic Coverage

Cities were distributed across the country's main island groups to reduce regional bias:

- `Luzon`
- `Visayas`
- `Mindanao`

Within the current dataset, this is represented by cities from `Luzon`, `Visayas`, and `Mindanao`.

Regional grouping in the project is implemented in:

- [constants.py](d:/Dev/Python/ph-aqi/constants.py)

## Selected Cities

The current selected 10 cities are:

- `Manila`
- `Quezon City`
- `Makati City`
- `Pasig`
- `Cebu City`
- `Lapu-Lapu City`
- `Iloilo City`
- `Bacolod`
- `Davao`
- `Cagayan de Oro`

These cities were chosen because they satisfy the selection basis on population concentration, Highly Urbanized City classification, and geographic coverage.

## AQI Dataset Inclusion Constraint

Only cities available in the AQI dataset used by the study were included.

Dataset source:

- Kaggle dataset: `Philippine Cities Air Quality Index Data 2026` (`Kaggle, 2026`)
- https://www.kaggle.com/datasets/bwandowando/philippine-cities-air-quality-index-data-2026

Local project files derived from that dataset:

- [datasets/all/2026/2026_CombinedData.csv](d:/Dev/Python/ph-aqi/datasets/all/2026/2026_CombinedData.csv)
- [datasets/selected/2026/2026_SelectedData_Imputed.csv](d:/Dev/Python/ph-aqi/datasets/selected/2026/2026_SelectedData_Imputed.csv)
