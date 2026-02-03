import ee

# Earth Engine Project Config
# Make sure to run `earthengine authenticate` in your terminal first.
EE_PROJECT_ID = 'nimble-anagram-486108-e1' # Updated with valid Project ID

# Data Config
SENTINEL_BANDS = ['B2', 'B3', 'B4'] # RGB
SCALE_LR = 10 # Sentinel-2 is 10m
SCALE_HR = 1.25 # Target resolution (approx 8x upscaling)

# Regions of Interest (Diverse Cities for better generalization)
# Format: [Lon, Lat] (GEE uses GeoJSON format)
ROIs = {
    'SF_Urban': [[-122.43, 37.75], [-122.41, 37.75], [-122.41, 37.77], [-122.43, 37.77]],
    'NY_Dense': [[-74.01, 40.70], [-73.99, 40.70], [-73.99, 40.72], [-74.01, 40.72]], # Manhattan
    'LA_Sprawl': [[-118.25, 34.05], [-118.23, 34.05], [-118.23, 34.07], [-118.25, 34.07]], # Downtown LA
    'Austin_Suburb': [[-97.75, 30.25], [-97.73, 30.25], [-97.73, 30.27], [-97.75, 30.27]]
}
