from pathlib import Path

import osmnx as ox

ROOT_FOLDERPATH = Path(__file__).parent.parent.parent
DATA_FOLDERPATH = ROOT_FOLDERPATH / "data"
BRONZE_FOLDERPATH = DATA_FOLDERPATH / "bronze"
SILVER_FOLDERPATH = DATA_FOLDERPATH / "silver"

# Ghana tree coords
GH_TREE_COORDS_FILEPATH = BRONZE_FOLDERPATH / "GH Tree coordinates.xlsx"

# Sentinel scene
SENTINEL_SCENES_FOLDERPATH = BRONZE_FOLDERPATH / "sentinel-scenes"

# ETHZ cocoa map
ETHZ_COCOA_MAP_FILEPATH = BRONZE_FOLDERPATH / "cocoa_map.tif"
ETHZ_COCOA_MAP_THRESHOLDED_FILEPATH = BRONZE_FOLDERPATH / "cocoa_map_th_065.tif"

# Ghana gdf
GHANA_GDF = ox.geocode_to_gdf("Ghana")
GH_ASHANTI_GDF = ox.geocode_to_gdf("Ashanti Region, Ghana")

# Earth-Search API URL
EARCH_SEARCH_API_URL = "https://earth-search.aws.element84.com/v1"
SENTINEL_BANDS = ["blue", "green", "red", "rededge1", "nir", "swir22"]

# ETHZ uint8 cocoa map
ETHZ_COCOA_MAP_UINT8_FILEPATH = SILVER_FOLDERPATH / "cocoa_map_uint8.tif"

# ETHZ masks
ETHZ_COCOA_MASK_FILEPATH = SILVER_FOLDERPATH / "cocoa_mask.tif"

SENTINEL_ASHANTI_SCENE_IDS_FILEPATH = SILVER_FOLDERPATH / "ashanti_2020_scene_ids.joblib"
