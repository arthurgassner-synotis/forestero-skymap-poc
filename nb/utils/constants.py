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

# Ghana gdf
GHANA = ox.geocode_to_gdf("Ghana")

# Earth-Search API URL
EARCH_SEARCH_API_URL = "https://earth-search.aws.element84.com/v1"
SENTINEL_BANDS = ["blue", "green", "red", "nir", "swir22"]
