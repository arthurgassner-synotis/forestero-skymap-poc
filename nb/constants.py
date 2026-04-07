from pathlib import Path

ROOT_FOLDERPATH = Path(__file__).parent.parent
DATA_FOLDERPATH = ROOT_FOLDERPATH / "data"
BRONZE_FOLDERPATH = DATA_FOLDERPATH / "bronze"

# Ghana tree coords
GH_TREE_COORDS_FILEPATH = BRONZE_FOLDERPATH / "GH Tree coordinates.xlsx"

# Sentinel scene
SENTINEL_SCENES_FOLDERPATH = BRONZE_FOLDERPATH / "sentinel-scenes"