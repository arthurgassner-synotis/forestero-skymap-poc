from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import joblib
import rasterio
from pystac.item import Item as PySTACItem
from shapely.geometry import box

from .constants import SENTINEL_SCENES_FOLDERPATH
from .site import Site


@dataclass
class S2Item:
    id: str

    @property
    def folderpath(self) -> Path:
        return SENTINEL_SCENES_FOLDERPATH / self.id

    @property
    def stac_item(self) -> PySTACItem:
        stac_item_filepath = self.folderpath / "stac_item.joblib"
        return joblib.load(stac_item_filepath)

    @property
    def bbox_wgs84(self) -> tuple[float, float, float, float]:
        return tuple(self.stac_item.bbox)

    @property
    def crs(self) -> rasterio.crs.CRS:
        crs_str = self.stac_item.properties["proj:code"]
        return rasterio.crs.CRS.from_string(crs_str)

    @property
    def cloud_cover(self) -> float:
        return self.stac_item.properties["eo:cloud_cover"]

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(geometry=[box(*self.bbox_wgs84)], crs="EPSG:4326")

    @staticmethod
    def load_all() -> list["S2Item"]:
        s2item_filepaths = SENTINEL_SCENES_FOLDERPATH.glob("*/stac_item.joblib")
        return [S2Item(id=e.parent.name) for e in s2item_filepaths]

    @staticmethod
    def load_from_ids(s2item_ids: list[str]) -> list["S2Item"]:
        s2item_filepaths = SENTINEL_SCENES_FOLDERPATH.glob("*/stac_item.joblib")
        s2item_filepaths = [e for e in s2item_filepaths if e.parent.name in set(s2item_ids)]
        return [S2Item(id=e.parent.name) for e in s2item_filepaths]

    def overlaps(self, site: Site) -> bool:
        """Whether site overlaps with this S2Item."""

        minx2, miny2, maxx2, maxy2 = self.bbox_wgs84
        minx1, miny1, maxx1, maxy1 = site.bbox_wgs84

        return minx1 <= maxx2 and maxx1 >= minx2 and miny1 <= maxy2 and maxy1 >= miny2

    def contains(self, lon: float, lat: float) -> bool:
        min_lon, min_lat, max_lon, max_lat = self.bbox_wgs84
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return True

        return False

    @staticmethod
    def find_from_site(site: Site) -> list["S2Item"]:
        """Load S2Items that overlap with the provided site's bbox."""

        intersecting_s2item_ids = []
        for s2item in S2Item.load_all():
            if s2item.overlaps(site):
                intersecting_s2item_ids.append(s2item.id)

        return S2Item.load_from_ids(intersecting_s2item_ids)
