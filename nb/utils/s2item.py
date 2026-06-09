from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import rasterio
from loguru import logger
from pystac.item import Item as PySTACItem

from .constants import SENTINEL_SCENES_FOLDERPATH
from .site import Site

PX_TO_M = 10


@dataclass
class S2Item:
    id: str

    @property
    def stac_item(self) -> PySTACItem:
        stac_item_filepath = SENTINEL_SCENES_FOLDERPATH / self.id / "stac_item.joblib"
        return joblib.load(stac_item_filepath)

    @property
    def bbox_wgs84(self) -> tuple[float, float, float, float]:
        return tuple(self.stac_item.bbox)

    @property
    def crs(self) -> rasterio.crs.CRS:
        crs_str = self.stac_item.properties["proj:code"]
        return rasterio.crs.CRS.from_string(crs_str)

    @staticmethod
    def load_all() -> list["S2Item"]:
        s2item_filepaths = SENTINEL_SCENES_FOLDERPATH.glob("*/stac_item.joblib")
        return [S2Item(id=e.parent.name) for e in s2item_filepaths]

    @staticmethod
    def load_from_ids(s2item_ids: list[str]) -> list["S2Item"]:
        s2item_filepaths = SENTINEL_SCENES_FOLDERPATH.glob("*/stac_item.joblib")
        s2item_filepaths = [e for e in s2item_filepaths if e.parent.name in set(s2item_ids)]
        return [S2Item(id=e.parent.name) for e in s2item_filepaths]

    def has_overlap(self, site: Site) -> bool:
        """Whether site overlaps with this S2Item."""

        minx2, miny2, maxx2, maxy2 = self.bbox_wgs84
        minx1, miny1, maxx1, maxy1 = site.bbox_wgs84

        return minx1 <= maxx2 and maxx1 >= minx2 and miny1 <= maxy2 and maxy1 >= miny2

    @staticmethod
    def _load_raster(p: Path) -> np.ndarray:
        """Load raster image."""
        with rasterio.open(p) as src:
            raster = src.read(1)

            # 0.0 is the no-data value
            raster = raster.astype(float)
            raster[raster == 0.0] = np.nan

            return raster

    @staticmethod
    def _load_bounds_and_crs(scene_id: str) -> tuple[rasterio.coords.BoundingBox, rasterio.crs.CRS]:
        all_bounds, all_crs = [], []
        for suffix in ["_red", "_green", "_blue", "_nir", "_swir22"]:
            p = SENTINEL_SCENES_FOLDERPATH / scene_id / f"{scene_id}{suffix}.tif"
            with rasterio.open(p) as src:
                all_bounds.append(src.bounds)
                all_crs.append(src.crs)

        if len(set(all_bounds)) != 1:
            logger.error(f"Several bounds found for {scene_id}: {set(all_bounds)}")
            raise ValueError()

        if len(set(all_crs)) != 1:
            logger.error(f"Several CRS's found for {scene_id}: {set(all_crs)}")
            raise ValueError()

        return all_bounds[0], all_crs[0]

    @staticmethod
    def find_from_site(site: Site) -> list["S2Item"]:
        """Load S2Items that overlap with the provided site's bbox."""

        intersecting_s2_item_ids = []
        for s2_item in S2Item.load_all():
            if s2_item.has_overlap(site):
                intersecting_s2_item_ids.append(s2_item.id)

        return S2Item.load_from_ids(intersecting_s2_item_ids)
