from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import rasterio
from loguru import logger
from pystac.item import Item as PySTACItem
from scipy.ndimage import zoom

from .constants import SENTINEL_SCENES_FOLDERPATH

PX_TO_M = 10


@dataclass
class STACItem:
    id: str
    bounds: rasterio.coords.BoundingBox
    crs: rasterio.CRS

    @staticmethod
    def get_downloaded_scene_ids() -> list[str]:
        scene_filepaths = SENTINEL_SCENES_FOLDERPATH.glob("*/stac_item.joblib")
        return [e.parent.name for e in scene_filepaths]

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
    def _load_bbox_and_crs(scene_id: str) -> tuple[tuple[float, float, float, float], rasterio.crs.CRS]:
        p = SENTINEL_SCENES_FOLDERPATH / scene_id
        if not p.exists():
            raise ValueError()

        stac_item: PySTACItem = joblib.load(p / "stac_item.joblib")
        crs_str = stac_item.properties["proj:code"]
        bbox_wgs84 = tuple(stac_item.bbox)

        return bbox_wgs84, rasterio.crs.CRS.from_string(crs_str)

    @staticmethod
    def find_from_bbox(bbox_wgs84: tuple[float, float, float, float]) -> list[str]:
        """Load STACItems that itersects with the provided WGS84 bbox."""

        # Unpack the target WGS84 bbox: (min_lon, min_lat, max_lon, max_lat)
        minx1, miny1, maxx1, maxy1 = bbox_wgs84

        intersecting_scene_ids = []
        for scene_id in STACItem.get_downloaded_scene_ids():
            bbox_wgs84, _ = STACItem._load_bbox_and_crs(scene_id)

            # Check if bbox intersects with bbox_wgs84 using AABB collision logic
            minx2, miny2, maxx2, maxy2 = bbox_wgs84
            intersects = minx1 <= maxx2 and maxx1 >= minx2 and miny1 <= maxy2 and maxy1 >= miny2
            if intersects:
                intersecting_scene_ids.append(scene_id)

        return intersecting_scene_ids

    @staticmethod
    def load_from_id(scene_id: str) -> "STACItem":
        """Load a STACItem from the data available at SENTINEL_SCENES_FOLDERPATH/"""

        # Figure out datetime
        date_str = scene_id.split("_")[2]
        yyyy = int(date_str[:4])
        mm = int(date_str[4:6])
        dd = int(date_str[6:])
        dt = date(yyyy, mm, dd)

        # Load each raster in their .tif
        p = SENTINEL_SCENES_FOLDERPATH / scene_id
        red = STACItem._load_raster(p / f"{p.name}_red.tif")
        green = STACItem._load_raster(p / f"{p.name}_green.tif")
        blue = STACItem._load_raster(p / f"{p.name}_blue.tif")
        red_edge = STACItem._load_raster(p / f"{p.name}_rededge1.tif")
        nir = STACItem._load_raster(p / f"{p.name}_nir.tif")
        swir = STACItem._load_raster(p / f"{p.name}_swir22.tif")

        red_edge = zoom(red_edge, zoom=2, order=1)  # Zoom into red edge, since one pix == 20m x 20m
        swir = zoom(swir, zoom=2, order=1)  # Zoom into SWIR, since one pix == 20m x 20m

        rgb_re_nir_swir = np.dstack((red, green, blue, red_edge, nir, swir)).astype("float32")

        # Load CRS and bounds
        original_bounds, original_crs = STACItem._load_bounds_and_crs(scene_id)

        return STACItem(bounds=original_bounds, dt=dt, scene_id=scene_id, rgb_re_nir_swir=rgb_re_nir_swir, crs=original_crs)
