from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from .constants import SENTINEL_SCENES_FOLDERPATH


@dataclass
class SentinelScene:
    _bounds: rasterio.coords.BoundingBox
    _crs: rasterio.crs.CRS
    scene_id: str
    red: np.ndarray
    green: np.ndarray
    blue: np.ndarray
    nir: np.ndarray
    swir: np.ndarray
    dt: date

    @property
    def bounds(self) -> rasterio.coords.BoundingBox:
        """Returns EPSG:4326 bbox (Lon/Lat)"""
        converted_bounds = transform_bounds(self._crs, "EPSG:4326", self._bounds.left, self._bounds.bottom, self._bounds.right, self._bounds.top)
        return rasterio.coords.BoundingBox(*converted_bounds)  # (left/lon_min, bottom/lat_min, right/lon_max, top/lat_max)

    @property
    def rgb(self) -> np.ndarray:
        rgb = np.dstack((self.red, self.green, self.blue)).astype("float32")
        return rgb

    @property
    def processed_rgb(self) -> np.ndarray:
        # Normalize each band
        red = (self.red.astype("float32") - self.red.min()) / (self.red.max() - self.red.min())
        green = (self.green.astype("float32") - self.green.min()) / (self.green.max() - self.green.min())
        blue = (self.blue.astype("float32") - self.blue.min()) / (self.blue.max() - self.blue.min())

        # Brighten
        gamma = 2.5
        red = np.power(red, 1 / gamma)
        green = np.power(green, 1 / gamma)
        blue = np.power(blue, 1 / gamma)

        return np.dstack((red, green, blue))

    @staticmethod
    def load_tif(p: Path) -> tuple[np.ndarray, rasterio.coords.BoundingBox, rasterio.crs.CRS]:
        with rasterio.open(p) as src:
            return src.read(1), src.bounds, src.crs

    @staticmethod
    def from_scene_id(scene_id: str) -> "SentinelScene":
        # Figure out datetime
        date_str = scene_id.split("_")[2]
        yyyy = int(date_str[:4])
        mm = int(date_str[4:6])
        dd = int(date_str[6:])
        dt = date(yyyy, mm, dd)

        # Load each .tif
        p = SENTINEL_SCENES_FOLDERPATH / scene_id
        red, bounds, crs = SentinelScene.load_tif(p / f"{p.name}_red.tif")
        green, _, _ = SentinelScene.load_tif(p / f"{p.name}_green.tif")
        blue, _, _ = SentinelScene.load_tif(p / f"{p.name}_blue.tif")
        nir, _, _ = SentinelScene.load_tif(p / f"{p.name}_nir.tif")
        swir, _, _ = SentinelScene.load_tif(p / f"{p.name}_swir22.tif")

        return SentinelScene(dt=dt, _bounds=bounds, _crs=crs, scene_id=scene_id, red=red, green=green, blue=blue, nir=nir, swir=swir)
