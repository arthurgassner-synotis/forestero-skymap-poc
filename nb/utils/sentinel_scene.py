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

    def crop(self, bbox: tuple[float, float, float, float]) -> None:
        """Crop the scene in-place using an EPSG:4326 bounding box.
        bbox format: (min_lon, min_lat, max_lon, max_lat)
        """

        min_lon, min_lat, max_lon, max_lat = bbox

        # EPSG:4326 -> self._crs conversion
        n_left, n_bottom, n_right, n_top = transform_bounds("EPSG:4326", self._crs, min_lon, min_lat, max_lon, max_lat)

        # Figure out pixel resolution
        height, width = self.red.shape
        x_res = (self._bounds.right - self._bounds.left) / width
        y_res = (self._bounds.top - self._bounds.bottom) / height

        # Spatial coordinates -> pixel indices conversion
        # (row 0 is at _bounds.top, col 0 is at _bounds.left)
        col_min = int(max(0, (n_left - self._bounds.left) / x_res))
        col_max = int(min(width, (n_right - self._bounds.left) / x_res))
        row_min = int(max(0, (self._bounds.top - n_top) / y_res))
        row_max = int(min(height, (self._bounds.top - n_bottom) / y_res))

        # Ensure cropping makes sense
        if col_min >= col_max or row_min >= row_max:
            raise ValueError("The provided bounding box does not intersect this scene.")

        # Crop
        self.red = self.red[row_min:row_max, col_min:col_max]
        self.green = self.green[row_min:row_max, col_min:col_max]
        self.blue = self.blue[row_min:row_max, col_min:col_max]
        self.nir = self.nir[row_min:row_max, col_min:col_max]
        self.swir = self.swir[row_min:row_max, col_min:col_max]

        # Update ._bounds
        new_left = self._bounds.left + (col_min * x_res)
        new_right = self._bounds.left + (col_max * x_res)
        new_top = self._bounds.top - (row_min * y_res)
        new_bottom = self._bounds.top - (row_max * y_res)
        self._bounds = rasterio.coords.BoundingBox(new_left, new_bottom, new_right, new_top)

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
