from dataclasses import dataclass
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from loguru import logger
from matplotlib import pyplot as plt
from rasterio.warp import transform_bounds
from scipy.ndimage import zoom
from shapely import Polygon

from .constants import SENTINEL_SCENES_FOLDERPATH


@dataclass
class SentinelScene:
    scene_id: str
    rgbns: np.ndarray  # RGB NIR SWIR
    dt: date
    crs: rasterio.CRS = rasterio.crs.CRS.from_epsg(4326)

    @property
    def rgb(self) -> np.ndarray:
        return self.rgbns[:3, :, :]

    @property
    def processed_rgb(self) -> np.ndarray:
        # Normalize each band
        red = (self.rgbns[0] - self.rgbns[0].min()) / (self.rgbns[0].max() - self.rgbns[0].min())
        green = (self.rgbns[1] - self.rgbns[1].min()) / (self.rgbns[1].max() - self.rgbns[1].min())
        blue = (self.rgbns[2] - self.rgbns[2].min()) / (self.rgbns[2].max() - self.rgbns[2].min())

        # Brighten
        gamma = 2.5
        red = np.power(red, 1 / gamma)
        green = np.power(green, 1 / gamma)
        blue = np.power(blue, 1 / gamma)

        return np.dstack((red, green, blue))

    @property
    def nir(self) -> np.ndarray:
        return self.rgbns[3]

    @property
    def swir(self) -> np.ndarray:
        return self.rgbns[4]

    def plot(self, polygon: Polygon | None = None, padding_m: int = 100) -> None:
        sentinel_scene = self
        if polygon is not None:
            sentinel_scene = self.crop(polygon.bounds, padding_m)

        # Figure out ticks
        bounds = sentinel_scene._bounds
        extent = (0, bounds.right - bounds.left, 0, bounds.top - bounds.bottom)

        # NOTE: extent lets matplotlib handles the tick logic, supplying the bounds
        plt.imshow(sentinel_scene.processed_rgb, extent=extent)

        if polygon:
            gs = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(self._crs)
            gs = gs.translate(xoff=-bounds.left, yoff=-bounds.bottom)  # So the gs aligns with the 0-based extent
            gs.plot(ax=plt.gca(), color="red", alpha=0.8)

        plt.title(f"Processed RGB \n {sentinel_scene.dt} \n {sentinel_scene.scene_id}")
        plt.xlabel("m")
        plt.ylabel("m")

    def crop(self, bbox: tuple[float, float, float, float], padding_m: int = 0) -> "SentinelScene":
        """Crop the scene using an EPSG:4326 bounding box (min_lon, min_lat, max_lon, max_lat)"""

        min_lon, min_lat, max_lon, max_lat = bbox

        # EPSG:4326 -> self._crs conversion
        n_left, n_bottom, n_right, n_top = transform_bounds("EPSG:4326", self._crs, min_lon, min_lat, max_lon, max_lat)

        # Apply padding, assuming self._crs uses meters
        n_left -= padding_m
        n_bottom -= padding_m
        n_right += padding_m
        n_top += padding_m

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
        cropped_rgbns = self.rgbns[:, row_min:row_max, col_min:col_max]

        # Update ._bounds
        new_left = self._bounds.left + (col_min * x_res)
        new_right = self._bounds.left + (col_max * x_res)
        new_top = self._bounds.top - (row_min * y_res)
        new_bottom = self._bounds.top - (row_max * y_res)
        bounds = rasterio.coords.BoundingBox(new_left, new_bottom, new_right, new_top)

        return SentinelScene(_bounds=bounds, _crs=self._crs, scene_id=self.scene_id, rgbns=cropped_rgbns, dt=self.dt)

    @staticmethod
    def load_raster(p: Path) -> np.ndarray:
        with rasterio.open(p) as src:
            return src.read(1)

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
    def from_scene_id(scene_id: str) -> "SentinelScene":
        # Figure out datetime
        date_str = scene_id.split("_")[2]
        yyyy = int(date_str[:4])
        mm = int(date_str[4:6])
        dd = int(date_str[6:])
        dt = date(yyyy, mm, dd)

        # Load each raster in their .tif
        p = SENTINEL_SCENES_FOLDERPATH / scene_id
        red = SentinelScene.load_raster(p / f"{p.name}_red.tif")
        green = SentinelScene.load_raster(p / f"{p.name}_green.tif")
        blue = SentinelScene.load_raster(p / f"{p.name}_blue.tif")
        nir = SentinelScene.load_raster(p / f"{p.name}_nir.tif")
        swir = SentinelScene.load_raster(p / f"{p.name}_swir22.tif")
        swir = zoom(swir, zoom=2, order=1)  # Zoom into SWIR, since one pix == 20m x 20m

        rgbns = np.dstack((red, green, blue, nir, swir)).astype("float32")

        # Load CRS and bounds
        original_bounds, original_crs = SentinelScene._load_bounds_and_crs(scene_id)
        bounds_4326 = transform_bounds(original_crs, "EPSG:4326", *original_bounds)

        return SentinelScene(bounds=rasterio.coords.BoundingBox(*bounds_4326), dt=dt, scene_id=scene_id, rgbns=rgbns)
