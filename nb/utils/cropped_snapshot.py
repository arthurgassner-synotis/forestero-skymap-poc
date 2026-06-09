from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import zoom

from .constants import SENTINEL_SCENES_FOLDERPATH
from .s2item import S2Item


@dataclass
class CroppedSnapshot:
    s2id: str  # Sentinel2 ID, i.e. S2Item's id
    rgb_re_nir_swir: np.ndarray  # RGB RE NIR SWIR
    bounds: rasterio.coords.BoundingBox
    crs: rasterio.CRS

    @property
    def red(self) -> np.ndarray:
        return self.rgb_re_nir_swir[:, :, 0]

    @property
    def green(self) -> np.ndarray:
        return self.rgb_re_nir_swir[:, :, 1]

    @property
    def blue(self) -> np.ndarray:
        return self.rgb_re_nir_swir[:, :, 2]

    @property
    def red_edge(self) -> np.ndarray:
        return self.rgb_re_nir_swir[:, :, 3]

    @property
    def nir(self) -> np.ndarray:
        return self.rgb_re_nir_swir[:, :, 4]

    @property
    def swir(self) -> np.ndarray:
        return self.rgb_re_nir_swir[:, :, 5]

    @property
    def ndvi(self) -> np.ndarray:
        """Normalized Difference Vegetation Index"""
        return (self.nir - self.red) / (self.nir + self.red)

    @property
    def tci(self) -> np.ndarray:
        """Triangular Chlorophyll Index"""
        return 1.2 * (self.red_edge - self.green) - 1.5 * (self.red - self.green) * np.sqrt(self.red_edge / self.red)

    @cached_property
    def processed_rgb(self) -> np.ndarray:
        """Normalized & Gamma-corrected RGB."""
        # Normalize each band
        red = (self.red - np.nanmin(self.red)) / (np.nanmax(self.red) - np.nanmin(self.red))
        green = (self.green - np.nanmin(self.green)) / (np.nanmax(self.green) - np.nanmin(self.green))
        blue = (self.blue - np.nanmin(self.blue)) / (np.nanmax(self.blue) - np.nanmin(self.blue))

        # Brighten
        gamma = 2.5  # Hand-picked so that it looks nice
        red = np.power(red, 1 / gamma)
        green = np.power(green, 1 / gamma)
        blue = np.power(blue, 1 / gamma)

        return np.dstack((red, green, blue))

    @staticmethod
    def _load_from_tif(tif_path: Path, bbox_wgs84: tuple[float, float, float, float], padding_m: float) -> np.ndarray:
        """Loads a subset of a raster around a WGS84 bbox, with added padding in meters.

        Args:
            tif_path: Path to the raster file.
            bbox_wgs84: Tuple of (min_lon, min_lat, max_lon, max_lat).
            padding_m: Padding to add in all directions, in meters.

        Returns:
            data: The cropped numpy array.
            transform: The affine transform for the new cropped image.
        """
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84

        with rasterio.open(tif_path) as src:
            # Project the WGS84 bbox into the raster's native CRS
            minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat)

            # Add padding in meters
            unit = src.crs.linear_units
            if unit not in ["metre", "meter"]:
                raise ValueError()

            minx -= padding_m
            miny -= padding_m
            maxx += padding_m
            maxy += padding_m

            # Convert the spatial bounding box into a pixel/array Window
            window = window_from_bounds(minx, miny, maxx, maxy, transform=src.transform)

            # Read the data within that window
            # boundless=True pads the array with fill_values if padded bbox extends beyond the actual edges of the raster image.
            data = src.read(1, window=window, boundless=True, fill_value=src.nodata)

        return data

    @staticmethod
    def load_from_s2item(s2item: S2Item, bbox_wgs84: tuple[float, float, float, float], padding_m: float) -> "CroppedSnapshot":
        # Load each raster in their .tif
        p = SENTINEL_SCENES_FOLDERPATH / s2item.id
        red = CroppedSnapshot._load_from_tif(p / f"{p.name}_red.tif", bbox_wgs84, padding_m)
        green = CroppedSnapshot._load_from_tif(p / f"{p.name}_green.tif", bbox_wgs84, padding_m)
        blue = CroppedSnapshot._load_from_tif(p / f"{p.name}_blue.tif", bbox_wgs84, padding_m)
        red_edge = CroppedSnapshot._load_from_tif(p / f"{p.name}_rededge1.tif", bbox_wgs84, padding_m)
        nir = CroppedSnapshot._load_from_tif(p / f"{p.name}_nir.tif", bbox_wgs84, padding_m)
        swir = CroppedSnapshot._load_from_tif(p / f"{p.name}_swir22.tif", bbox_wgs84, padding_m)

        # Calculate exact zoom factors to match the 10m 'red' band shape perfectly
        re_zoom_y = red.shape[0] / red_edge.shape[0]
        re_zoom_x = red.shape[1] / red_edge.shape[1]
        red_edge = zoom(red_edge, zoom=(re_zoom_y, re_zoom_x), order=1)

        swir_zoom_y = red.shape[0] / swir.shape[0]
        swir_zoom_x = red.shape[1] / swir.shape[1]
        swir = zoom(swir, zoom=(swir_zoom_y, swir_zoom_x), order=1)

        rgb_re_nir_swir = np.dstack((red, green, blue, red_edge, nir, swir)).astype("float32")

        # Figure out bounds and crs using the 10m 'red' band as the spatial reference
        with rasterio.open(p / f"{p.name}_red.tif") as src:
            crs = src.crs
            min_lon, min_lat, max_lon, max_lat = bbox_wgs84

            # Recalculate padded coordinates
            minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, min_lon, min_lat, max_lon, max_lat)
            minx -= padding_m
            miny -= padding_m
            maxx += padding_m
            maxy += padding_m

            # Calculate the pixel-snapped bounding box
            window = window_from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            bounds = src.window_bounds(window)

        return CroppedSnapshot(s2id=s2item.id, rgb_re_nir_swir=rgb_re_nir_swir, bounds=bounds, crs=crs)
