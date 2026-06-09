import concurrent.futures
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
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
    def _load_from_tif(tif_path: Path, lonlats_wgs84: list[tuple[float, float]], padding_m: float = 0.0) -> np.ndarray:
        """Loads a subset of a raster around a list of WGS84 coordinates, with optional metric padding.

        Args:
            tif_path: Path to the raster file.
            lonlats_wgs84: List of (lon, lat) tuples.
            padding_m: Padding to add in all directions, in meters. Can be 0.

        Returns:
            data: The cropped numpy array.
        """
        # Extract the bounding box from the list of coordinates
        lons = [c[0] for c in lonlats_wgs84]
        lats = [c[1] for c in lonlats_wgs84]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        with rasterio.open(tif_path) as src:
            # Project the WGS84 bbox into the raster's native CRS
            minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat)

            # Only process padding and enforce metric CRS if padding > 0
            if padding_m > 0:
                unit = src.crs.linear_units
                if unit not in ["metre", "meter"]:
                    raise ValueError(f"Cannot apply padding in meters. Raster CRS unit is '{unit}'.")

                minx -= padding_m
                miny -= padding_m
                maxx += padding_m
                maxy += padding_m

            # Convert the spatial bounding box into a pixel/array Window
            window = window_from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            window = window.round_offsets().round_lengths()
            window = Window(col_off=window.col_off, row_off=window.row_off, width=max(1, window.width), height=max(1, window.height))

            # Read the data within that window
            # boundless=True pads the array with fill_values if padded bbox extends beyond edges.
            data = src.read(1, window=window, boundless=True, fill_value=src.nodata)

        return data

    @staticmethod
    def load_from_s2item(s2item: S2Item, bbox_wgs84: tuple[float, float, float, float], padding_m: float) -> "CroppedSnapshot":
        # Load each raster in their .tif
        p = SENTINEL_SCENES_FOLDERPATH / s2item.id
        lonlats_wgs84 = [(bbox_wgs84[0], bbox_wgs84[1]), (bbox_wgs84[2], bbox_wgs84[3])]

        band_names = ["red", "green", "blue", "rededge1", "nir", "swir22"]
        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_to_band = {
                executor.submit(CroppedSnapshot._load_from_tif, p / f"{p.name}_{band}.tif", lonlats_wgs84, padding_m): band for band in band_names
            }

            # Retrieve the results as they finish
            for future in concurrent.futures.as_completed(future_to_band):
                band = future_to_band[future]
                try:
                    results[band] = future.result()
                except Exception as exc:
                    print(f"Loading {band} generated an exception: {exc}")
                    raise

        # Extract the results (ready to be passed into your class constructor)
        red = results["red"]
        green = results["green"]
        blue = results["blue"]
        red_edge = results["rededge1"]
        nir = results["nir"]
        swir = results["swir22"]

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

    @property
    def features(self) -> np.ndarray:
        ndvi = np.max(self.ndvi)
        tci = np.max(self.tci)

        return np.array([tci, ndvi])
