from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import zoom
from shapely import Polygon

from .constants import ETHZ_COCOA_MAP_FILEPATH, SENTINEL_SCENES_FOLDERPATH

PX_TO_M = 10


@dataclass
class SentinelScene:
    scene_id: str
    rgb_re_nir_swir: np.ndarray  # RGB NIR SWIR
    dt: date
    bounds: rasterio.coords.BoundingBox
    crs: rasterio.CRS = rasterio.crs.CRS.from_epsg(4326)

    @property
    def array_size(self) -> str:
        """Size in Giga Bytes"""
        return f"{self.rgb_re_nir_swir.nbytes / (1024**3):.2f} GB"

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
        red = (self.red - self.red.min()) / (self.red.max() - self.red.min())
        green = (self.green - self.green.min()) / (self.green.max() - self.green.min())
        blue = (self.blue - self.blue.min()) / (self.blue.max() - self.blue.min())

        # Brighten
        gamma = 2.5  # Hand-picked so that it looks nice
        red = np.power(red, 1 / gamma)
        green = np.power(green, 1 / gamma)
        blue = np.power(blue, 1 / gamma)

        return np.dstack((red, green, blue))

    def compute_Xy(self) -> tuple[pd.DataFrame, pd.Series]:
        """Extract features & label for each pixel in this scene."""
        Xy = pd.DataFrame(
            {
                "tci": self.tci.flatten(),
                "ndvi": self.ndvi.flatten(),
                "nir": self.nir.flatten(),
                "swir": self.swir.flatten(),
                "y_proba": self.ethz_array.flatten(),
            }
        )
        Xy = Xy.dropna().reset_index(drop=True)
        return Xy[["tci", "ndvi", "nir", "swir"]], Xy[["y_proba"]]

    @staticmethod
    def _plot(img: np.ndarray, max_resolution_px: int, ax: plt.axes._axes.Axes) -> None:
        """Plot a downscaled version of the img."""
        height, width = img.shape[:2]

        # Cheap downscale to match max resolution
        if height >= max_resolution_px:
            stride = height // max_resolution_px
            img = img[::stride, ::stride]

        # Define what the img's side correspond to in "world coords"
        height_m, width_m = height * PX_TO_M, width * PX_TO_M
        extent = [0, width_m, height_m, 0]  # left, right, bottom, top

        ax.imshow(img, extent=extent)

        ax.set_xlabel("m")
        ax.set_ylabel("m")

    def plot(self, max_resolution_px: int = 1_000, plot_scale: float = 1.0) -> None:
        """Plot RGB, Red-Edge, NIR & SWIR"""
        _, axes = plt.subplots(1, 4, figsize=(10 * plot_scale, 3 * plot_scale), sharey=True)
        plt.suptitle(f"{self.dt} \n {self.scene_id}")

        # Plot RGB
        axes[0].set_title("Processed RGB")
        SentinelScene._plot(self.processed_rgb, max_resolution_px=max_resolution_px, ax=axes[0])

        # Plot Red Edge + NIR + SWIR
        axes[1].set_title("Red Edge")
        SentinelScene._plot(self.red_edge, max_resolution_px=max_resolution_px, ax=axes[1])

        axes[2].set_title("Near Infrared (NIR)")
        SentinelScene._plot(self.nir, max_resolution_px=max_resolution_px, ax=axes[2])

        axes[3].set_title("Short-Wave Infrared (SWIR)")
        SentinelScene._plot(self.nir, max_resolution_px=max_resolution_px, ax=axes[3])

        plt.tight_layout()

    def plot_bbox(self, polygon: Polygon | None = None, padding_m: int = 100, plot_ethz: bool = False) -> None:
        # Figure out bounds
        min_lon, min_lat, max_lon, max_lat = self.bounds
        mid_lon = min_lon + (min_lon + max_lon) / 2
        mid_lat = min_lat + (min_lat + max_lat) / 2
        bounds = (mid_lon, mid_lat, mid_lon, mid_lat)
        if polygon is not None:
            bounds = polygon.bounds

        sentinel_scene = self.crop(bounds, padding_m)

        # Figure out ticks
        extent = (0, self.bounds.right - self.bounds.left, 0, self.bounds.top - self.bounds.bottom)

        # NOTE: extent lets matplotlib handles the tick logic, supplying the bounds
        plt.imshow(sentinel_scene.processed_rgb, extent=extent)

        if polygon:
            gs = gpd.GeoSeries([polygon], crs="EPSG:4326")
            gs = gs.translate(xoff=-self.bounds.left, yoff=-self.bounds.bottom)  # So the gs aligns with the 0-based extent
            gs.plot(ax=plt.gca(), color="red", alpha=0.8)

        if plot_ethz:
            plt.imshow(sentinel_scene.ethz_array, ax=plt.gca(), color="green", alpha=self.ethz_array)

        plt.title(f"Processed RGB \n {sentinel_scene.dt} \n {sentinel_scene.scene_id}")
        plt.xlabel("m")
        plt.ylabel("m")

    def crop(self, bbox: tuple[float, float, float, float], padding_m: int = 0) -> "SentinelScene":
        """Crop the scene using an EPSG:4326 bounding box (min_lon, min_lat, max_lon, max_lat)"""

        min_lon, min_lat, max_lon, max_lat = bbox

        # EPSG:4326 -> ? conversion
        n_left, n_bottom, n_right, n_top = transform_bounds(self.crs, self.c, min_lon, min_lat, max_lon, max_lat)
        # FIXME
        raise NotImplementedError()

        # Apply padding, assuming self._crs uses meters
        n_left -= padding_m
        n_bottom -= padding_m
        n_right += padding_m
        n_top += padding_m

        # Figure out pixel resolution
        height, width, _ = self.rgb_re_nir_swir.shape
        x_res = (self.bounds.right - self.bounds.left) / width
        y_res = (self.bounds.top - self.bounds.bottom) / height

        # Spatial coordinates -> pixel indices conversion
        # (row 0 is at bounds.top, col 0 is at bounds.left)
        col_min = int(max(0, (n_left - self.bounds.left) / x_res))
        col_max = int(min(width, (n_right - self.bounds.left) / x_res))
        row_min = int(max(0, (self.bounds.top - n_top) / y_res))
        row_max = int(min(height, (self.bounds.top - n_bottom) / y_res))

        # Ensure cropping makes sense
        if col_min >= col_max or row_min >= row_max:
            raise ValueError("The provided bounding box does not intersect this scene.")

        # Crop
        cropped_rgb_re_nir_swir = self.rgb_re_nir_swir[row_min:row_max, col_min:col_max, :]

        # Update .bounds
        new_left = self.bounds.left + (col_min * x_res)
        new_right = self.bounds.left + (col_max * x_res)
        new_top = self.bounds.top - (row_min * y_res)
        new_bottom = self.bounds.top - (row_max * y_res)
        bounds = rasterio.coords.BoundingBox(new_left, new_bottom, new_right, new_top)

        return SentinelScene(bounds=bounds, scene_id=self.scene_id, rgb_re_nir_swir=cropped_rgb_re_nir_swir, dt=self.dt)

    @staticmethod
    def load_raster(p: Path) -> np.ndarray:
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
    def from_scene_id(scene_id: str) -> "SentinelScene":
        """Load a SentinelScene from the data available at SENTINEL_SCENES_FOLDERPATH/"""

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
        red_edge = SentinelScene.load_raster(p / f"{p.name}_rededge1.tif")
        nir = SentinelScene.load_raster(p / f"{p.name}_nir.tif")
        swir = SentinelScene.load_raster(p / f"{p.name}_swir22.tif")

        red_edge = zoom(red_edge, zoom=2, order=1)  # Zoom into red edge, since one pix == 20m x 20m
        swir = zoom(swir, zoom=2, order=1)  # Zoom into SWIR, since one pix == 20m x 20m

        rgb_re_nir_swir = np.dstack((red, green, blue, red_edge, nir, swir)).astype("float32")

        # Load CRS and bounds
        original_bounds, original_crs = SentinelScene._load_bounds_and_crs(scene_id)
        bounds_4326 = transform_bounds(original_crs, "EPSG:4326", *original_bounds)

        return SentinelScene(bounds=rasterio.coords.BoundingBox(*bounds_4326), dt=dt, scene_id=scene_id, rgb_re_nir_swir=rgb_re_nir_swir)

    @cached_property
    def ethz_array(self) -> None:
        """Load an external ETHZ raster and dynamically aligns/resamples it to perfectly match the bounds, CRS, and resolution."""
        height, width, _ = self.rgb_re_nir_swir.shape

        # Calculate affine transform from this scene's bounds -> this scene's pixels
        dst_transform = from_bounds(self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top, width, height)

        # Prepare an empty numpy array to hold the reprojected data
        aligned_ethz = np.empty((height, width), dtype="float32")

        with rasterio.open(ETHZ_COCOA_MAP_FILEPATH) as src:
            # Figure out what bounding box to read from the source ETHZ raster
            src_left, src_bottom, src_right, src_top = transform_bounds(self.crs, src.crs, *self.bounds)

            # Calculate the pixel window for that bounding box
            window = window_from_bounds(src_left, src_bottom, src_right, src_top, transform=src.transform)

            src_array = src.read(1, window=window, boundless=True)
            src_window_transform = src.window_transform(window)

            reproject(
                source=src_array,
                destination=aligned_ethz,
                src_transform=src_window_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=self.crs,
                resampling=Resampling.bilinear,
            )

            # Only keep probas
            aligned_ethz[aligned_ethz <= 0.0] = np.nan

        return aligned_ethz
