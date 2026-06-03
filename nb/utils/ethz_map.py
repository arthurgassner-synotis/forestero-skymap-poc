from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.transform
from loguru import logger
from tqdm import tqdm


@dataclass
class ETHZMap:
    profile: rasterio.profiles.Profile
    map: np.ndarray

    @staticmethod
    def from_tif(tif_filepath: Path, dtype=np.uint8) -> "ETHZMap":
        """Load from a .tif"""

        logger.info(f"Loading {tif_filepath}...")
        with rasterio.open(tif_filepath) as src:
            logger.info("Boundaries of the map")
            logger.info(f"Lon: {src.bounds.left:.2f}° --> {src.bounds.right:.2f}°")
            logger.info(f"Lat: {src.bounds.bottom:.2f}° --> {src.bounds.top:.2f}°")

            # Allocate empty array
            data = np.empty(src.shape, dtype=dtype)

            # Iterate through the file block-by-block to avoid RAM overload
            n_window = len(list(src.block_windows(bidx=1)))
            for _, window in tqdm(src.block_windows(bidx=1), total=n_window):
                chunk = src.read(indexes=1, window=window, masked=True)
                data[window.toslices()] = chunk

            # Profile
            profile = src.profile

        logger.info(f"Memory used by array: {data.nbytes / (1024**3):.2f} GB")

        return ETHZMap(profile=profile, map=data)

    def plot(self, scale_factor: int = 50) -> None:
        """Plot the full map"""
        plt.imshow(self.map[::scale_factor, ::scale_factor], vmin=0, vmax=100, cmap="viridis")
        plt.title(f"ETHZ map (scale: {scale_factor}x)")
        plt.colorbar(label="Map Values")

    def plot_around_lat_lons(self, lat_lons: list[tuple[float, float]], padding_px: int = 2, scale_factor: int = 1) -> None:
        """Plot around the provided lat,lon."""

        if not lat_lons:
            logger.warning("No locations provided to plot.")
            return

        rows, cols = [], []
        affine_transform = self.profile["transform"]
        for lat, lon in lat_lons:
            row, col = rasterio.transform.rowcol(affine_transform, lon, lat)
            rows.append(row)
            cols.append(col)

        # Figure out bbox
        min_row = min(rows) - padding_px
        max_row = max(rows) + padding_px
        min_col = min(cols) - padding_px
        max_col = max(cols) + padding_px

        # 4. Clamp pixel boundaries to valid array dimensions to avoid IndexError
        height, width = self.map.shape
        start_row = max(0, min_row)
        end_row = min(height, max_row)
        start_col = max(0, min_col)
        end_col = min(width, max_col)

        # Ensure calculated window actually captures pixels
        if start_row == end_row or start_col == end_col:
            logger.error("The calculated bounding box falls outside the raster canvas or is too small.")
            return

        # Convert the final cropped pixel corners back to Lon/Lat geographic coordinates
        # xy() takes (transform, row, col) and returns (X=lon, Y=lat)
        # We find the geo-coordinates for the actual clamped corners of our slice
        left_lon, top_lat = rasterio.transform.xy(affine_transform, start_row, start_col)
        right_lon, bottom_lat = rasterio.transform.xy(affine_transform, end_row, end_col)

        cropped_map = self.map[start_row:end_row:scale_factor, start_col:end_col:scale_factor]

        plt.figure(figsize=(10, 6))

        # Extent sets up the axis ticks using geographic limits: [left, right, bottom, top]
        extent = [left_lon, right_lon, bottom_lat, top_lat]

        plt.imshow(cropped_map, extent=extent, cmap="viridis", vmin=0, vmax=100)
        plt.xlabel("Longitude (°)")
        plt.ylabel("Latitude (°)")
        plt.title(f"ETHZ Map Crop Around {len(lat_lons)} Locations (Padding: {padding_px}px, Scale: {scale_factor}x)")
        plt.colorbar(label="Map Values")

        # Plot the target points on top of the image to make sure they are centered
        lons = [pt[1] for pt in lat_lons]
        lats = [pt[0] for pt in lat_lons]
        plt.scatter(lons, lats, color="red", marker="x", s=50, label="Target Locations")
        plt.legend()
        plt.show()
