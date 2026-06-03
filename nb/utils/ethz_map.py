from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
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
