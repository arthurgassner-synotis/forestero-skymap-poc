from dataclasses import dataclass

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

from .constants import GHANA_GDF


@dataclass
class Tree:
    point: Point

    def plot(self) -> None:
        f, axes = plt.subplots(1, 2, figsize=(6, 5))

        # Plot full map
        GHANA_GDF.plot(ax=axes[0], color="whitesmoke", edgecolor="black", linewidth=1.5)

        # Plot single tree
        gdf = gpd.GeoDataFrame(geometry=[self.point])
        gdf.plot(ax=axes[0])

        # Plot zoomed-in version
        PADDING = 0.00001
        lon, lat = self.point.coords[0]
        min_lon, min_lat = lon - PADDING, lat - PADDING
        max_lon, max_lat = lon + PADDING, lat + PADDING

        GHANA_GDF.plot(ax=axes[1], color="whitesmoke", edgecolor="black", linewidth=1.5)
        gdf = gpd.GeoDataFrame(geometry=[self.point])
        gdf.plot(ax=axes[1])

        axes[1].set_xlim(min_lon, max_lon)
        axes[1].set_ylim(min_lat, max_lat)
