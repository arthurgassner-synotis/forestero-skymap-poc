from dataclasses import dataclass

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from shapely.geometry import MultiPoint, Polygon

from .constants import GHANA
from .tree import Tree


@dataclass
class Site:
    trees: list[Tree]
    name: str
    region: str

    @property
    def polygon(self) -> Polygon:
        return MultiPoint([e.point for e in self.trees]).convex_hull

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(geometry=[e.point for e in self.trees], crs="EPSG:4326")

    @property
    def gs(self) -> gpd.GeoDataFrame:
        return gpd.GeoSeries([self.polygon], crs="EPSG:4326")

    @property
    def area_m2(self) -> float:
        local_utm = self.gs.estimate_utm_crs()
        return self.gs.to_crs(local_utm).area.iloc[0]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """min-lon, min-lat, max-lon, max-lat"""
        return tuple(self.polygon.bounds)

    def plot(self) -> None:
        f, axes = plt.subplots(1, 2, figsize=(5, 6))

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.5f}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.5f}"))

        # Plot full map
        GHANA.plot(ax=axes[0], color="whitesmoke", edgecolor="black", linewidth=1.5)
        self.gdf.plot(ax=axes[0])
        axes[0].set_title(f"Plantation {self.name}\n (Region: {self.region})")

        # Plot zoomed-in version
        GHANA.plot(ax=axes[1], color="whitesmoke", edgecolor="black", linewidth=1.5)
        self.gs.plot(ax=axes[1], color="lightgreen", alpha=0.5)
        self.gdf.plot(ax=axes[1])

        min_lon, min_lat, max_lon, max_lat = self.polygon.bounds
        PADDING = 0.00001
        min_lon -= PADDING
        min_lat -= PADDING
        max_lon += PADDING
        max_lat += PADDING

        axes[1].set_xlim(min_lon, max_lon)
        axes[1].set_ylim(min_lat, max_lat)
        axes[1].set_title(f"{len(self.trees)} trees\n spanning {self.area_m2:.1f} m2")

        plt.tight_layout()
        plt.show()
