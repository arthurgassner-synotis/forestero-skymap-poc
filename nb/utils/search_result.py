from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import requests
from pystac.item import Item as PySTACItem
from pystac_client.client import Client as PySTACClient
from tqdm import tqdm

from utils.constants import SENTINEL_BANDS, SENTINEL_SCENES_FOLDERPATH


@dataclass
class SearchResult:
    bbox: tuple[float, float, float, float]
    scenes: list[PySTACItem]

    @property
    def dts(self) -> list[date]:
        dts = []
        for scene in self.scenes:
            # Figure out datetime
            date_str = scene.id.split("_")[2]
            yyyy = int(date_str[:4])
            mm = int(date_str[4:6])
            dd = int(date_str[6:])
            dt = date(yyyy, mm, dd)

            dts.append(dt)

        return dts

    def keep(self, min_date: date = date(1, 1, 1), max_date: date = date(3000, 1, 1)) -> None:
        kept_scenes = []
        for dt, scene in zip(self.dts, self.scenes):
            if min_date <= dt <= max_date:
                kept_scenes.append(scene)

        self.scenes = kept_scenes

    def keep_least_cloudy(self) -> None:
        scene_idx = np.argmin([e.properties["eo:cloud_cover"] for e in self.scenes])
        self.scenes = [self.scenes[scene_idx]]

    @staticmethod
    def _search_stac(dt: str, bbox: tuple[float, float, float, float], client: PySTACClient) -> tuple[str, list[PySTACItem]]:
        """Helper function to run a single search."""
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=dt,  # e.g. "2020-01-01/2021-01-30"
            query={"eo:cloud_cover": {"lt": 20}},
        )
        scenes = list(search.items())
        return dt, scenes

    @staticmethod
    def from_search(dts: list[str], bbox: tuple[float, float, float, float], client: PySTACClient) -> "SearchResult":
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {executor.submit(SearchResult._search_stac, dt, bbox, client): dt for dt in dts}

        # Fetch futures
        agg_scenes = []
        for future in tqdm(as_completed(futures), total=len(dts), desc="Fetching scenes"):
            _, scenes = future.result()
            agg_scenes.extend(scenes)

        return SearchResult(bbox=bbox, scenes=agg_scenes)

    @staticmethod
    def download_file(url: str, filepath: Path) -> None:
        """Downloads a file using a .part extension to prevent corruption if interrupted."""
        if filepath.exists():
            print(f"File {filepath} already exists. Skipping...")
            return

        part_filepath = filepath.with_suffix(filepath.suffix + ".part")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            part_filepath.open("wb") as f,
            tqdm(desc=filepath.name, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024, leave=False) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)

        part_filepath.rename(filepath)

    def download(self) -> None:
        print(f"Downloading {len(self.scenes)} scenes ...")

        for scene in self.scenes:
            scene_folder = SENTINEL_SCENES_FOLDERPATH / scene.id
            scene_folder.mkdir(parents=True, exist_ok=True)

            for band in SENTINEL_BANDS:
                if band in scene.assets:
                    download_url = scene.assets[band].href
                    filename = f"{scene.id}_{band}.tif"
                    filepath = scene_folder / filename

                    try:
                        SearchResult.download_file(download_url, filepath)
                    except Exception as e:
                        print(f"Error downloading {band}: {e}")
