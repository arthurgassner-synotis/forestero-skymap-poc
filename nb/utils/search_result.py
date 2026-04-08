from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from pystac.item import Item
from pystac_client.client import Client as PySTACClient
from tqdm import tqdm


@dataclass
class SearchResult:
    dt: str
    bbox: tuple[float, float, float, float]
    scenes: list[Item]

    @staticmethod
    def _search_stac(dt: str, bbox: tuple[float, float, float, float], client: PySTACClient) -> tuple[str, list[Item]]:
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
    def from_search(dts: list[str], bbox: tuple[float, float, float, float], client: PySTACClient) -> list["SearchResult"]:
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {executor.submit(SearchResult._search_stac, dt, bbox, client): dt for dt in dts}

        dt_to_scenes = {}
        for future in tqdm(as_completed(futures), total=len(dts), desc="Fetching scenes"):
            dt, scenes = future.result()
            dt_to_scenes[dt] = scenes

        return [SearchResult(dt=dt, bbox=bbox, scenes=scenes) for dt, scenes in dt_to_scenes.items()]
