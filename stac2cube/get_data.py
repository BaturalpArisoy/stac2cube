from .vector_refiner import polygon_2_bbox

import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
from pystac_client import Client as pystacclient
from odc.stac import stac_load
import planetary_computer
import os
import re
import datetime


_S2_COMMON_TO_BNUM = {
    "coastal": "B01",
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir": "B08",
    "nir08": "B8A",
    "nir09": "B09",
    "swir16": "B11",
    "swir22": "B12",
    "scl": "SCL",
    "aot": "AOT",
    "wvp": "WVP",
}
_S2_BNUM_TO_COMMON = {v: k for k, v in _S2_COMMON_TO_BNUM.items()}


def get_stac(
    mission: str,
    polygon,
    resolution: int,
    daterange: list,
    bands: list,
    max_cc: int,
    cloud_masking: bool,
    source: str = "element84",
):
    _source_aliases = {"e84": "element84", "tb": "terrabyte", "pc": "planetary_computer"}
    source = _source_aliases.get(source, source)

    catalogues = {
        "sentinel_2_l2a": {
            "element84": ("https://earth-search.aws.element84.com/v1/", "sentinel-2-l2a"),
            "terrabyte": ("https://stac.terrabyte.lrz.de/public/api/", "sentinel-2-c1-l2a"),
            "planetary_computer": ("https://planetarycomputer.microsoft.com/api/stac/v1", "sentinel-2-l2a"),
        },
        "sentinel_2_l1c": (
            "https://earth-search.aws.element84.com/v1/",
            "sentinel-2-l1c",
        ),
        "cop_dem_glo_30": (
            "https://stac.terrabyte.lrz.de/public/api/",
            "cop-dem-glo-30",
        ),
        "landsat_c2_l2": (
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            "landsat-c2-l2",
        ),
        "sentinel_1_rtc": (
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            "sentinel-1-rtc",
        ),
    }

    if resolution is not None:
        resolution = resolution
    else:
        resolutions = {
            "sentinel_2_l2a": 10,
            "sentinel_2_l1c": 10,
            "cop_dem_glo_30": None,
            "landsat_c2_l2": 30,
            "sentinel_1_rtc": 10,
        }
        resolution = resolutions[mission]

    if isinstance(polygon, list):
        bbox = polygon
    else:
        bbox = polygon_2_bbox(polygon)

    if mission == "sentinel_2_l2a":
        sources = catalogues["sentinel_2_l2a"]
        if source not in sources:
            raise ValueError(
                f"Unknown source '{source}' for sentinel_2_l2a. "
                f"Valid options: e84, tb, pc (or full names: {list(sources.keys())})."
            )
        url, collection = sources[source]
    else:
        url, collection = catalogues[mission]

    needs_pc_auth = mission in ("sentinel_1_rtc", "landsat_c2_l2") or (
        mission == "sentinel_2_l2a" and source == "planetary_computer"
    )
    if needs_pc_auth:
        catalog = pystacclient.open(url, modifier=planetary_computer.sign_inplace)
    else:
        catalog = pystacclient.open(url)

    query = {"eo:cloud_cover": {"gte": 0, "lte": max_cc}}

    if mission in ("cop_dem_glo_30", "sentinel_1_rtc"):
        query = None

    season_spec = _parse_season_daterange(daterange)

    if season_spec is None:
        items, crs, stac_mission, tiles = _catalogue_search(
            catalog, collection, bbox, daterange, query, mission
        )
    else:
        start_md, end_md, years_spec = season_spec
        years = _parse_years_spec(years_spec, mission)
        windows = _expand_season_windows(start_md, end_md, years)

        all_items = []
        crs = None
        stac_mission = None
        tiles_set = set()

        for win in windows:
            win_items, win_crs, win_stac_mission, win_tiles = _catalogue_search(
                catalog, collection, bbox, win, query, mission, allow_empty=True
            )
            if win_items:
                all_items.extend(list(win_items))
                if crs is None:
                    crs = win_crs
                    stac_mission = win_stac_mission
                if win_tiles is not None:
                    tiles_set.update(list(win_tiles))

        if len(all_items) < 1:
            raise ValueError(
                "No scenes found by the given parameters in season mode. "
                "Please check your polygon's geometry, season window or increase max cloud coverage."
            )

        items = all_items
        tiles = np.array(sorted(tiles_set)) if tiles_set else None

    if mission == "sentinel_2_l2a":
        bands = list(dict.fromkeys(_S2_BNUM_TO_COMMON.get(b, b) for b in bands))

    if cloud_masking is True:
        if mission == "sentinel_2_l2a" and "scl" not in bands:
            bands.append("scl")
        if mission == "landsat_c2_l2":
            bands.append("qa_pixel")

    band_map = _get_band_map(mission, source)
    if band_map is not None:
        bands = [band_map.get(band, band) for band in bands]

    # Pre-filter duplicate items for sentinel_2_l1c based on processing baseline
    if mission == "sentinel_2_l1c":
        from collections import defaultdict

        grouped = defaultdict(list)
        for item in items:
            # Use date string (first 10 characters) as solar day key
            date_key = item.properties.get("datetime", "")[:10]
            grouped[date_key].append(item)
        filtered_items = []
        for date_key, group in grouped.items():
            # Choose item with highest processing baseline (converted to float)
            best_item = max(
                group,
                key=lambda it: float(it.properties.get("s2:processing_baseline", "0")),
            )
            filtered_items.append(best_item)
        items = filtered_items

    stac = stac_load(
        items,
        bands=bands,
        crs=crs,
        resolution=resolution,
        resampling="bilinear",
        chunks={},
        groupby="solar_day",
        bbox=bbox,
    )

    if band_map is not None:
        reverse_band_map = {v: k for k, v in band_map.items()}
        if reverse_band_map:
            rename_dict = {
                band: reverse_band_map.get(band, band)
                for band in stac.data_vars
                if band in reverse_band_map
            }
            stac = stac.rename(rename_dict)

    if mission == "sentinel_2_l1c":
        date_list = [item.properties["datetime"] for item in items]
        processing_baseline_list = [
            item.properties["s2:processing_baseline"] for item in items
        ]
        dates = pd.to_datetime(date_list, format="mixed").to_numpy(
            dtype="datetime64[ns]"
        )
        baseline_da = xr.DataArray(
            processing_baseline_list,
            dims=["time"],
            coords={"time": dates},
            name="processing_baseline",
        )
        baseline_da_filtered = baseline_da.sel(time=baseline_da.time.isin(stac.time))
        unique_times, counts = np.unique(
            baseline_da_filtered.time.values, return_counts=True
        )
        duplicate_times = unique_times[counts > 1]
        stac = stac.sel(time=~np.isin(stac.time, duplicate_times))
        baselines = baseline_da_filtered.sel(
            time=~np.isin(baseline_da_filtered.time, duplicate_times)
        )

        return stac, baselines, tiles
    else:
        if mission == "sentinel_1_rtc":
            from datetime import datetime

            orbit_state_by_day = {}
            for item in items:
                item_date = datetime.fromisoformat(item.properties["datetime"]).date()
                if item_date not in orbit_state_by_day:
                    orbit_state_by_day[item_date] = item.properties["sat:orbit_state"]
            solar_days_in_stac = [pd.Timestamp(t).date() for t in stac.time.values]
            aligned_orbit_states = [
                orbit_state_by_day.get(day, None) for day in solar_days_in_stac
            ]
            if None in aligned_orbit_states:
                print(
                    "Warning: Some dates in the stac dataset did not have a matching orbit state."
                )
            stac = stac.assign_coords(orbit_state=("time", aligned_orbit_states))

        return stac, None, tiles


# ==========================================================
# DATE RANGE HELPERS
# ==========================================================
_MMDD_RE = re.compile(r"^\d{2}-\d{2}$")


def _is_mmdd(s: str) -> bool:
    """Return True if string is in MM-DD format and represents a valid calendar day."""
    if not isinstance(s, str) or not _MMDD_RE.match(s.strip()):
        return False
    mm, dd = map(int, s.split("-"))
    try:
        # Use a leap year to allow 02-29 in case someone needs it
        datetime.date(2000, mm, dd)
    except ValueError:
        return False
    return True


def _parse_season_daterange(daterange):
    """Detect 'season mode' daterange.

    Supported:
      1) daterange = ["MM-DD", "MM-DD"]  -> season for years="all"
      2) daterange = {"season": ["MM-DD", "MM-DD"], "years": "all" | [years] | "YYYY-YYYY" | "YYYY,YYYY"}
    """
    if isinstance(daterange, dict) and "season" in daterange:
        season = daterange.get("season")
        years = daterange.get("years", "all")
        if not isinstance(season, (list, tuple)) or len(season) != 2:
            raise ValueError(
                "Season daterange must be like {'season': ['MM-DD', 'MM-DD'], 'years': ...}."
            )
        start_md, end_md = season
        if not (_is_mmdd(str(start_md)) and _is_mmdd(str(end_md))):
            raise ValueError(
                "Season start/end must be in 'MM-DD' format (e.g., '04-01', '10-31')."
            )
        return str(start_md), str(end_md), years

    if isinstance(daterange, (list, tuple)) and len(daterange) == 2:
        a, b = daterange
        if _is_mmdd(str(a)) and _is_mmdd(str(b)):
            return str(a), str(b), "all"

    return None


def _mission_year_span(mission: str):
    """Default year span for 'years="all"' in season mode.

    Note: these are conservative defaults to avoid overly long loops.
    Users can always override via daterange dict 'years'.
    """
    current_year = datetime.date.today().year
    spans = {
        "sentinel_2_l2a": (2015, current_year),
        "sentinel_2_l1c": (2015, current_year),
        "sentinel_1_rtc": (2014, current_year),
        "landsat_c2_l2": (1982, current_year),
    }
    return spans.get(mission)


def _parse_years_spec(years_spec, mission: str):
    """Parse years spec for season mode."""
    if years_spec is None or (isinstance(years_spec, str) and years_spec.strip().lower() == "all"):
        span = _mission_year_span(mission)
        if span is None:
            raise ValueError(
                f"Season mode with years='all' is not supported for mission '{mission}'. "
                "Please specify years explicitly, e.g. {'season': ['04-01','10-31'], 'years': [2020, 2021]}."
            )
        y0, y1 = span
        return list(range(int(y0), int(y1) + 1))

    if isinstance(years_spec, int):
        return [int(years_spec)]

    if isinstance(years_spec, (list, tuple, set)):
        years = sorted({int(y) for y in years_spec})
        if not years:
            raise ValueError("Years list is empty.")
        return years

    if isinstance(years_spec, str):
        s = years_spec.strip()
        m = re.match(r"^(\d{4})\s*-\s*(\d{4})$", s)
        if m:
            a, b = map(int, m.groups())
            if b < a:
                a, b = b, a
            return list(range(a, b + 1))

        if re.match(r"^\d{4}(?:\s*,\s*\d{4})+$", s):
            return sorted({int(x.strip()) for x in s.split(",")})

    raise ValueError(
        "Invalid years specification. Use 'all', [2019,2020], '2019-2024', or '2019,2021,2023'."
    )


def _expand_season_windows(start_md: str, end_md: str, years):
    """Expand a season (MM-DD .. MM-DD) into per-year concrete ISO windows.

    If start_md is later than end_md (e.g. 11-01 .. 03-31), season crosses year boundary.
    """
    sm, sd = map(int, start_md.split("-"))
    em, ed = map(int, end_md.split("-"))
    crosses_year = (sm, sd) > (em, ed)

    windows = []
    for y in years:
        start_date = f"{int(y)}-{start_md}"
        end_year = int(y) + 1 if crosses_year else int(y)
        end_date = f"{end_year}-{end_md}"
        windows.append([start_date, end_date])

    return windows



def _catalogue_search(catalog, collection, bbox, daterange, query, mission, allow_empty: bool = False):

    results = catalog.search(
        bbox=bbox,
        collections=[collection],
        datetime=daterange,
        query=query,
    )

    items = results.item_collection()

    if mission == "sentinel_2_l1c":
        for item in items:
            for asset in item.assets.values():
                asset.href = asset.href.replace("sentinel-s2-l2a", "sentinel-s2-l1c")
        os.environ["AWS_REQUEST_PAYER"] = "requester"
        os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

    if len(items) < 1:
        if allow_empty:
            return [], None, None, None
        raise ValueError(
            "No scenes found by the given parameters. Please check your polygon's geometry, date range or increase max cloud coverage."
        )

    sample_item = items[0]
    crs = sample_item.properties.get("proj:code") or sample_item.properties.get(
        "proj:epsg"
    )
    stac_mission = sample_item.to_dict().get("collection")
    # Get Sentinel tile ID
    if mission in ("sentinel_2_l2a", "sentinel_2_l1c"):
        gdf = gpd.GeoDataFrame.from_features(items, "epsg:4326")
        if "mgrs:utm_zone" in gdf.columns:
            gdf["granule"] = (
                gdf["mgrs:utm_zone"].apply(lambda x: f"{x:02d}")
                + gdf["mgrs:latitude_band"]
                + gdf["mgrs:grid_square"]
            )
        elif "s2:mgrs_tile" in gdf.columns:
            gdf["granule"] = gdf["s2:mgrs_tile"]
        else:
            gdf["granule"] = "unknown"
        tiles = gdf["granule"].unique()
    else:
        tiles = None

    return items, crs, stac_mission, tiles


def _get_band_map(mission: str, source: str = None):
    if mission == "sentinel_2_l2a" and source in ("terrabyte", "planetary_computer"):
        return dict(_S2_COMMON_TO_BNUM)

    band_maps = {
        "landsat_ot_c2_l2": {
            "coastal": "B01",
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "nir": "B05",
            "swir1": "B06",
            "swir2": "B07",
            "thermal": "B10",
            "qa_temp": "QA_Temp",
            "qa_pixel": "QA_Pixel",
            "qa_radsat": "QA_Radsat",
            "qa_aerosol": "QA_Aerosol",
        },
        "landsat_c2_l2": {
            "coastal": "coastal",
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir": "nir08",
            "swir1": "swir16",
            "swir2": "swir22",
            "thermal": "lwir11",  # SCALE FACTOR FOR THERMAL IS MISSING!
            "qa_pixel": "qa_pixel",
            "qa_radsat": "qa_radsat",
            "qa_aerosol": "qa_aerosol",
        },
    }

    return band_maps.get(mission)