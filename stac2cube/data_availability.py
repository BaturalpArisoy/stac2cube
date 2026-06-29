"""
Sentinel-2 scene availability across the STAC catalogues stac2cube can download from.

Used by the Data Cube Builder GUI's "Check data availability" button: given the
same area + date range the user set for a cube, it lists how many scenes each
catalogue offers per acquisition day, so users can pick the source that best
covers their study area before committing to a (possibly long) download.

Only Sentinel-2 L2A and L1C are supported (the only missions stac2cube serves
from more than one catalogue). Each catalogue is queried independently and
fail-safe: a catalogue that errors (e.g. CDSE when its access keys are not set,
or a transient API error) becomes an all-NaN column instead of aborting the
whole table.
"""

import time

import pandas as pd
from pystac_client import Client as pystacclient

from .vector_refiner import polygon_2_bbox
from .get_data import (
    _parse_season_daterange,
    _parse_years_spec,
    _expand_season_windows,
)

# mission -> ordered list of (source_key, label, url, collection).
# Order matches the Data Source dropdown: element84, PC, terrabyte, Copernicus.
# Planetary Computer has no L1C collection, so it is omitted from L1C.
_AVAILABILITY_CATALOGUES = {
    "sentinel_2_l2a": [
        ("element84", "Element84",
         "https://earth-search.aws.element84.com/v1/", "sentinel-2-l2a"),
        ("planetary_computer", "Planetary Computer",
         "https://planetarycomputer.microsoft.com/api/stac/v1", "sentinel-2-l2a"),
        ("terrabyte", "terrabyte",
         "https://stac.terrabyte.lrz.de/public/api/", "sentinel-2-c1-l2a"),
        ("cdse", "Copernicus",
         "https://stac.dataspace.copernicus.eu/v1", "sentinel-2-l2a"),
    ],
    # NOTE: terrabyte L1C (sentinel-2-c1-l1c) is listed here for availability
    # comparison only. get_data.py does not wire a terrabyte L1C download source,
    # so terrabyte is NOT offered in the Data Source dropdown for L1C - users who
    # see L1C scenes here must request L1C data from DLR to actually use it.
    "sentinel_2_l1c": [
        ("element84", "Element84",
         "https://earth-search.aws.element84.com/v1/", "sentinel-2-l1c"),
        ("terrabyte", "terrabyte",
         "https://stac.terrabyte.lrz.de/public/api/", "sentinel-2-c1-l1c"),
        ("cdse", "Copernicus",
         "https://stac.dataspace.copernicus.eu/v1", "sentinel-2-l1c"),
    ],
}

_CLIENTS = {}


def _client(url):
    if url not in _CLIENTS:
        _CLIENTS[url] = pystacclient.open(url)
    return _CLIENTS[url]


def _resolve_windows(mission, daterange):
    """Turn the GUI's daterange object into a list of concrete [start, end] ISO
    windows to query. Mirrors get_stac's season handling so seasonal ranges
    (e.g. {"season": ["04-01","10-31"], "years": [2020, 2021]}) are expanded the
    same way the real download would expand them. A plain [start, end] becomes a
    single window; None means the entire archive (a single window of None)."""
    season_spec = _parse_season_daterange(daterange)
    if season_spec is None:
        return [daterange]  # ["YYYY-MM-DD", "YYYY-MM-DD"] or None
    start_md, end_md, years_spec = season_spec
    years = _parse_years_spec(years_spec, mission)
    return _expand_season_windows(start_md, end_md, years)


def _count_days_for_window(client, collection, bbox, window, limit=100, retries=3):
    """Per-day scene counts for one catalogue over one date window. Returns a
    dict {day (YYYY-MM-DD): n_scenes}. Uses the STAC 'fields' extension to fetch
    only properties.datetime (much lighter/faster on full-archive queries, and
    avoids CDSE's 100-item cap on untrimmed sentinel-2-l2a responses). Iterates
    raw dicts via items_as_dicts() because the trimmed response drops the
    top-level 'type' field that pystac Item parsing needs. Retries transient API
    errors before giving up."""
    last = None
    for attempt in range(retries):
        try:
            search = client.search(
                collections=[collection],
                bbox=bbox,
                datetime=window,
                limit=limit,
                fields={"include": ["properties.datetime"]},
            )
            counts = {}
            for d in search.items_as_dicts():
                dt = d.get("properties", {}).get("datetime")
                if dt:
                    day = dt[:10]
                    counts[day] = counts.get(day, 0) + 1
            return counts
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def check_scene_availability(mission, polygon, daterange):
    """
    Build a per-date scene-availability table across every catalogue that serves
    ``mission``, for the given area and date range.

    Parameters
    ----------
    mission : str
        Must be ``"sentinel_2_l2a"`` or ``"sentinel_2_l1c"``. Any other mission
        raises ValueError (there is no multi-catalogue availability to compare).
    polygon : str | list
        Either a path to a vector file or a WGS84 bbox [xmin, ymin, xmax, ymax].
    daterange : list | dict | None
        The same shapes the builder uses (standard, seasonal, or None for the
        full archive).

    Returns
    -------
    (df, errors)
        df : pandas.DataFrame indexed by acquisition day (YYYY-MM-DD), one column
             per catalogue, values = number of scenes that day (0 = queried, none
             that day). A catalogue that failed is an all-NaN column. Empty df if
             no scenes were found in any working catalogue.
        errors : dict {column_label: error_message} for catalogues that failed.

    Notes
    -----
    This reports RAW archive availability (catalogue listing only). It does NOT
    apply a cloud-cover filter, and a scene being listed does not guarantee its
    pixels are downloadable without credentials (e.g. CDSE needs access keys to
    read pixels, though its catalogue search is anonymous).
    """
    if mission not in _AVAILABILITY_CATALOGUES:
        raise ValueError(
            "Scene availability is only available for Sentinel-2 L2A and L1C."
        )

    if polygon is None:
        raise ValueError("Please provide a polygon file or a bbox first.")

    bbox = polygon if isinstance(polygon, (list, tuple)) else polygon_2_bbox(polygon)
    if not bbox:
        raise ValueError("Could not derive a bounding box from the polygon input.")

    windows = _resolve_windows(mission, daterange)
    catalogues = _AVAILABILITY_CATALOGUES[mission]

    per_source = {}   # label -> {day: count}; a missing label means it failed
    errors = {}

    for _source_key, label, url, collection in catalogues:
        try:
            client = _client(url)
            counts = {}
            for win in windows:
                win_counts = _count_days_for_window(client, collection, bbox, win)
                for day, n in win_counts.items():
                    counts[day] = counts.get(day, 0) + n
            per_source[label] = counts
        except Exception as e:
            errors[label] = f"{type(e).__name__}: {e}"

    labels = [label for _key, label, _url, _coll in catalogues]
    all_days = sorted({day for counts in per_source.values() for day in counts})

    data = {}
    for label in labels:
        if label in per_source:
            counts = per_source[label]
            data[label] = [counts.get(day, 0) for day in all_days]
        else:
            data[label] = [float("nan")] * len(all_days)

    df = pd.DataFrame(data, index=all_days)
    df.index.name = "date"
    return df, errors
