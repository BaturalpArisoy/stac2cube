from .vector_refiner import polygon_2_bbox
from .cdse_auth import configure_cdse_environment, configure_anonymous_aws_environment

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

# --- Copernicus Data Space Ecosystem (CDSE) asset naming --------------------
# CDSE STAC asset keys differ from element84:
#   * L1C assets are plain band numbers (B01..B12, B8A, B10).
#   * L2A assets are resolution-suffixed (B04_10m, B05_20m, B01_60m, SCL_20m...)
#     and there is NO plain "B04" asset, so each band must be requested at one
#     concrete resolution. We pick each band's NATIVE resolution; odc.stac then
#     resamples to the user-requested output resolution.
# Maps go common-name -> CDSE asset key. After loading, the assets are renamed
# back to the canonical common names the rest of the pipeline expects.
_S2_CDSE_L1C = {
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
    "cirrus": "B10",
    "swir16": "B11",
    "swir22": "B12",
}
_S2_CDSE_L2A = {
    "coastal": "B01_60m",
    "blue": "B02_10m",
    "green": "B03_10m",
    "red": "B04_10m",
    "rededge1": "B05_20m",
    "rededge2": "B06_20m",
    "rededge3": "B07_20m",
    "nir": "B08_10m",
    "nir08": "B8A_20m",
    "nir09": "B09_60m",
    "swir16": "B11_20m",
    "swir22": "B12_20m",
    "scl": "SCL_20m",
    "aot": "AOT_10m",
    "wvp": "WVP_10m",
}


# --- Scene-level STAC metadata coordinates ----------------------------------
# Optional per-scene item properties the user can attach to the cube as
# (time,) coordinates (scene_metadata parameter). Canonical, NetCDF-safe names
# (no colons) mapped to per-source resolution logic below. Availability was
# verified empirically against all four live APIs on the SAME acquisition
# (S2A T32UPU 2024-06-29):
#   * element84 lacks view:azimuth / view:incidence_angle (they exist only in
#     the granule MTD_TL.xml, not as item properties) and sat:relative_orbit
#     (derived here from s2:product_uri instead).
#   * planetary_computer lacks view:azimuth / view:incidence_angle too and
#     stores solar geometry as s2:mean_solar_azimuth / s2:mean_solar_zenith
#     (elevation derived as 90 - zenith, exact per the probe).
#   * terrabyte and cdse publish everything (cdse baseline sits in
#     processing:version, handled by _s2_baseline).
# NOTE on acq_datetime: providers disagree on the timestamp convention for the
# same scene - element84 stores the tile-level sensing time while tb/pc/cdse
# store the datatake start (minutes apart). Values are correct per source but
# not reproducible across sources.
_SCENE_METADATA_FIELDS = [
    "acq_datetime",
    "view_azimuth",
    "sun_azimuth",
    "sun_elevation",
    "incidence_angle",
    "relative_orbit",
    "processing_baseline",
]

# Which canonical fields each source actually publishes (verified above).
# The GUI reads this to offer only selectable-and-non-empty options per source.
SCENE_METADATA_AVAILABILITY = {
    "element84": [
        "acq_datetime", "sun_azimuth", "sun_elevation",
        "relative_orbit", "processing_baseline",
    ],
    "terrabyte": list(_SCENE_METADATA_FIELDS),
    "planetary_computer": [
        "acq_datetime", "sun_azimuth", "sun_elevation",
        "relative_orbit", "processing_baseline",
    ],
    "cdse": list(_SCENE_METADATA_FIELDS),
}

_REL_ORBIT_RE = re.compile(r"_R(\d{3})_")


def _prop_view_azimuth(p):
    return p.get("view:azimuth")


def _prop_sun_azimuth(p):
    v = p.get("view:sun_azimuth")
    if v is None:
        v = p.get("s2:mean_solar_azimuth")  # planetary_computer
    return v


def _prop_sun_elevation(p):
    v = p.get("view:sun_elevation")
    if v is None:
        z = p.get("s2:mean_solar_zenith")  # planetary_computer
        if z is not None:
            v = 90.0 - float(z)
    return v


def _prop_incidence_angle(p):
    return p.get("view:incidence_angle")


def _prop_relative_orbit(p):
    v = p.get("sat:relative_orbit")
    if v is None:
        # element84: not a property, but deterministically parseable from the
        # product name (e.g. S2A_MSIL2A_..._R065_T32UPU_...).
        m = _REL_ORBIT_RE.search(str(p.get("s2:product_uri", "")))
        if m:
            v = int(m.group(1))
    return v


def _prop_processing_baseline(p):
    v = p.get("s2:processing_baseline")
    if v is None:
        v = p.get("processing:version")  # cdse
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# canonical name -> (getter over item.properties, reduction over a solar day).
# Reductions when several items share a solar day (tile overlap / adjacent
# granules merged by groupby="solar_day"):
#   * angles -> mean (they differ by well under a degree within one orbit)
#   * acq_datetime -> earliest
#   * discrete values (relative_orbit, processing_baseline) -> must be unique;
#     if items disagree, take the first and warn - never invent a merged value.
_SCENE_METADATA_GETTERS = {
    "view_azimuth": (_prop_view_azimuth, "mean"),
    "sun_azimuth": (_prop_sun_azimuth, "mean"),
    "sun_elevation": (_prop_sun_elevation, "mean"),
    "incidence_angle": (_prop_incidence_angle, "mean"),
    "relative_orbit": (_prop_relative_orbit, "unique"),
    "processing_baseline": (_prop_processing_baseline, "unique"),
}


def extract_scene_metadata(items, stac, scene_metadata, source):
    """Attach requested per-scene STAC properties as (time,) coordinates.

    ``items`` are the STAC items the cube was loaded from, ``stac`` the loaded
    (lazy) dataset/array whose time axis follows groupby="solar_day". Values
    are grouped by the item's UTC acquisition date and reduced per solar day
    (see _SCENE_METADATA_GETTERS). Days without a matching item, and fields
    the source does not publish, are NaN/NaT - stated in a warning, never
    guessed.

    Returns ``(stac, n_multi_item_days)`` where the count says on how many
    solar days values were reduced from >1 item (tile overlap) - the GUI uses
    it for its "polygon spans multiple tiles" note.
    """
    requested = [str(f) for f in scene_metadata]
    unknown = [f for f in requested if f not in _SCENE_METADATA_FIELDS]
    if unknown:
        raise ValueError(
            f"Unknown scene_metadata field(s) {unknown}. "
            f"Valid options: {_SCENE_METADATA_FIELDS}."
        )

    # Group item properties per UTC acquisition date (same day-key convention
    # as the orbit_state / solar_azimuths_for_days helpers).
    per_day = {}
    for item in items:
        day = item.properties.get("datetime", "")[:10]
        per_day.setdefault(day, []).append(item.properties)

    solar_days = [str(pd.Timestamp(t).date()) for t in stac.time.values]
    n_multi = sum(
        1 for d in set(solar_days) if len(per_day.get(d, [])) > 1
    )

    coords = {}
    for field in requested:
        if field == "acq_datetime":
            values = []
            for day in solar_days:
                stamps = [
                    p.get("datetime")
                    for p in per_day.get(day, [])
                    if p.get("datetime")
                ]
                if stamps:
                    values.append(
                        min(pd.to_datetime(s, format="mixed") for s in stamps)
                    )
                else:
                    values.append(pd.NaT)
            arr = pd.to_datetime(values).tz_localize(None).to_numpy(
                dtype="datetime64[ns]"
            )
            if pd.isnull(arr).all():
                print(
                    f"Warning: scene_metadata field 'acq_datetime' has no "
                    f"values on source '{source}' - coordinate is all-NaT."
                )
            coords["acq_datetime"] = ("time", arr)
            continue

        getter, reduction = _SCENE_METADATA_GETTERS[field]
        values = []
        for day in solar_days:
            day_vals = [
                v for v in (getter(p) for p in per_day.get(day, []))
                if v is not None
            ]
            if not day_vals:
                values.append(np.nan)
            elif reduction == "mean":
                values.append(float(np.mean([float(v) for v in day_vals])))
            else:  # unique
                uniq = sorted({float(v) for v in day_vals})
                if len(uniq) > 1:
                    print(
                        f"Warning: scene metadata '{field}' has conflicting "
                        f"values {uniq} on {day} (items from different "
                        f"orbits/baselines merged into one solar day) - "
                        f"keeping the first item's value."
                    )
                values.append(float(day_vals[0]))
        arr = np.asarray(values, dtype="float64")
        if np.isnan(arr).all():
            print(
                f"Warning: scene_metadata field '{field}' is not published "
                f"by source '{source}' (or no item carried it) - coordinate "
                f"is all-NaN. Sources with full coverage: terrabyte, cdse."
            )
        coords[field] = ("time", arr)

    return stac.assign_coords(coords), n_multi


# Categorical layers (class codes / bit-packed QA). These must ALWAYS be
# loaded with "nearest" resampling: interpolating between class codes
# produces meaningless fractional classes (verified: bilinear SCL yields
# values like 4.0625 at class boundaries), and interpolated bit-packed QA
# words are garbage. Spectral bands keep the user-selected method.
_CATEGORICAL_BANDS = {"scl", "qa_pixel", "qa_radsat", "qa_aerosol", "qa_temp"}

# User-friendly aliases -> rasterio.enums.Resampling names.
_RESAMPLING_ALIASES = {"bicubic": "cubic"}


# STAC catalogue endpoints per mission (and, for Sentinel-2, per source).
# Module-level so metadata-only helpers (e.g. get_solar_geometry) can reuse
# the exact same endpoints as the pixel loader.
_CATALOGUES = {
    "sentinel_2_l2a": {
        "element84": ("https://earth-search.aws.element84.com/v1/", "sentinel-2-l2a"),
        "terrabyte": ("https://stac.terrabyte.lrz.de/public/api/", "sentinel-2-c1-l2a"),
        "planetary_computer": ("https://planetarycomputer.microsoft.com/api/stac/v1", "sentinel-2-l2a"),
        "cdse": ("https://stac.dataspace.copernicus.eu/v1", "sentinel-2-l2a"),
    },
    "sentinel_2_l1c": {
        "element84": ("https://earth-search.aws.element84.com/v1/", "sentinel-2-l1c"),
        "cdse": ("https://stac.dataspace.copernicus.eu/v1", "sentinel-2-l1c"),
    },
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


def _item_tile(item):
    """MGRS tile id (e.g. "47TPK") of a Sentinel-2 STAC item.

    Mirrors the tile derivation in :func:`_catalogue_search`: element84/PC/tb
    expose the MGRS zone/band/square (or ``s2:mgrs_tile``), CDSE the
    ``grid:code`` (``MGRS-47TPK``). Returns "unknown" if none is present.
    """
    p = item.properties
    if p.get("mgrs:utm_zone") is not None:
        return (
            f"{int(p['mgrs:utm_zone']):02d}"
            f"{p.get('mgrs:latitude_band', '')}"
            f"{p.get('mgrs:grid_square', '')}"
        )
    if p.get("s2:mgrs_tile"):
        return str(p["s2:mgrs_tile"])
    if p.get("grid:code"):
        return str(p["grid:code"]).replace("MGRS-", "")
    return "unknown"


def _s2_baseline(item):
    """Sentinel-2 processing baseline as a float.

    element84/PC/terrabyte expose it as ``s2:processing_baseline``; CDSE exposes
    it as ``processing:version`` (e.g. "05.12"). Returns 0.0 if unavailable.
    """
    v = item.properties.get("s2:processing_baseline")
    if v is None:
        v = item.properties.get("processing:version")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def get_stac(
    mission: str,
    polygon,
    resolution: int,
    daterange: list,
    bands: list,
    max_cc: int,
    cloud_masking: bool,
    source: str = "element84",
    resampling: str = "nearest",
    scene_metadata: list = None,
    tile_handling: str = "mosaic",
):
    _source_aliases = {"e84": "element84", "tb": "terrabyte", "pc": "planetary_computer"}
    source = _source_aliases.get(source, source)

    catalogues = _CATALOGUES

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

    mission_cat = catalogues[mission]
    if isinstance(mission_cat, dict):
        if source not in mission_cat:
            raise ValueError(
                f"Unknown source '{source}' for {mission}. "
                f"Valid options: {list(mission_cat.keys())} "
                "(short aliases: e84, tb, pc, cdse)."
            )
        url, collection = mission_cat[source]
    else:
        url, collection = mission_cat

    # CDSE serves pixels from s3://eodata and needs the user's S3 keys at
    # compute time. Export them now (also gives an early, clear error if the
    # credentials file is missing/unfilled) and reset any anonymous settings
    # that a previous source may have left behind.
    if source == "cdse":
        configure_cdse_environment()
    elif source == "element84":
        # element84 reads AWS-hosted public buckets. Reset the S3 env to unsigned
        # AWS access so a CDSE query earlier in the SAME session (which points
        # AWS_S3_ENDPOINT at eodata.dataspace.copernicus.eu) can't misdirect these
        # reads and fail them all silently. L2A COGs (sentinel-cogs) are free; the
        # L1C bucket (sentinel-s2-l1c) is requester-pays and gets its flag in
        # _catalogue_search when the hrefs are rewritten.
        configure_anonymous_aws_environment(
            requester_pays=(mission == "sentinel_2_l1c")
        )

    needs_pc_auth = mission in ("sentinel_1_rtc", "landsat_c2_l2") or (
        mission == "sentinel_2_l2a" and source == "planetary_computer"
    )
    if needs_pc_auth:
        catalog = pystacclient.open(url, modifier=planetary_computer.sign_inplace)
    else:
        catalog = pystacclient.open(url)

    # Only filter on cloud cover when an explicit max_cc is given. A None upper
    # bound (max_cc not supplied) must mean "no cloud filter": element84's STAC
    # server tolerates {"lte": null}, but Planetary Computer / terrabyte / CDSE
    # treat "cloud_cover <= null" as matching nothing -> spurious "No scenes
    # found". Missions without cloud metadata never get a cloud query.
    if max_cc is None or mission in ("cop_dem_glo_30", "sentinel_1_rtc"):
        query = None
    else:
        query = {"eo:cloud_cover": {"gte": 0, "lte": max_cc}}

    season_spec = _parse_season_daterange(daterange)

    if season_spec is None:
        items, crs, stac_mission, tiles = _catalogue_search(
            catalog, collection, bbox, daterange, query, mission, source=source
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
                catalog, collection, bbox, win, query, mission,
                allow_empty=True, source=source,
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

    canonical_bands = list(bands) if bands else []
    band_map = _get_band_map(mission, source)
    if band_map is not None:
        bands = [band_map.get(band, band) for band in bands]

    # --- resampling strategy ------------------------------------------------
    # Normalize/validate the user's method, then build a per-band config:
    # spectral bands use the requested method, categorical layers (SCL / QA)
    # are pinned to "nearest" regardless of that choice.
    resampling = _RESAMPLING_ALIASES.get(str(resampling).lower(), str(resampling).lower())
    from rasterio.enums import Resampling as _Resampling

    if resampling not in _Resampling.__members__:
        raise ValueError(
            f"Unknown resampling method '{resampling}'. Valid options: "
            f"{sorted(_Resampling.__members__)} (alias: 'bicubic' -> 'cubic')."
        )
    resampling_cfg = {"*": resampling}
    for canon, mapped in zip(canonical_bands, bands or []):
        if str(canon).lower() in _CATEGORICAL_BANDS:
            resampling_cfg[mapped] = "nearest"

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
            best_item = max(group, key=_s2_baseline)
            filtered_items.append(best_item)
        items = filtered_items

    _load_kwargs = dict(
        bands=bands,
        crs=crs,
        resolution=resolution,
        # Per-band dict: the user's method (default "nearest") for spectral
        # bands, "nearest" pinned for categorical layers - odc-stac resolves
        # each band against this mapping with "*" as the fallback.
        resampling=resampling_cfg,
        chunks={},
        groupby="solar_day",
        bbox=bbox,
        # Don't abort the whole (potentially terabyte-scale) load when a single
        # granule is missing/unreadable on the archive. Such reads are skipped
        # and filled with the band's nodata value instead, so the remaining
        # dates and bands still load. This guards against terrabyte STAC index
        # being out of sync with the physical .jp2 files on DSS.
        fail_on_error=False,
    )

    # --- Tile handling --------------------------------------------------------
    # "mosaic" (default): the original behaviour - groupby="solar_day" merges
    # every item of a solar day (incl. items from adjacent MGRS tiles that both
    # overlap the AOI) into ONE timestep. "separate": AOIs that straddle two
    # N-S tiles (e.g. Orog Nuur, Mongolia = 47TPK/47TPL) keep each tile's
    # acquisition as its OWN timestep. Implemented as one solar-day load PER
    # tile (which still mosaics duplicate orbits/processing within a tile+day)
    # then concatenated along time, so each timestep carries an exact `tile`
    # coordinate and the full-precision acquisition time stays unique even when
    # two tiles share a solar day (verified on the real split AOI).
    _separate = mission == "sentinel_2_l2a" and tile_handling == "separate"
    if _separate:
        from collections import defaultdict

        by_tile = defaultdict(list)
        for it in items:
            by_tile[_item_tile(it)].append(it)

        parts = []
        for tname in sorted(by_tile):
            tds = stac_load(by_tile[tname], **_load_kwargs)
            tds = tds.assign_coords(
                tile=("time", np.array([tname] * tds.sizes["time"], dtype="U8"))
            )
            # Per-tile scene metadata: extracting on this tile's items alone
            # keeps the values tile-specific (no cross-tile averaging). Done
            # here (not in the post-load block below) because that block would
            # re-key by solar day across the concatenated, tile-mixed axis.
            if scene_metadata:
                tds, _ = extract_scene_metadata(
                    by_tile[tname], tds, scene_metadata, source
                )
            parts.append(tds)

        # coords="minimal"/compat="override": spatial_ref is an identical scalar
        # coord on every part; let the first win instead of failing the concat.
        stac = xr.concat(parts, dim="time", coords="minimal", compat="override")
        stac = stac.sortby("time")
    else:
        stac = stac_load(items, **_load_kwargs)

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
        processing_baseline_list = [_s2_baseline(item) for item in items]
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

        if scene_metadata:
            # Runs after the baseline dedup so the coords reflect the items
            # that actually survive into the cube. The multi-item-day count is
            # stashed as a Dataset attr; main.py re-attaches it to the final
            # DataArray (Dataset attrs do not survive the band concat there).
            stac, _n_multi = extract_scene_metadata(
                items, stac, scene_metadata, source
            )
            stac.attrs["scene_metadata_multiday"] = int(_n_multi)

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

        if scene_metadata and not _separate:
            stac, _n_multi = extract_scene_metadata(
                items, stac, scene_metadata, source
            )
            stac.attrs["scene_metadata_multiday"] = int(_n_multi)
        elif _separate:
            # Tiles are kept apart, never cross-merged, so the "polygon spans
            # multiple tiles -> mean" note does not apply here.
            stac.attrs["scene_metadata_multiday"] = 0

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



def _catalogue_search(catalog, collection, bbox, daterange, query, mission,
                      allow_empty: bool = False, source: str = None):

    results = catalog.search(
        bbox=bbox,
        collections=[collection],
        datetime=daterange,
        query=query,
    )

    items = results.item_collection()

    # element84 stores L1C under the same S3 path as L2A with anonymous,
    # requester-pays access -> rewrite hrefs and (re)assert a clean unsigned AWS
    # env. The full reset (not just the two flags) is what clears any CDSE S3
    # endpoint/keys a previous CDSE query left behind, which would otherwise
    # misdirect these reads and fail every band silently.
    # CDSE L1C uses its own s3://eodata hrefs and *signed* access, so it must
    # NOT go through this path (its env is configured in get_stac instead).
    if mission == "sentinel_2_l1c" and source == "element84":
        for item in items:
            for asset in item.assets.values():
                asset.href = asset.href.replace("sentinel-s2-l2a", "sentinel-s2-l1c")
        configure_anonymous_aws_environment(requester_pays=True)

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
    # CDSE keeps the CRS at the asset level (proj:code = "EPSG:32632"), not in
    # item.properties. Fall back to the first asset that carries it.
    if crs is None:
        for asset in sample_item.assets.values():
            asset_crs = asset.extra_fields.get("proj:code") or asset.extra_fields.get(
                "proj:epsg"
            )
            if asset_crs is not None:
                crs = asset_crs
                break
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
        elif "grid:code" in gdf.columns:
            # CDSE exposes the tile as grid:code = "MGRS-32UPU".
            gdf["granule"] = (
                gdf["grid:code"].astype(str).str.replace("MGRS-", "", regex=False)
            )
        else:
            gdf["granule"] = "unknown"
        tiles = gdf["granule"].unique()
    else:
        tiles = None

    return items, crs, stac_mission, tiles


def get_solar_geometry(mission, polygon, daterange, source="element84", max_cc=None):
    """Per-solar-day mean solar geometry from STAC item metadata (no pixels).

    Queries the same catalogue endpoint as :func:`get_stac` and reads the mean
    solar angles the providers store on every Sentinel-2 item:

      * element84 / CDSE:        ``view:sun_azimuth`` / ``view:sun_elevation``
      * planetary_computer / tb: ``s2:mean_solar_azimuth`` / ``s2:mean_solar_zenith``
        (elevation derived as 90 - zenith)

    Returns ``{"YYYY-MM-DD": {"sun_azimuth": float, "sun_elevation": float}}``.
    When several items share a solar day (tile overlap) the angles are
    averaged - they differ by well under a degree within one orbit.
    Days where neither property exists are OMITTED (never guessed); the
    caller must handle missing days explicitly.
    """
    _source_aliases = {"e84": "element84", "tb": "terrabyte", "pc": "planetary_computer"}
    source = _source_aliases.get(source, source)

    mission_cat = _CATALOGUES[mission]
    if isinstance(mission_cat, dict):
        if source not in mission_cat:
            raise ValueError(
                f"Unknown source '{source}' for {mission}. "
                f"Valid options: {list(mission_cat.keys())}."
            )
        url, collection = mission_cat[source]
    else:
        url, collection = mission_cat

    if isinstance(polygon, list):
        bbox = polygon
    else:
        bbox = polygon_2_bbox(polygon)

    # Item search only - no asset access, so no signing / S3 env needed.
    catalog = pystacclient.open(url)
    query = None
    if max_cc is not None:
        query = {"eo:cloud_cover": {"gte": 0, "lte": max_cc}}
    results = catalog.search(
        bbox=bbox, collections=[collection], datetime=daterange, query=query
    )

    per_day = {}
    for item in results.item_collection():
        p = item.properties
        azimuth = p.get("view:sun_azimuth")
        if azimuth is None:
            azimuth = p.get("s2:mean_solar_azimuth")
        elevation = p.get("view:sun_elevation")
        if elevation is None:
            zenith = p.get("s2:mean_solar_zenith")
            if zenith is not None:
                elevation = 90.0 - float(zenith)
        if azimuth is None:
            continue  # no angle metadata on this item - skip, never guess
        day = p.get("datetime", "")[:10]
        per_day.setdefault(day, []).append(
            (float(azimuth), float(elevation) if elevation is not None else np.nan)
        )

    out = {}
    for day, vals in per_day.items():
        arr = np.asarray(vals, dtype=float)
        out[day] = {
            "sun_azimuth": float(np.mean(arr[:, 0])),
            "sun_elevation": float(np.nanmean(arr[:, 1])) if not np.all(np.isnan(arr[:, 1])) else None,
        }
    return out


def export_granule_metadata(stac, output_dir, q=False):
    """Download the granule metadata XML (MTD_TL.xml) of every scene in a cube.

    Self-contained from the cube's own metadata (mission / bbox / stac_api
    attrs + the time coordinate): re-queries the SAME STAC catalogue the cube
    was built from (item search only - no pixels) and downloads the
    ``granule_metadata`` asset of every item whose acquisition day is in the
    cube. One XML per granule: dates covered by several granules (tile
    overlap) get one file each, named ``<item_id>.xml``.

    Per-source access (asset hrefs differ, verified live):
      * element84 / terrabyte-style https  -> plain HTTP GET
      * planetary_computer                 -> hrefs SAS-signed at search time
      * cdse (s3://eodata)                 -> boto3 with the user's CDSE keys
      * terrabyte (file:///dss/...)        -> local copy; ONLY works on the
        terrabyte cluster itself - off-cluster a clear error is raised.

    Returns the list of written file paths. Days with no matching item and
    per-file failures are reported, never silently skipped.
    """
    import requests as _requests

    attrs = dict(stac.attrs)
    mission = attrs.get("mission")
    if mission not in ("sentinel_2_l2a", "sentinel_2_l1c"):
        raise ValueError(
            "Granule metadata export is available for Sentinel-2 cubes only."
        )
    if "time" not in getattr(stac, "coords", {}):
        raise ValueError(
            "Granule metadata export needs the cube's time coordinate "
            "(a temporal composite has none)."
        )
    bbox = attrs.get("bbox")
    if bbox is None:
        raise ValueError("The cube carries no bbox attribute - cannot re-query.")
    bbox = [float(v) for v in np.asarray(bbox).ravel()]
    source = str(attrs.get("stac_api", "element84"))
    _source_aliases = {"e84": "element84", "tb": "terrabyte", "pc": "planetary_computer"}
    source = _source_aliases.get(source, source)

    url, collection = _CATALOGUES[mission][source] if isinstance(
        _CATALOGUES[mission], dict
    ) else _CATALOGUES[mission]

    days = sorted(
        {str(t)[:10] for t in np.asarray(stac["time"].values).astype("datetime64[D]")}
    )
    daterange = [days[0], days[-1]]

    # PC asset hrefs are private Azure blobs; sign_inplace attaches the SAS
    # token during the search so the hrefs below are directly downloadable.
    if source == "planetary_computer":
        import planetary_computer as _pc

        catalog = pystacclient.open(url, modifier=_pc.sign_inplace)
    else:
        catalog = pystacclient.open(url)
    results = catalog.search(bbox=bbox, collections=[collection], datetime=daterange)
    items = [
        it for it in results.item_collection()
        if it.properties.get("datetime", "")[:10] in set(days)
    ]
    if not items:
        raise ValueError(
            "The catalogue returned no items for the cube's dates - nothing "
            "to download (was the cube built from a different source?)."
        )

    # CDSE XMLs live on s3://eodata and need the user's keys (same file the
    # pixel reads use). One boto3 client for all downloads.
    _s3 = None
    if source == "cdse":
        import boto3
        from .cdse_auth import read_cdse_credentials

        access, secret, endpoint, region = read_cdse_credentials()
        _s3 = boto3.client(
            "s3",
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            endpoint_url=f"https://{endpoint}",
            region_name=region,
        )

    os.makedirs(output_dir, exist_ok=True)
    written = []
    failed = []
    found_days = set()
    for item in items:
        # element84/terrabyte/cdse name the asset granule_metadata, PC uses
        # granule-metadata. Take whichever exists - never guess a URL.
        asset = item.assets.get("granule_metadata") or item.assets.get(
            "granule-metadata"
        )
        if asset is None:
            failed.append((item.id, "no granule_metadata asset on the item"))
            continue
        href = asset.href
        out_path = os.path.join(output_dir, f"{item.id}.xml")
        try:
            if href.startswith("s3://"):
                if _s3 is None:
                    raise ValueError(
                        f"s3:// asset href on source '{source}' - no S3 "
                        "client configured for it."
                    )
                _bucket, _key = href[5:].split("/", 1)
                _s3.download_file(_bucket, _key, out_path)
            elif href.startswith("file://"):
                # terrabyte publishes cluster-local DSS paths. Copy works on
                # the cluster; anywhere else the path simply does not exist.
                import shutil
                from urllib.request import url2pathname
                from urllib.parse import urlparse

                local = url2pathname(urlparse(href).path)
                if not os.path.exists(local):
                    raise FileNotFoundError(
                        "terrabyte stores granule metadata as cluster-local "
                        f"paths ({href}); they can only be downloaded when "
                        "running ON the terrabyte cluster itself. Off-cluster, "
                        "rebuild from element84/cdse to export the XMLs."
                    )
                shutil.copyfile(local, out_path)
            else:
                r = _requests.get(href, timeout=60)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(r.content)
            written.append(out_path)
            found_days.add(item.properties.get("datetime", "")[:10])
        except Exception as e:  # report per file, keep going
            failed.append((item.id, f"{type(e).__name__}: {e}"))

    missing_days = [d for d in days if d not in found_days]
    if not q:
        print(
            f"Granule metadata: {len(written)} XML file(s) written to "
            f"{output_dir}",
            flush=True,
        )
    if failed:
        print("WARNING: some granule metadata files could not be downloaded:")
        for iid, err in failed:
            print(f"  - {iid}: {err}")
    if missing_days:
        print(
            "WARNING: no granule metadata found for these cube dates: "
            + ", ".join(missing_days)
        )
    if not written and failed:
        raise ValueError(
            "Granule metadata export failed for every scene - see the "
            "warnings above (first error: "
            f"{failed[0][1]})."
        )
    return written


def _get_band_map(mission: str, source: str = None):
    if mission == "sentinel_2_l2a" and source in ("terrabyte", "planetary_computer"):
        return dict(_S2_COMMON_TO_BNUM)
    if source == "cdse" and mission == "sentinel_2_l2a":
        return dict(_S2_CDSE_L2A)
    if source == "cdse" and mission == "sentinel_2_l1c":
        return dict(_S2_CDSE_L1C)

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