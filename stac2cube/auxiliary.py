"""
Interactive map preview of the satellite footprints that cover an area of interest.

Used by the Data Cube Builder GUI's polygon section (next to "Draw the area on a
map"): once a user has an area and a date range, this answers - before any pixel is
downloaded - "which orbits actually see it, how much of it does each one cover, and
how many tiles / projections does it straddle?".

This is an INFORMATION tool, not a filter: it reports what the archive holds and
changes nothing about the build. That is why it reads the user's whole date range
by default rather than a cheap sample - the most useful thing it has to say is how
much a single orbit's coverage varies from pass to pass, and that only exists
across many dates (measured on a 6 km AOI sitting on a swath edge: one orbit
covered it 100% on all 71 passes of 2024 while the neighbouring orbit ranged from
9% to 58%, with no usable periodicity). ``window_days`` restores the old cheap
probe for anyone who only needs the orbit/tile geometry.

Only STAC metadata is read (item geometries and a handful of properties), so a
preview costs one catalogue search per date window and no pixel traffic.

What the footprints ARE
-----------------------
A Sentinel-2 STAC item's geometry is the GRANULE footprint: the MGRS tile clipped
to the datatake swath. It is therefore not a full orbit swath, and a granule at
the across-track edge of a datatake is only a sliver of its tile. Grouping the
granule footprints by relative orbit reconstructs the part of each swath that
touches the AOI, which is the useful view: one orbit = one repeating acquisition
geometry = one set of dates that all look the same.

Caveats (please keep them in any user-facing text)
-------------------------------------------------
* STAC footprint polygons are GENERALISED outlines (7-8 vertices for a Sentinel-2
  granule), not pixel-exact masks. Coverage percentages computed here are
  geometric estimates, accurate to the polygon, not to the pixel.
* They therefore PREDICT, but do not equal, the ``scene_coverage`` coordinate
  stac2cube attaches to a built cube (:func:`stac2cube.clip.compute_scene_coverage`),
  which is measured on real pixels. Both are shares of the SAME denominator, the
  AOI, so the two numbers are directly comparable and the gap between them is
  polygon-versus-pixel alone. (``scene_coverage`` divided by the imaged union of
  all scenes until that was changed in favour of the AOI; a cube built before
  the change carries the old meaning and nothing in the file says so.)
* No cloud filter is applied, deliberately: this is about acquisition geometry,
  which does not change with cloudiness. A cube built with a Max cloud % below
  100 will have fewer dates than the preview reports.
* Coverage numbers are unreliable for an AOI that crosses the antimeridian
  (the union of footprints in lon/lat is meaningless there). Flagged in the
  returned info dict rather than silently reported.
"""

import datetime
import math
import os
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pystac_client import Client as pystacclient

from .export_cfg import open_cube
from .vector_refiner import polygon_2_bbox, polygon_2_features, polygon_2_gdf
from .get_data import (
    _CATALOGUES,
    _EQUAL_AREA,
    _item_tile,
    _prop_relative_orbit,
    _probe_window,
)
from .data_availability import _resolve_windows

__all__ = [
    "preview_scene_footprints",
    "get_scene_footprints",
    "summarize_scene_footprints",
    "footprint_map",
    "satellite_map",
    "export_cube_statistics",
]


# Item properties the footprint query asks for. Requesting a field a catalogue
# does not publish is harmless (it is simply absent from the response), so one
# list serves all four Sentinel-2 catalogues plus Landsat / Sentinel-1.
# "geometry" is the point of the query; everything else identifies the footprint.
_FOOTPRINT_FIELDS = {
    "include": [
        "geometry",
        "properties.datetime",
        "properties.platform",
        # relative orbit: published by PC / terrabyte / CDSE, parsed out of the
        # product name on element84 (see _prop_relative_orbit).
        "properties.sat:relative_orbit",
        "properties.s2:product_uri",
        # MGRS tile, in each catalogue's own spelling (see _item_tile).
        "properties.mgrs:utm_zone",
        "properties.mgrs:latitude_band",
        "properties.mgrs:grid_square",
        "properties.s2:mgrs_tile",
        "properties.grid:code",
        # native projection; CDSE keeps it at asset level, hence the tile-derived
        # fallback in _crs_of.
        "properties.proj:code",
        "properties.proj:epsg",
        # Landsat has no relative orbit; its WRS-2 path/row plays the same role.
        "properties.landsat:wrs_path",
        "properties.landsat:wrs_row",
    ]
}

# Fill colours for the per-orbit layers, cycled. Chosen to stay legible on top of
# dark satellite imagery (the preview map's background), which rules out the
# usual matplotlib defaults - navy and dark green vanish over terrain.
_ORBIT_COLORS = [
    "#00b0f0",  # cyan blue
    "#ffb000",  # amber
    "#00d67d",  # green
    "#ff5ea8",  # pink
    "#b083ff",  # violet
    "#ffe100",  # yellow
    "#ff6f3c",  # orange red
    "#3ddbd9",  # teal
]

# Outline-only styles for the reference layers drawn on top of the orbit fills.
# Missions whose items are a static mosaic rather than dated acquisitions. There
# is nothing to probe a date window for (Copernicus DEM GLO-30 tiles carry a
# nominal timestamp years in the past, so any window anchored on the user's range
# finds nothing), and no orbit to group by: the preview shows their tiles.
_STATIC_MISSIONS = {"cop_dem_glo_30"}

# Safety cap for the undated search a static mission needs. A static collection
# returns a handful of tiles for any AOI; the cap only exists so a mistakenly
# undated search can never page through an entire archive.
_STATIC_MAX_ITEMS = 200

_TILE_STYLE = {
    "color": "#ffffff",
    "weight": 1,
    "dashArray": "4,4",
    "fillOpacity": 0,
    "opacity": 0.9,
}
_BBOX_STYLE = {
    "color": "#ffe100",
    "weight": 2,
    "dashArray": "6,4",
    "fillOpacity": 0,
}
_AOI_STYLE = {
    "color": "#ff2d2d",
    "weight": 3,
    "fillOpacity": 0,
}


# --- footprint identity ------------------------------------------------------


def _orbit_of(props):
    """Repeat-cycle track of a scene, or None when the catalogue exposes none.

    Sentinel-2 / Sentinel-1: the relative orbit number (:func:`_prop_relative_orbit`
    handles element84's missing property by parsing the product name). Landsat has
    no relative orbit, so its WRS-2 path is used instead - the same idea, a fixed
    ground track that repeats. The value is returned as-is (int for an orbit,
    "path NNN" for Landsat) and only ever used as a grouping key and label.
    """
    orbit = _prop_relative_orbit(props)
    if orbit is not None:
        try:
            return int(orbit)
        except (TypeError, ValueError):
            return orbit
    path = props.get("landsat:wrs_path")
    if path is not None:
        return f"path {path}"
    return None


def _tile_of(props):
    """Tiling-grid cell of a scene, or None for a mission that has no grid.

    Sentinel-2's MGRS tile via :func:`_item_tile` (which is fed a stand-in object
    because it reads ``item.properties``, and this module works on the raw
    response dicts the trimmed search returns). Landsat's WRS-2 path/row is
    reported as "PPP/RRR". Sentinel-1 RTC and the DEM have no tiling grid the user
    would recognise, so they get None and the tile layer is simply skipped.
    """
    tile = _item_tile(SimpleNamespace(properties=props))
    if tile and tile != "unknown":
        return tile
    path, row = props.get("landsat:wrs_path"), props.get("landsat:wrs_row")
    if path is not None and row is not None:
        return f"{path}/{row}"
    return None


def _crs_of(props, tile):
    """Native projection of a scene's pixels as "EPSG:<code>", or None.

    ``proj:code`` / ``proj:epsg`` when the catalogue publishes them at item level.
    CDSE does not - it keeps them per asset, which a trimmed metadata search does
    not return - so for a Sentinel-2 MGRS tile the code is derived from the tile id
    instead: an MGRS grid square is defined inside one UTM zone, and its latitude
    band letter gives the hemisphere ("N" and later = north). That is a definition,
    not a guess, but it is only ever a fallback.
    """
    crs = props.get("proj:code") or props.get("proj:epsg")
    if crs is not None:
        if isinstance(crs, int):
            return f"EPSG:{crs}"
        text = str(crs).strip()
        if text.isdigit():
            return f"EPSG:{text}"
        if text.upper().startswith("EPSG:"):
            return "EPSG:" + text.split(":", 1)[1].strip()
        return text
    m = re.fullmatch(r"(\d{2})([A-Z])([A-Z]{2})", str(tile or ""))
    if m:
        zone, band = int(m.group(1)), m.group(2)
        return f"EPSG:{32600 + zone if band >= 'N' else 32700 + zone}"
    return None


# --- catalogue query ---------------------------------------------------------


def _resolve_endpoint(mission, source):
    """(url, collection) for a mission/source pair, with the same source aliases
    and error messages the download path uses."""
    source = {"e84": "element84", "tb": "terrabyte", "pc": "planetary_computer"}.get(
        source, source
    )
    mission_cat = _CATALOGUES.get(mission)
    if mission_cat is None:
        raise ValueError(f"No STAC catalogue configured for mission {mission!r}.")
    if isinstance(mission_cat, dict):
        if source not in mission_cat:
            raise ValueError(
                f"Unknown source {source!r} for {mission}. "
                f"Valid options: {list(mission_cat.keys())}."
            )
        return mission_cat[source]
    return mission_cat


def _footprint_window(mission, daterange, window_days):
    """A short date window to read the acquisition geometry from.

    Only used when the caller asks for a short probe instead of the full range
    (``window_days`` set). Cheap, because which orbits cross an area and roughly
    what footprint each leaves is set by orbit and tile geometry, which repeats
    every cycle: ~55 items for 12 days versus ~4700 for 3 years on the Naryn AOI.
    What it CANNOT show is how much a single orbit's coverage varies between
    passes (measured on a 6 km AOI at a swath edge: 9% to 58% over a year), which
    is why the full range is the default.

    The window is anchored at the END of the user's range, and clamped to its
    start so a short cube never probes dates it did not ask for. A seasonal
    ``daterange`` is expanded exactly as the build expands it and the LAST season
    window is used, so the probe stays inside a season the user actually chose.
    ``None`` (whole archive) anchors at today.
    """
    window = daterange
    if isinstance(daterange, dict):
        windows = _resolve_windows(mission, daterange)
        window = windows[-1] if windows else None

    start, end = None, None
    if isinstance(window, (list, tuple)) and len(window) == 2:
        try:
            start = datetime.date.fromisoformat(str(window[0])[:10])
        except (ValueError, TypeError):
            start = None
        try:
            end = datetime.date.fromisoformat(str(window[1])[:10])
        except (ValueError, TypeError):
            end = None

    probe = _probe_window([window[0], window[1]] if end else None, window_days)
    probe_start = datetime.date.fromisoformat(probe[0])
    if start is not None and probe_start < start:
        probe[0] = start.isoformat()
    return probe


def _full_windows(mission, daterange):
    """Every window that makes up the user's date range, or None if unbounded.

    A plain ``[start, end]`` is one window. A seasonal spec becomes one window per
    year, expanded exactly as the build expands it, so the preview reports the
    dates a seasonal cube would actually contain and nothing from the off-season
    gaps in between. ``None`` means the whole archive, which is not a range that
    can be queried in full here - the caller falls back to a probe window.
    """
    if daterange is None:
        return None
    if isinstance(daterange, dict):
        windows = _resolve_windows(mission, daterange)
        return [list(w) for w in windows if w] or None
    if isinstance(daterange, (list, tuple)) and len(daterange) == 2:
        return [list(daterange)]
    return None


def get_scene_footprints(
    mission,
    polygon,
    source="element84",
    daterange=None,
    window_days=12,
    q=False,
):
    """
    Footprints of the scenes that cover an area, one row per STAC item.

    Parameters
    ----------
    mission : str
        Any mission stac2cube can download (e.g. ``"sentinel_2_l2a"``).
    polygon : str | list | geopandas.GeoDataFrame
        A vector file path, a WGS84 bbox ``[xmin, ymin, xmax, ymax]``, or a
        GeoDataFrame. Multi-feature files are treated as ONE area (their combined
        extent); a build instead runs once per feature.
    source : str
        Catalogue to query, same values as the builder's Data Source.
    daterange : list | dict | None
        The builder's date range (standard, seasonal, or None for the whole
        archive). By default every date in it is read.
    window_days : int | None
        ``None`` (default) reads the WHOLE date range: this is an information
        tool, and the thing users want to know - how much one orbit's coverage
        varies from pass to pass - only exists across many dates. Set an integer
        to read a short probe window at the end of the range instead (see
        :func:`_footprint_window`), which is much faster on multi-year ranges but
        reports 2 to 3 dates per orbit. A range of ``None`` (whole archive) always
        uses a probe window, since the archive has no end to anchor on.
    q : bool
        Quiet: suppress the progress print.

    Returns
    -------
    (gdf, windows)
        gdf : GeoDataFrame in EPSG:4326 with columns ``date``, ``datetime``,
              ``orbit``, ``tile``, ``crs``, ``platform`` and the footprint
              geometry. Empty if the catalogue returned nothing.
        windows : list of ``[start, end]`` actually queried - one entry for a plain
              range, one per year for a seasonal range, empty for a static mission.
    """
    if polygon is None:
        raise ValueError("Please provide a polygon file or a bbox first.")

    url, collection = _resolve_endpoint(mission, source)
    bbox = polygon if isinstance(polygon, (list, tuple)) else polygon_2_bbox(polygon)
    if not bbox:
        raise ValueError("Could not derive a bounding box from the polygon input.")

    catalog = pystacclient.open(url)
    if mission in _STATIC_MISSIONS:
        windows = []
        dicts = _search_dicts(
            catalog, collection, bbox, None, max_items=_STATIC_MAX_ITEMS
        )
    elif window_days is None:
        # Full range: one search per window (a seasonal range is several), so the
        # off-season gaps are never queried and the dates reported are the dates a
        # cube over this range would really hold.
        windows = _full_windows(mission, daterange)
        if windows is None:
            windows = [_footprint_window(mission, daterange, 12)]
        dicts = []
        for win in windows:
            dicts.extend(_search_dicts(catalog, collection, bbox, win))
    else:
        dicts, windows = [], []
        for factor in (1, 4, 12):
            windows = [_footprint_window(mission, daterange, window_days * factor)]
            dicts = _search_dicts(catalog, collection, bbox, windows[0])
            if dicts:
                break
    if not q:
        print(f"Read {len(dicts)} scene footprints for {_span_text(windows)}.")
    if not dicts:
        return gpd.GeoDataFrame(
            columns=["date", "datetime", "orbit", "tile", "crs", "platform", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        ), windows

    from shapely.geometry import shape as _shape

    rows = []
    for d in dicts:
        geom = d.get("geometry")
        if not geom:
            continue  # no footprint published: nothing to draw, and no coverage
        props = d.get("properties", {}) or {}
        tile = _tile_of(props)
        stamp = str(props.get("datetime") or "")
        rows.append(
            {
                "date": stamp[:10],
                "datetime": stamp,
                "orbit": _orbit_of(props),
                "tile": tile,
                "crs": _crs_of(props, tile),
                "platform": props.get("platform"),
                "geometry": _shape(geom),
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf["orbit"] = _orbit_column(gdf["orbit"])
    return gdf, windows


def _span_text(windows):
    """Human wording for the dates a preview covers."""
    if not windows:
        return "the whole collection"
    if len(windows) == 1:
        return f"{windows[0][0]} to {windows[0][1]}"
    return (
        f"{windows[0][0]} to {windows[-1][1]} "
        f"({len(windows)} season windows)"
    )


def _orbit_column(values):
    """The orbit column as object dtype holding real ints and real None.

    Needed because pandas infers a list like ``[7, None]`` as float64 and turns
    the None into NaN - after which ``orbit == 7`` still works but the missing
    entries are neither None nor equal to themselves, so grouping and ``isna()``
    based filtering silently disagree about which rows have no orbit. Floats that
    came from that inference are put back to int.
    """
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
        elif isinstance(v, float):
            out.append(int(v))
        else:
            out.append(v)
    return pd.Series(out, index=getattr(values, "index", None), dtype=object)


def _search_dicts(catalog, collection, bbox, window, retries=3, max_items=None):
    """Items of one search as raw dicts, trimmed to the footprint fields.

    The STAC 'fields' extension keeps the response small (a full Sentinel-2 item
    is mostly asset definitions this module never reads) and, on CDSE, avoids the
    100-item cap on untrimmed sentinel-2-l2a responses. ``items_as_dicts()`` is
    used rather than pystac Items because a trimmed response drops the top-level
    'type' field that Item parsing requires. Falls back to an untrimmed search if
    a catalogue rejects the extension, and retries transient API errors.

    ``max_items`` stops the iteration early; used only for the undated search a
    static mission needs, so it can never page through a whole archive.
    """
    import time

    def _take(iterator):
        out = []
        for d in iterator:
            out.append(d)
            if max_items is not None and len(out) >= max_items:
                break
        return out

    last = None
    for attempt in range(retries):
        try:
            search = catalog.search(
                collections=[collection],
                bbox=bbox,
                datetime=window,
                limit=100,
                fields=_FOOTPRINT_FIELDS,
            )
            return _take(search.items_as_dicts())
        except Exception as e:
            last = e
            if attempt < retries - 1:  # no point sleeping before giving up
                time.sleep(2 * (attempt + 1))
    try:
        search = catalog.search(
            collections=[collection], bbox=bbox, datetime=window, limit=100
        )
        return [i.to_dict() for i in _take(search.items())]
    except Exception:
        raise last


# --- coverage summary --------------------------------------------------------


def _aoi_geometries(polygon, coverage_geometry):
    """(aoi_gdf, bbox, aoi_geom_wgs84) for the area the preview is about.

    ``aoi_gdf`` is the exact outline when a vector file was given (None for a
    bbox input), used for drawing. ``aoi_geom_wgs84`` is what coverage is measured
    against: the bounding box by default, because that IS the cube unless the user
    ticks "clip to exact polygon outline"; ``coverage_geometry="polygon"`` measures
    against the outline instead.
    """
    from shapely.geometry import box as _box

    if coverage_geometry not in ("bbox", "polygon"):
        # Silently falling back to bbox would report percentages the caller did
        # not ask for, which is worse than failing.
        raise ValueError(
            f"coverage_geometry must be 'bbox' or 'polygon', got {coverage_geometry!r}."
        )

    if isinstance(polygon, (list, tuple)):
        bbox = [float(v) for v in polygon]
        return None, bbox, _box(*bbox)

    aoi_gdf = polygon_2_gdf(polygon)
    if aoi_gdf is None or aoi_gdf.empty:
        raise ValueError("Could not read any geometry from the polygon input.")
    bbox = [float(v) for v in aoi_gdf.total_bounds]
    if coverage_geometry == "polygon":
        return aoi_gdf, bbox, _union(list(aoi_gdf.geometry))
    return aoi_gdf, bbox, _box(*bbox)


def _union(geoms):
    """Union of WGS84 geometries, repairing invalid footprints if needed.

    A self-intersecting outline (rare, but published) makes unary_union raise;
    buffer(0) repairs it. Returns None only if nothing usable is left.
    """
    from shapely.ops import unary_union

    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return None
    try:
        return unary_union(geoms)
    except Exception:
        try:
            return unary_union([g.buffer(0) for g in geoms])
        except Exception:
            return None


def _eq_area(geom):
    """Geometry reprojected to the equal-area CRS used for fair area ratios.

    EPSG:6933, the same CRS the download path uses to compare AOI coverage
    between candidate projections. Never used for pixel data.
    """
    if geom is None or geom.is_empty:
        return None
    return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(_EQUAL_AREA).iloc[0]


def _share(geoms, aoi_eq):
    """Share (0..1) of the AOI covered by the union of WGS84 ``geoms``, or None."""
    return _share_eq([_eq_area(g) for g in geoms], aoi_eq)


def _share_eq(geoms_eq, aoi_eq):
    """Same, for geometries ALREADY in the equal-area CRS.

    Clamped to 1: an intersection cannot exceed the AOI, so a ratio of
    1.000000000017 (seen on the exact-outline path, where the AOI is itself the
    union of the footprints' own vertices) is floating-point noise, not coverage.
    """
    if aoi_eq is None or aoi_eq.area <= 0:
        return None
    merged = _union(geoms_eq)
    if merged is None:
        return None
    try:
        return min(1.0, float(merged.intersection(aoi_eq).area / aoi_eq.area))
    except Exception:
        return None


def _orbit_label(orbit):
    if orbit is None:
        # Not "Orbit unknown": for a static mosaic there IS no orbit, and for a
        # catalogue that simply does not publish one, saying so is the honest label.
        return "No orbit info"
    if isinstance(orbit, str):
        return orbit[:1].upper() + orbit[1:]  # Landsat "path 176" -> "Path 176"
    return f"Orbit {orbit}"


def _orbit_sort_key(orbit):
    """Numeric orbits before named ones, unknown last - a stable colour order."""
    if orbit is None:
        return (2, "")
    if isinstance(orbit, str):
        return (1, orbit)
    return (0, f"{orbit:06d}")


# Below this share of an area, an acquisition brings so little that it is worth
# leaving out of the download entirely: the timestep costs the full grid either
# way (see filter_items_by_footprint). Used both for the per-feature "grazing"
# count and for the builder's suggestion under Check area coverage, so the two
# can never drift apart.
GRAZE_THRESHOLD = 0.10


def summarize_per_feature_footprints(gdf, polygon, coverage_geometry="bbox"):
    """Per-FEATURE coverage summary for a multi-feature (batch) polygon file.

    A polygon file holding more than one feature is built as one cube PER
    feature (see get_stac_layers' batching branch), each against that feature's
    own bounding box. Measuring the preview against the combined extent of all
    features would therefore describe a cube that is never built: on a file of
    river reaches spread along a valley, the combined bbox is the whole valley,
    and orbits that only clip its far end look relevant while reaching no reach
    at all.

    This measures each feature separately, against exactly the geometry its own
    cube will use, reusing the footprints already fetched - so it costs shapely
    intersections only, no extra catalogue query.

    Returns
    -------
    (df, info)
        df : one row per feature - dates, orbits that reach it, and the average
             / best / worst share of it a single date holds.
        info : dict with ``n_features``, ``orbits_reaching_none`` (orbits that
               touch the combined extent but no individual feature - the exact
               false-alarm case this function exists to catch),
               ``features_with_grazing`` (areas holding at least one date under
               ``GRAZE_THRESHOLD``), ``features_never_full`` (areas no single
               date covers completely - not areas without data) and
               ``features_with_no_dates``.
    """
    features = polygon_2_features(polygon)
    n_features = len(features)

    # One reprojection of every footprint for the whole loop, not one per
    # feature: with 47 features that is the difference between ~1 s and a minute.
    if len(gdf):
        gdf = gdf.assign(_geom_eq=list(gdf.to_crs(_EQUAL_AREA).geometry))

    records = []
    orbits_reached = set()
    for pos, feature in enumerate(features, start=1):
        try:
            _, _, feat_geom = _aoi_geometries(feature, coverage_geometry)
            feat_eq = _eq_area(feat_geom)
        except Exception:
            feat_eq = None
        if feat_eq is None or feat_eq.area <= 0 or not len(gdf):
            records.append({"Area": pos, "Dates": 0, "Orbits": "-"})
            continue

        per_date, date_orbits = {}, {}
        for date, rows in gdf.groupby("date"):
            share = _share_eq(list(rows["_geom_eq"]), feat_eq)
            # A footprint that misses this feature entirely contributes 0, not a
            # date: only acquisitions that actually touch it become timesteps.
            if share is not None and share > 0:
                per_date[date] = share
                date_orbits[date] = {o for o in rows["orbit"]}

        shares = list(per_date.values())
        feat_orbits = sorted(
            {o for os_ in date_orbits.values() for o in os_}, key=_orbit_sort_key
        )
        orbits_reached.update(feat_orbits)
        records.append(
            {
                "Area": pos,
                "Dates": len(shares),
                "Orbits": ", ".join(str(_orbit_label(o)) for o in feat_orbits) or "-",
                "Average coverage %": _pct(_mean(shares)),
                "Best date %": _pct(max(shares)) if shares else None,
                "Worst date %": _pct(min(shares)) if shares else None,
                # Not a column: the count drives the summary line above the
                # table, where it is actionable, and a per-row copy of it only
                # widened a table that is already 47 rows long.
                "_grazing": sum(1 for s in shares if s < GRAZE_THRESHOLD),
            }
        )

    df = pd.DataFrame(records)
    n_grazing = 0
    if not df.empty:
        for col in ("Average coverage %", "Best date %", "Worst date %"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "_grazing" in df:
            n_grazing = int((df["_grazing"] > 0).sum())
            df = df.drop(columns=["_grazing"])
        df = df.set_index("Area")

    all_orbits = sorted({o for o in gdf["orbit"]} if len(gdf) else set(), key=_orbit_sort_key)
    info = {
        "n_features": n_features,
        "graze_threshold": GRAZE_THRESHOLD,
        "orbits_reaching_none": [o for o in all_orbits if o not in orbits_reached],
        "features_with_grazing": n_grazing,
        "features_never_full": (
            int((df["Best date %"] < 99.5).sum())
            if not df.empty and "Best date %" in df
            else 0
        ),
        "features_with_no_dates": (
            int((df["Dates"] == 0).sum()) if not df.empty else 0
        ),
    }
    return df, info


# --- pre-load footprint prefilter --------------------------------------------


def _solar_day(item):
    """The solar day odc-stac would group this item into, or None.

    Mirrors odc.stac's own rule exactly (``ParsedItem.solar_date`` ->
    ``_convert_to_solar_time``): the nominal timestamp shifted by the item
    centroid's longitude snapped to whole hours, then truncated to a date.
    Reproduced here rather than imported because it lives in odc's private
    model module - and a prefilter that grouped items differently from the
    loader could drop an item odc would have merged into a KEPT timestep,
    silently changing a date instead of removing one.
    """
    ts = getattr(item, "datetime", None)
    props = getattr(item, "properties", {}) or {}
    if ts is None:
        for key in ("start_datetime", "end_datetime"):
            raw = props.get(key)
            if raw:
                try:
                    ts = datetime.datetime.fromisoformat(
                        str(raw).replace("Z", "+00:00")
                    )
                except ValueError:
                    ts = None
                if ts is not None:
                    break
    if ts is None:
        return None

    geom = getattr(item, "geometry", None)
    if not geom:
        return ts.date()

    from shapely.geometry import shape as _shape

    try:
        lon = _shape(geom).centroid.x
    except Exception:
        return ts.date()
    return (ts + datetime.timedelta(seconds=int(lon / 15) * 3600)).date()


def filter_items_by_footprint(
    items,
    polygon,
    min_coverage,
    coverage_geometry="bbox",
    tile_handling="mosaic",
    q=False,
):
    """Drop scenes that barely overlap the AOI, BEFORE any pixel is read.

    The cheapest possible filter: it works on the STAC item outlines the search
    already returned, so a dropped acquisition is never loaded, never has its
    SCL read, and never gets a full-grid slab allocated in the cube. Every other
    scene filter in stac2cube runs on the BUILT cube and therefore shrinks the
    output without saving the work - this one saves the work.

    It exists for sprawling AOIs whose bounding box reaches far beyond the
    polygon: orbits that clip only a corner of that box contribute a date per
    revisit cycle, each costing a full-size, almost entirely NaN timestep.

    Parameters
    ----------
    items : list of pystac.Item
        The search result, before ``stac_load``.
    polygon : str | list | geopandas.GeoDataFrame
        The AOI the items were searched for.
    min_coverage : float
        Keep an acquisition when its outlines cover at least this fraction
        (0..1) of the AOI. ``None`` / ``0`` disables the filter entirely and
        returns the items untouched.
    coverage_geometry : {"bbox", "polygon"}
        What "the AOI" means, exactly as in :func:`summarize_scene_footprints`:
        the bounding box (what the cube is by default) or the exact outline
        (what it is when the caller clips).
    tile_handling : {"mosaic", "separate"}
        Must match the build's setting, because it decides what a timestep is:
        "mosaic" groups per solar day (odc's ``groupby="solar_day"``),
        "separate" per tile AND solar day.

    Returns
    -------
    (kept_items, info)
        ``info`` is None when the filter was disabled, else a dict with
        ``threshold``, ``n_before`` / ``n_after`` item counts, ``kept`` /
        ``dropped`` group counts and ``dropped_shares`` (group key -> fraction)
        so the caller can report exactly what went and why.

    Notes
    -----
    This is a GEOMETRIC estimate from published outlines, not a pixel
    measurement, and the catalogues do not agree to the last decimal (measured
    on one AOI: Planetary Computer and terrabyte identical, Element84 ~3.3
    percentage points lower on partial scenes). It is therefore meant for a LOW
    threshold - removing acquisitions that barely graze the area - and is not a
    substitute for :func:`stac2cube.clip.drop_partial_scenes`, which measures
    real pixels and is the only one of the two that can see nodata INSIDE an
    outline (a failed or incomplete granule reads as fully covering here).

    Items without a published outline or timestamp are always KEPT: there is
    nothing to judge them on, and dropping data on missing metadata would be
    the wrong way to fail.
    """
    items = list(items or [])
    if not items or not min_coverage:
        return items, None

    thr = float(min_coverage)
    if thr <= 0:
        return items, None
    if thr > 1:
        raise ValueError(
            f"min_footprint_coverage must be a fraction between 0 and 1, got {thr}."
        )

    _, _, aoi_geom = _aoi_geometries(polygon, coverage_geometry)
    aoi_eq = _eq_area(aoi_geom)
    if aoi_eq is None or aoi_eq.area <= 0:
        return items, None

    from collections import defaultdict

    from shapely.geometry import shape as _shape

    separate = str(tile_handling) == "separate"
    geoms, keys = [None] * len(items), [None] * len(items)
    for i, it in enumerate(items):
        geom = getattr(it, "geometry", None)
        day = _solar_day(it)
        if not geom or day is None:
            continue
        try:
            geoms[i] = _shape(geom)
        except Exception:
            continue
        keys[i] = (
            (_tile_of(getattr(it, "properties", {}) or {}), day) if separate else day
        )

    usable = [i for i, k in enumerate(keys) if k is not None and geoms[i] is not None]
    if not usable:
        return items, None

    # ONE reprojection for every footprint, then all the unions and area ratios
    # in projected space - the same optimisation summarize_scene_footprints
    # needed, for the same reason (per-group reprojection dominated the cost).
    eq = gpd.GeoSeries([geoms[i] for i in usable], crs="EPSG:4326").to_crs(_EQUAL_AREA)
    for j, i in enumerate(usable):
        geoms[i] = eq.iloc[j]

    groups = defaultdict(list)
    for i in usable:
        groups[keys[i]].append(i)

    shares = {k: _share_eq([geoms[i] for i in v], aoi_eq) for k, v in groups.items()}
    dropped = {k for k, s in shares.items() if s is not None and s < thr}

    if dropped and len(dropped) == len(groups):
        best = max((s for s in shares.values() if s is not None), default=0.0)
        raise ValueError(
            f"The footprint prefilter (min_footprint_coverage={thr:.2%}) would "
            f"drop every acquisition: the best one covers only {best:.2%} of the "
            "area. Lower the threshold, widen the date range, or check the area."
        )

    kept = [it for i, it in enumerate(items) if keys[i] is None or keys[i] not in dropped]
    info = {
        "threshold": thr,
        "coverage_geometry": coverage_geometry,
        "n_before": len(items),
        "n_after": len(kept),
        "groups": len(groups),
        "dropped": len(dropped),
        "kept": len(groups) - len(dropped),
        "dropped_shares": {str(k): shares[k] for k in sorted(dropped, key=str)},
    }
    if not q and dropped:
        print(
            f"Footprint prefilter: skipped {len(dropped)} of {len(groups)} "
            f"acquisitions covering less than {thr:.2%} of the area "
            f"({len(items) - len(kept)} of {len(items)} scenes not downloaded)."
        )
    return kept, info


def summarize_scene_footprints(gdf, polygon, coverage_geometry="bbox"):
    """
    Per-orbit coverage table for the footprints in ``gdf``.

    Parameters
    ----------
    gdf : GeoDataFrame
        Output of :func:`get_scene_footprints`.
    polygon : str | list | geopandas.GeoDataFrame
        The same area the footprints were queried for.
    coverage_geometry : {"bbox", "polygon"}
        What "the AOI" means when measuring coverage. ``"bbox"`` (default) matches
        the cube the builder returns by default; ``"polygon"`` matches a cube
        clipped to the exact outline.

    Returns
    -------
    (df, info)
        df : one row per orbit, indexed by orbit label: how many dates it brings,
             the average / best / worst share of the AOI a single date of that orbit
             covers, and the tiles and projections involved. Sorted by average
             coverage, descending.
        info : dict of context the caller can render - the queried windows, the
               per-date coverage series, ``orbit_stats`` (the same numbers as
               fractions), ``union_coverage`` (the ceiling: what ALL orbits
               together see), the tile and CRS lists, per-orbit colours, and
               ``notes``, a list of short plain-language findings.

    Notes
    -----
    Percentages are geometric estimates from generalised STAC outlines, not
    pixel-exact measurements. See the module docstring.
    """
    aoi_gdf, bbox, aoi_geom = _aoi_geometries(polygon, coverage_geometry)
    aoi_eq = _eq_area(aoi_geom)

    # Reproject every footprint to the equal-area CRS ONCE, then do all the unions
    # and area ratios below in projected space. Over a full date range this is
    # hundreds of dates, and reprojecting inside each per-date union made the
    # summary the slowest part of the preview by far.
    if len(gdf):
        gdf = gdf.assign(_geom_eq=list(gdf.to_crs(_EQUAL_AREA).geometry))

    orbits = sorted({o for o in gdf["orbit"]} if len(gdf) else set(), key=_orbit_sort_key)
    colors = {o: _ORBIT_COLORS[i % len(_ORBIT_COLORS)] for i, o in enumerate(orbits)}

    # Per acquisition date: how much of the AOI that day's scenes hold together.
    # This is the number that decides whether a timestep is a usable full scene
    # or an across-track sliver.
    per_date, date_orbits = {}, {}
    for date, rows in gdf.groupby("date") if len(gdf) else []:
        per_date[date] = _share_eq(list(rows["_geom_eq"]), aoi_eq)
        date_orbits[date] = sorted({o for o in rows["orbit"]}, key=_orbit_sort_key)

    records, orbit_stats = [], {}
    for orbit in orbits:
        rows = gdf[gdf["orbit"].isna() if orbit is None else gdf["orbit"] == orbit]
        dates = sorted(set(rows["date"]))
        # Measured per date WITHIN this orbit, not read off the all-orbit per_date
        # series above: two different orbits can image one AOI on the same day
        # (adjacent tracks, or an AOI wide enough to catch both), and crediting
        # each of them with that day's combined figure would overstate what either
        # one sees on its own.
        by_date = dict(list(rows.groupby("date")["_geom_eq"]))
        shares = [
            s
            for s in (
                _share_eq(list(by_date[d]), aoi_eq) for d in dates
            )
            if s is not None
        ]
        stats = {
            "dates": len(dates),
            "average": _mean(shares),
            "best": max(shares) if shares else None,
            "worst": min(shares) if shares else None,
            "tiles": sorted({t for t in rows["tile"] if t}),
            "crs": sorted({c for c in rows["crs"] if c}),
        }
        orbit_stats[orbit] = stats
        records.append(
            {
                "Orbit": _orbit_label(orbit),
                # Dates carries real information now that the default reads the
                # whole date range: it is how many timesteps this orbit contributes
                # to a cube over these dates, and the sample size behind the three
                # coverage columns. (It was dropped while the tool read a fixed
                # 12-day window, where it could only ever be 2 or 3 and so said
                # more about the window than about the area.)
                "Dates": stats["dates"],
                "Average AOI coverage %": _pct(stats["average"]),
                "Best date %": _pct(stats["best"]),
                "Worst date %": _pct(stats["worst"]),
                # "-" not None: a mission without a tiling grid (Sentinel-1) or
                # without a published CRS would otherwise render as "None" in the
                # GUI's HTML table.
                "Tiles": ", ".join(stats["tiles"]) or "-",
                "CRS": ", ".join(stats["crs"]) or "-",
                "_orbit": orbit,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        # Percent columns must be numeric, not object: a None in an object column
        # makes sort_values raise instead of sorting, and pandas renders a real
        # NaN as "NaN" in the GUI table the way the availability table does.
        for col in ("Average AOI coverage %", "Best date %", "Worst date %"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values(
            "Average AOI coverage %", ascending=False, na_position="last"
        )
        df = df.set_index("Orbit").drop(columns=["_orbit"])

    tiles = sorted({t for t in gdf["tile"] if t}) if len(gdf) else []
    crs_list = sorted({c for c in gdf["crs"] if c}) if len(gdf) else []
    union_coverage = _share(list(gdf.geometry), aoi_eq) if len(gdf) else None

    info = {
        "n_scenes": int(len(gdf)),
        "n_dates": len(per_date),
        "orbits": orbits,
        "orbit_labels": {o: _orbit_label(o) for o in orbits},
        "orbit_colors": colors,
        "orbit_stats": orbit_stats,
        "tiles": tiles,
        "crs": crs_list,
        "multi_crs": len(crs_list) > 1,
        "bbox": bbox,
        "coverage_geometry": coverage_geometry,
        "union_coverage": union_coverage,
        "per_date_coverage": dict(sorted(per_date.items())),
        "per_date_orbits": date_orbits,
        "antimeridian": bool(bbox[2] - bbox[0] > 180),
        "notes": [],
    }
    info["notes"] = _footprint_notes(info)
    return df, info


def _pct(share):
    """Share (0..1) as a percentage, to 2 decimals.

    Two decimals rather than one because swath-edge granules really do clip an
    AOI by a few hundredths of a percent (measured: 0.03% for orbit 5 over the
    Naryn AOI), and rounding those to "0.0%" would report a footprint that exists
    as no coverage at all. Rounding up to a floor like 0.1 instead would be
    inventing a number.
    """
    return None if share is None else round(100.0 * share, 2)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _footprint_notes(info, partial_threshold=0.9):
    """Short plain-language findings for the GUI to show under the map.

    Each note states a fact the user can act on in the builder (tile handling,
    partial-scene removal, target CRS), in one line. Deliberately terse: detail
    belongs in the tool's "?" help panel, not here.
    """
    notes = []
    shares = [v for v in info["per_date_coverage"].values() if v is not None]
    area = "bounding box" if info["coverage_geometry"] == "bbox" else "polygon"

    if info["antimeridian"]:
        notes.append(
            "Your area crosses the 180th meridian, so the coverage percentages "
            "below cannot be trusted. The footprints on the map are still correct."
        )

    if info["n_scenes"] == 0:
        notes.append("No scenes were found for this area in the probe window.")
        return notes

    if info.get("static_mission"):
        notes.append(
            "This mission is a static mosaic, so it has no orbits and no "
            "acquisition dates. The map shows the tiles that cover your area."
        )
        if len(info["tiles"]) > 1:
            notes.append(
                f"Your area spans {len(info['tiles'])} tiles, which are "
                "mosaicked into one cube."
            )
        return notes

    if info["union_coverage"] is not None and info["union_coverage"] < 0.995:
        notes.append(
            f"All orbits together cover {_pct(info['union_coverage'])}% of your "
            f"{area}. The rest is never imaged and will be empty in the cube."
        )
    if shares and max(shares) < 0.995:
        notes.append(
            f"No single date covers your whole {area} (best date: "
            f"{_pct(max(shares))}%). Every timestep will be partial."
        )
    n_partial = sum(1 for v in shares if v < partial_threshold)
    if n_partial:
        notes.append(
            f"{n_partial} of {len(shares)} dates cover less than "
            f"{int(partial_threshold * 100)}% of your {area} (swath-edge scenes). "
            "\"Remove partial scenes\" can drop these."
        )
    if len(info["tiles"]) > 1:
        notes.append(
            f"Your area spans {len(info['tiles'])} tiles "
            f"({', '.join(info['tiles'])}), which are mosaicked into one cube."
        )
    if info["multi_crs"]:
        notes.append(
            f"The scenes arrive in {len(info['crs'])} projections "
            f"({', '.join(info['crs'])}). One is picked as the cube's CRS and the "
            "others are reprojected into it."
        )
    if len(info["orbits"]) > 1:
        notes.append(
            f"{len(info['orbits'])} orbits pass over your area, so acquisition "
            "geometry (view angle, time of day) differs between dates."
        )
    # The one case where the sample size behind the averages is worth saying out
    # loud: a single date means average, best and worst are all that one date, so
    # the table shows no spread even if the orbit's coverage does vary.
    singles = [
        info["orbit_labels"].get(o, _orbit_label(o))
        for o, s in info.get("orbit_stats", {}).items()
        if s.get("dates") == 1
    ]
    if singles:
        who = (
            ", ".join(singles)
            if len(singles) <= 3
            else f"{len(singles)} orbits"
        )
        notes.append(
            f"{who} only had one date in this window, so the average, best and "
            "worst columns are that single date."
        )
    return notes


# --- map ---------------------------------------------------------------------


def satellite_map(center=(20.0, 0.0), zoom=2, height="60vh", draw=False):
    """
    An empty leafmap map with the satellite + place-names background.

    The same hand-stacked "hybrid" background the builder's polygon drawing map
    uses: Esri satellite imagery at the bottom, a labels-only tile on top. Built
    by hand because leafmap's own "HYBRID" basemap needs a GOOGLE_MAPS_API_KEY and
    silently falls back to label-free Esri imagery without one, and because a full
    OpenStreetMap tile is opaque and would wash the imagery out.

    ``draw=False`` removes the drawing tools: on a preview map they are noise, and
    a shape drawn here would go nowhere.

    Returns None when leafmap is not installed, so callers can degrade gracefully
    instead of raising.
    """
    try:
        import leafmap
    except Exception:
        return None

    try:
        m = leafmap.Map(center=list(center), zoom=zoom, draw_control=draw)
    except Exception:
        m = leafmap.Map(center=list(center), zoom=zoom)
    m.layout.height = height
    m.layout.min_height = "320px"
    m.layout.max_height = "640px"
    m.layout.width = "100%"
    try:
        # Layers render in the order they are added, so drop leafmap's default OSM
        # base layer first: the imagery replaces it.
        for layer in list(m.layers):
            if getattr(layer, "name", "") == "OpenStreetMap":
                m.remove(layer)
        m.add_basemap("Esri.WorldImagery")             # background
        m.add_basemap("CartoDB.DarkMatterOnlyLabels")  # labels only
    except Exception:
        pass
    return m


def footprint_map(
    gdf,
    polygon,
    info=None,
    coverage_geometry="bbox",
    height="60vh",
    show_tiles=True,
    tile_labels=True,
    legend=True,
):
    """
    Draw the footprints of ``gdf`` over the AOI on a satellite map.

    Layers, bottom to top: one filled polygon per orbit (the union of that orbit's
    footprints, coloured, click for its numbers), the tiling grid as thin dashed
    outlines with tile labels, the AOI bounding box (yellow dashed), and the exact
    AOI outline (red). Every layer is named, so the map's layer control can toggle
    each orbit on and off - which is how a user sees what one orbit alone misses.

    ``info`` is the dict from :func:`summarize_scene_footprints`; pass it to keep
    the colours and percentages identical to the table. It is recomputed if
    omitted.

    Returns the leafmap Map, or None when leafmap is not installed.
    """
    if info is None:
        _, info = summarize_scene_footprints(gdf, polygon, coverage_geometry)

    # Only the outline and the bbox are needed here: every percentage shown on the
    # map comes from info, measured against whichever area the summary used.
    aoi_gdf, bbox, _aoi_geom = _aoi_geometries(
        polygon, info.get("coverage_geometry", coverage_geometry)
    )
    center = (0.5 * (bbox[1] + bbox[3]), 0.5 * (bbox[0] + bbox[2]))
    m = satellite_map(center=center, zoom=8, height=height)
    if m is None:
        return None

    orbits = list(info.get("orbits", []))
    if orbits == [None]:
        # Nothing to group by (a static mosaic, or a catalogue that publishes no
        # orbit): show the footprints themselves rather than one union labelled
        # "No orbit info", which would hide how many there are.
        plain = gdf[["date", "tile", "crs", "geometry"]].rename(
            columns={"date": "Date", "tile": "Tile", "crs": "CRS"}
        )
        m.add_gdf(
            gpd.GeoDataFrame(plain, geometry="geometry", crs="EPSG:4326"),
            layer_name=f"Footprints ({len(gdf)})",
            style={
                "color": _ORBIT_COLORS[0],
                "weight": 2,
                "fillColor": _ORBIT_COLORS[0],
                "fillOpacity": 0.22,
            },
            hover_style={"fillOpacity": 0.45, "weight": 3},
            info_mode="on_click",
        )
        orbits = []

    for orbit in orbits:
        rows = gdf[gdf["orbit"].isna() if orbit is None else gdf["orbit"] == orbit]
        merged = _union(list(rows.geometry))
        if merged is None:
            continue
        dates = sorted(set(rows["date"]))
        color = info["orbit_colors"].get(orbit, _ORBIT_COLORS[0])
        label = info["orbit_labels"].get(orbit, _orbit_label(orbit))
        # Read straight out of the summary rather than recomputed, so a popup and
        # its table row can never disagree.
        stats = info.get("orbit_stats", {}).get(orbit, {})
        # One-row GeoDataFrame per orbit: add_gdf shows its columns in the popup,
        # so these ARE the popup contents.
        layer_gdf = gpd.GeoDataFrame(
            [
                {
                    "Orbit": label,
                    "Dates": stats.get("dates", len(dates)),
                    "Average AOI coverage %": _pct(stats.get("average")),
                    "Best date %": _pct(stats.get("best")),
                    "Worst date %": _pct(stats.get("worst")),
                    "Tiles": ", ".join(stats.get("tiles") or []) or "-",
                    "CRS": ", ".join(stats.get("crs") or []) or "-",
                    "geometry": merged,
                }
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
        n_dates = stats.get("dates", len(dates))
        m.add_gdf(
            layer_gdf,
            layer_name=f"{label} ({n_dates} date{'' if n_dates == 1 else 's'})",
            style={
                "color": color,
                "weight": 2,
                "fillColor": color,
                "fillOpacity": 0.22,
            },
            hover_style={"fillOpacity": 0.45, "weight": 3},
            info_mode="on_click",
        )

    tiles = [t for t in info.get("tiles", [])]
    if show_tiles and tiles:
        tile_rows = []
        for tile in tiles:
            rows = gdf[gdf["tile"] == tile]
            merged = _union(list(rows.geometry))
            if merged is None:
                continue
            tile_rows.append(
                {
                    "Tile": tile,
                    "CRS": ", ".join(sorted({c for c in rows["crs"] if c})) or "-",
                    "Orbits": ", ".join(
                        _orbit_label(o)
                        for o in sorted(
                            {o for o in rows["orbit"]}, key=_orbit_sort_key
                        )
                    ),
                    "geometry": merged,
                }
            )
        if tile_rows:
            tile_gdf = gpd.GeoDataFrame(tile_rows, geometry="geometry", crs="EPSG:4326")
            m.add_gdf(
                tile_gdf,
                layer_name="Tiling grid",
                style=_TILE_STYLE,
                hover_style={"weight": 2, "fillOpacity": 0.05},
                info_mode="on_click",
            )
            if tile_labels:
                # Labels sit at the footprint centroids. Wrapped: add_labels is a
                # convenience layer and a failure here must not cost the map.
                try:
                    centroids = tile_gdf.geometry.centroid
                    m.add_labels(
                        pd.DataFrame(
                            {
                                "Tile": tile_gdf["Tile"],
                                "longitude": centroids.x,
                                "latitude": centroids.y,
                            }
                        ),
                        column="Tile",
                        font_size="11pt",
                        font_color="#ffffff",
                        font_weight="bold",
                        draggable=False,
                        layer_name="Tile labels",
                    )
                except Exception:
                    pass

    m.add_gdf(
        gpd.GeoDataFrame(
            [{"Area": "Bounding box of your area", "geometry": _box_geom(bbox)}],
            geometry="geometry",
            crs="EPSG:4326",
        ),
        layer_name="Your area (bounding box)",
        style=_BBOX_STYLE,
        hover_style={"weight": 3},
        info_mode="on_click",
    )
    if aoi_gdf is not None:
        # Only the geometry: an input file can carry any number of attribute
        # columns, and dumping them into a popup is noise here.
        m.add_gdf(
            gpd.GeoDataFrame(geometry=aoi_gdf.geometry.values, crs=aoi_gdf.crs),
            layer_name="Your area (outline)",
            style=_AOI_STYLE,
            hover_style={"weight": 4},
            info_mode=None,
        )

    if legend and orbits:
        try:
            m.add_legend(
                title="Relative orbit",
                legend_dict={
                    info["orbit_labels"].get(o, _orbit_label(o)): info["orbit_colors"][o]
                    for o in orbits
                },
                position="bottomright",
            )
        except Exception:
            pass

    try:
        # [[south, west], [north, east]], the order ipyleaflet expects.
        m.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])
    except Exception:
        pass
    return m


def _box_geom(bbox):
    from shapely.geometry import box as _box

    return _box(*[float(v) for v in bbox])


# ---------------------------------------------------------------------------
# Statistics table of a built cube (CSV)
# ---------------------------------------------------------------------------
#
# One long table with a `period` column: every row is one band over one period
# (a single date, a year, or a month), and the five statistic columns always
# mean the same thing. Everything is computed from the cube's OWN time series,
# never from stored composite layers and never from the per-date rows of the
# table itself - see the note in _row_statistics for why that distinction is
# not cosmetic.

_STAT_OPS = ("mean", "median", "min", "max", "std")

# Written out rather than taken from `calendar`, whose month names follow the
# process locale: a CSV must not change its labels with the machine it ran on.
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# Coordinates copied into the table when the cube carries them. Both are
# per-scene (time,) values: reported as stored for a date row, averaged over
# the scenes of a year / month row.
_CONTEXT_COORDS = ("cloud_percentage", "scene_coverage")


def _is_categorical_band(name):
    """Band whose values are class codes, not a measured quantity.

    A mean of SCL class numbers or of a 0/1 cloud mask is arithmetically valid
    and physically meaningless, so these are flagged rather than dropped - the
    caller may well want the mask's mean (its cloud fraction) on purpose.
    """
    n = str(name).lower()
    return n in {"scl", "qa", "qa60", "qa_pixel"} or n.startswith("cloud_mask")


def _cube_time_series(cube):
    """Resolve the input to (time series DataArray, dataset to close).

    Accepts a path to an exported cube (``.nc`` / ``.zarr``, opened through
    :func:`stac2cube.export_cfg.open_cube` so legacy variable names migrate),
    an already-open Dataset, or the time-series DataArray itself.
    """
    to_close = None
    if isinstance(cube, xr.DataArray):
        stac = cube
    elif isinstance(cube, xr.Dataset):
        stac = _time_series_var(cube)
    elif isinstance(cube, (str, os.PathLike)):
        if not os.path.exists(cube):
            raise FileNotFoundError(f"No cube at {cube}.")
        # "frames" = lazy, one chunk per scene (NetCDF); Zarr stores are
        # already written that way, so the flag is a no-op there.
        ds = open_cube(cube, chunks="frames")
        to_close = ds
        try:
            stac = _time_series_var(ds)
        except Exception:
            ds.close()
            raise
    else:
        raise TypeError(
            "cube must be a path to an exported .nc / .zarr cube, an "
            f"xarray.Dataset or an xarray.DataArray, got {type(cube).__name__}."
        )

    if "time" not in stac.dims:
        raise ValueError(
            "This cube has no time dimension, so it holds temporal composites "
            "only. A statistics table is built from the time series - rebuild "
            "the cube with the time series kept (keep_timeseries=True)."
        )
    if stac.sizes["time"] == 0:
        raise ValueError("This cube's time dimension is empty - no dates to report.")
    return stac, to_close


def _time_series_var(ds):
    """The time-series variable of a cube Dataset, or a helpful error."""
    if "Time_Series" in ds.data_vars:
        return ds["Time_Series"]
    others = [str(v) for v in ds.data_vars if str(v) != "spatial_ref"]
    if others:
        raise ValueError(
            "This cube holds no 'Time_Series' variable, only "
            f"{others[:8]}{' ...' if len(others) > 8 else ''}. Those are "
            "temporal composites; a statistics table needs the time series "
            "the composites were reduced from."
        )
    raise ValueError("This cube holds no data variables.")


def _period_rows(times):
    """Row specification of the whole table.

    Returns ``[(period, label, indices), ...]``: every date of the cube in
    chronological order, then one row per year, then one per month. Positions
    are taken by index rather than by slicing the time axis, so an unsorted
    time coordinate (update mode, tile_handling="separate") is handled without
    reordering the cube.
    """
    order = np.argsort(times.values)
    rows = [
        ("date", times[i].strftime("%Y-%m-%d"), np.array([i]))
        for i in order
    ]
    years = times.year.values
    for y in sorted(set(int(v) for v in years)):
        rows.append(("year", str(y), np.flatnonzero(years == y)))
    months = years * 100 + times.month.values
    for key in sorted(set(int(v) for v in months)):
        y, m = key // 100, key % 100
        rows.append(("month", f"{_MONTH_NAMES[m - 1]}_{y}", np.flatnonzero(months == key)))
    return rows


def _row_statistics(sub, ops):
    """The requested statistics of one row, per band.

    Reduced over every dimension except ``band`` - i.e. over the pixels of the
    scenes in this period, all at once. For a year or a month that is NOT the
    same as summarising the date rows above it: a mean of per-date means
    weights a date with 5% valid pixels like one with 100%, and a median or a
    standard deviation of per-date values is a different quantity altogether
    (spread BETWEEN dates rather than in the data). Only min and max come out
    identical either way.
    """
    dims = [d for d in sub.dims if d != "band"]
    with warnings.catch_warnings():
        # All-NaN period (a fully clouded date) -> NaN, written as an empty
        # cell. numpy's "All-NaN slice"/"Mean of empty slice" notices are the
        # expected path here, not a problem to report.
        warnings.simplefilter("ignore", RuntimeWarning)
        reduced = xr.Dataset(
            {op: getattr(sub, op)(dim=dims, skipna=True) for op in ops}
        ).compute()
    return {
        op: np.atleast_1d(np.asarray(reduced[op].values, dtype="float64"))
        for op in ops
    }


def _coord_value(sub, name):
    """Mean of a per-scene coordinate over the scenes of one row (NaN if none).

    A cube built lazily can carry cloud_percentage / scene_coverage as deferred
    coordinates; reading them here materialises them, which is the point.
    """
    if name not in sub.coords:
        return None
    values = np.atleast_1d(np.asarray(sub[name].values, dtype="float64"))
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(values))


def _default_csv_path(source):
    p = Path(source)
    name = p.name
    for ext in (".zarr", ".nc4", ".nc", ".netcdf"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return str(p.parent / f"{name}_statistics.csv")


def export_cube_statistics(
    cube,
    csv_path=None,
    bands=None,
    stats=None,
    decimals=6,
    q=False,
):
    """
    Statistics table of an exported data cube, written as one CSV.

    Every row is one band over one period; the ``period`` column says which
    kind of period the ``label`` names:

    ==========  ==============  ==================================
    period      label           rows
    ==========  ==============  ==================================
    ``date``    ``2024-04-01``  every date the cube holds
    ``year``    ``2024``        every year present
    ``month``   ``April_2024``  every month present
    ==========  ==============  ==================================

    Columns: ``period, label, band, n_dates, mean, median, min, max, std``,
    followed by ``cloud_percentage`` and ``scene_coverage`` where the cube
    carries them, and by ``tile`` for a cube built with
    ``tile_handling="separate"``.

    All three sections are computed from the cube's ``Time_Series`` variable,
    reduced over the pixels of the period's scenes. The year and month rows are
    NOT summaries of the date rows and not read from stored composite layers,
    so they hold for a cube that carries no composites at all - and they stay
    correct when the number of valid pixels varies from date to date, which
    cloud masking guarantees it does. ``std`` is the population standard
    deviation (ddof=0), xarray's default.

    A cube built without its time series (temporal composites only, no time
    dimension) cannot produce this table and raises.

    Parameters
    ----------
    cube : str | Path | xr.Dataset | xr.DataArray
        An exported ``.nc`` / ``.zarr`` cube, or an already-open one.
    csv_path : str, optional
        Where to write. Defaults to ``<cube>_statistics.csv`` next to the cube;
        pass ``None`` with an in-memory cube to skip writing and only get the
        table back.
    bands : str | list of str, optional
        Restrict to these bands (default: all, in the cube's band order).
    stats : list of str, optional
        Subset of ``mean, median, min, max, std`` (default: all five).
    decimals : int, optional
        Rounding of the numeric columns; ``None`` writes full precision. The
        default of 6 is already beyond the ~7 significant digits a float32 cube
        actually carries.
    q : bool
        Quiet: suppress the summary print.

    Returns
    -------
    pandas.DataFrame
        The same table that was written.

    Notes
    -----
    Memory: each row is reduced from the scenes of its period, one period at a
    time. ``median`` cannot be streamed - it needs the whole period in memory -
    so the peak is roughly the largest year of the cube. Drop it
    (``stats=["mean", "min", "max", "std"]``) to keep the whole run streaming
    scene by scene.

    ``n_dates`` counts the timesteps the period holds - including a fully
    clouded one that contributed no pixel - and for a
    ``tile_handling="separate"`` cube counts tiles, not solar days.

    Examples
    --------
    >>> df = export_cube_statistics("results/naryn.nc")
    >>> df = export_cube_statistics("results/naryn.zarr", bands=["ndvi"],
    ...                             stats=["mean", "std"])
    """
    ops = list(_STAT_OPS) if stats is None else [str(s).strip().lower() for s in
                                                 ([stats] if isinstance(stats, str) else stats)]
    unknown = [op for op in ops if op not in _STAT_OPS]
    if unknown:
        raise ValueError(
            f"Unsupported statistic(s) {unknown}. Available: {list(_STAT_OPS)}."
        )
    if not ops:
        raise ValueError("stats is empty - nothing to compute.")

    stac, to_close = _cube_time_series(cube)
    try:
        # A cube with a single band may carry it as a scalar coordinate or not
        # at all; promoting it to a length-1 dimension keeps one code path.
        if "band" not in stac.dims:
            name = str(stac["band"].values) if "band" in stac.coords else "band"
            stac = stac.expand_dims(band=[name])
        band_names = [str(b) for b in np.atleast_1d(stac["band"].values)]

        if bands is not None:
            wanted = [bands] if isinstance(bands, str) else [str(b) for b in bands]
            missing = [b for b in wanted if b not in band_names]
            if missing:
                raise ValueError(
                    f"Band(s) {missing} are not in this cube. Available: {band_names}."
                )
            stac = stac.sel(band=wanted)
            band_names = wanted

        categorical = [b for b in band_names if _is_categorical_band(b)]
        has_tile = "tile" in stac.coords and "time" in stac["tile"].dims
        context = [c for c in _CONTEXT_COORDS if c in stac.coords]

        times = pd.to_datetime(stac["time"].values)
        rows = _period_rows(times)
        # median has to hold a whole period at once (no streaming quantile);
        # without it every reduction is a running one and the cube can stay
        # lazy, one scene in memory at a time.
        materialize = "median" in ops

        records = []
        for period, label, idx in rows:
            sub = stac.isel(time=idx)
            if materialize:
                sub = sub.compute()
            values = _row_statistics(sub, ops)
            shared = {
                "period": period,
                "label": label,
                "n_dates": int(idx.size),
            }
            if has_tile:
                # Blank for year / month rows: they span whichever tiles the
                # period holds, so no single tile id describes them.
                shared["tile"] = (
                    str(np.atleast_1d(sub["tile"].values)[0]) if period == "date" else ""
                )
            ctx = {name: _coord_value(sub, name) for name in context}
            for i, band in enumerate(band_names):
                records.append({
                    **shared,
                    "band": band,
                    **{op: float(values[op][i]) for op in ops},
                    **ctx,
                })

        columns = ["period", "label", "band"]
        if has_tile:
            columns.append("tile")
        columns += ["n_dates"] + ops + context
        df = pd.DataFrame.from_records(records, columns=columns)
        if decimals is not None:
            numeric = [c for c in ops + context]
            df[numeric] = df[numeric].round(int(decimals))

        out_path = csv_path
        if out_path is None and isinstance(cube, (str, os.PathLike)):
            out_path = _default_csv_path(cube)
        if out_path is not None:
            df.to_csv(out_path, index=False)

        if not q:
            n_dates = sum(1 for p, _, _ in rows if p == "date")
            n_years = sum(1 for p, _, _ in rows if p == "year")
            n_months = sum(1 for p, _, _ in rows if p == "month")
            print(
                f"Statistics: {len(df)} rows - {n_dates} dates, {n_years} year(s), "
                f"{n_months} month(s) x {len(band_names)} band(s)."
            )
            if categorical:
                print(
                    f"  note: {', '.join(categorical)} carry class codes / "
                    "flags, so those rows describe category numbers, not "
                    "reflectance."
                )
            if out_path is not None:
                print(f"  written to {out_path}")
        return df
    finally:
        if to_close is not None:
            to_close.close()


def preview_scene_footprints(
    mission,
    polygon,
    source="element84",
    daterange=None,
    window_days=None,
    coverage_geometry="bbox",
    height="60vh",
    q=False,
):
    """
    Map + table of the satellite footprints that cover an area of interest.

    Metadata only, no pixels read. See the module docstring for what the footprints
    are and how far the percentages can be trusted.

    Parameters
    ----------
    mission, polygon, source, daterange, window_days
        As in :func:`get_scene_footprints`.
    coverage_geometry : {"bbox", "polygon"}
        What coverage is measured against, as in :func:`summarize_scene_footprints`.
    height : str
        CSS height of the map.
    q : bool
        Quiet: suppress the progress print.

    Returns
    -------
    (m, df, info)
        m : leafmap Map, ready to display (None if leafmap is not installed - the
            table and info are still returned).
        df : per-orbit coverage table, see :func:`summarize_scene_footprints`.
        info : context dict, including ``notes`` (short findings), ``windows`` (the
               date windows queried) and ``window`` (their overall span).

    Examples
    --------
    >>> m, df, info = preview_scene_footprints(
    ...     "sentinel_2_l2a", "polygons/naryn.gpkg", daterange=["2024-06-01", "2024-08-31"]
    ... )
    >>> df
    >>> m
    """
    gdf, windows = get_scene_footprints(
        mission, polygon, source=source, daterange=daterange,
        window_days=window_days, q=q,
    )
    df, info = summarize_scene_footprints(gdf, polygon, coverage_geometry)
    info["windows"] = windows
    # Overall span, for a caption. A seasonal range queries several windows, so this
    # is the outer envelope, not a claim that every date inside it was read.
    info["window"] = [windows[0][0], windows[-1][1]] if windows else None
    info["mission"] = mission
    info["source"] = source
    info["static_mission"] = mission in _STATIC_MISSIONS
    info["footprints"] = gdf  # kept so a caller can build other views without re-querying
    # A multi-feature file is built as one cube PER feature, each against its own
    # bounding box, so the union numbers above describe a cube nobody ever gets:
    # on a file of river reaches the union is the whole valley, and an orbit that
    # only clips its far end looks relevant while reaching no reach at all.
    # Measured here from the SAME footprints (no extra query) and handed over so
    # the caller can show the per-feature view instead.
    info["per_feature"], info["per_feature_info"] = None, None
    if len(gdf) and not info["static_mission"]:
        try:
            _fdf, _finfo = summarize_per_feature_footprints(
                gdf, polygon, coverage_geometry
            )
        except Exception:
            # A per-feature failure must not take the whole preview down: the
            # union view above is still valid and still worth showing.
            _fdf = _finfo = None
        if _finfo and _finfo.get("n_features", 1) > 1:
            info["per_feature"], info["per_feature_info"] = _fdf, _finfo
    info["notes"] = _footprint_notes(info)  # re-run: static_mission is known only here
    # Drawn even with no footprints: the user still gets their area on the map,
    # which is the answer to "did I really select what I think I did?".
    m = footprint_map(gdf, polygon, info=info, height=height)
    return m, df, info
