from .vector_refiner import polygon_2_bbox
from .cdse_auth import configure_cdse_environment, configure_anonymous_aws_environment
from .stac_processing import (
    expand_season_windows as _expand_season_windows,
    is_mmdd as _is_mmdd,
)

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
from collections import Counter, defaultdict

# Equal-area CRS used only to compare AOI coverage fairly between candidate
# projections (see _native_crs_area_shares). Never used for pixel data.
_EQUAL_AREA = "EPSG:6933"


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


def crs_attr_string(crs_like):
    """Machine-readable CRS string for a cube's ``attrs["crs"]``.

    Returns ``"EPSG:<code>"`` whenever the CRS is registered, otherwise its WKT.
    Accepts anything CRS-like: an ``"EPSG:32632"`` string, a bare code, a pyproj
    or rasterio CRS object.

    Why not the CRS *name*: cubes used to store ``spatial_ref.projected_crs_name``
    (e.g. "WGS 84 / UTM zone 32N"), which only round-trips for CRSs that happen to
    be in the PROJ name database. A custom PROJ string resolves to "unknown" and
    cannot be read back, and a geographic CRS has no ``projected_crs_name``
    attribute at all. An EPSG code (or WKT) always round-trips.
    """
    if crs_like is None:
        return None

    # pyproj / rasterio CRS objects expose to_epsg() directly.
    to_epsg = getattr(crs_like, "to_epsg", None)
    if callable(to_epsg):
        code = to_epsg()
        if code:
            return f"EPSG:{code}"
        to_wkt = getattr(crs_like, "to_wkt", None)
        return to_wkt() if callable(to_wkt) else str(crs_like)

    # Bare EPSG codes, as int or as a digit string (proj:epsg is an int).
    if isinstance(crs_like, int):
        return f"EPSG:{crs_like}"
    text = str(crs_like).strip()
    if text.isdigit():
        return f"EPSG:{text}"

    # pyproj, not rasterio: rasterio cannot parse a CRS *name*, and legacy cubes
    # stored exactly that ("WGS 84 / UTM zone 32N"), so reading one back must work.
    from pyproj import CRS as _PyprojCRS

    crs = _PyprojCRS.from_user_input(text)
    code = crs.to_epsg()
    return f"EPSG:{code}" if code is not None else crs.to_wkt()


def validate_target_crs(crs_like):
    """Canonicalise a user-requested target CRS, or raise with the reason.

    Rejects anything not measured in metres. ``resolution`` is interpreted by
    odc-stac in the units of the OUTPUT CRS, so a geographic CRS would silently
    turn ``resolution=10`` into 10-DEGREE pixels, and the shadow projection reads
    the pixel size as ground metres. Both fail quietly rather than loudly, which
    is why this is a hard error instead of a warning.
    """
    from pyproj import CRS as _PyprojCRS

    # Both steps must be inside the guard. crs_attr_string is a FORMATTER, not a
    # validator: a bare digit string short-circuits to "EPSG:<digits>" without
    # ever being looked up, so an incomplete code like "3" formats fine and only
    # fails here. Leaving this line outside let pyproj's CRSError escape raw
    # instead of becoming the friendly ValueError callers expect.
    try:
        target = crs_attr_string(crs_like)
        crs = _PyprojCRS.from_user_input(target)
    except Exception as exc:
        raise ValueError(
            f"crs={crs_like!r} is not a CRS that pyproj can read. Use an EPSG "
            f'code (e.g. crs="EPSG:3035"), a WKT string, or a PROJ string. '
            f"({type(exc).__name__}: {exc})"
        )

    if crs.is_geographic:
        raise ValueError(
            f"crs={crs_like!r} is a geographic (degree-based) CRS. stac2cube "
            "builds cubes on a metric grid: resolution is given in CRS units, so "
            "a geographic CRS would request degree-sized pixels. Use a projected, "
            'metre-based CRS instead (e.g. crs="EPSG:3035").'
        )
    units = {ax.unit_name for ax in crs.axis_info[:2]}
    if not units <= {"metre", "meter"}:
        raise ValueError(
            f"crs={crs_like!r} uses {sorted(units)} as its axis unit. stac2cube "
            "requires a metre-based CRS: resolution is expressed in CRS units and "
            "the cloud-shadow projection treats the pixel size as ground metres."
        )
    return target


def _item_crs(item):
    """EPSG string of a STAC item's pixels, or None when it declares none.

    element84/PC/terrabyte publish ``proj:code`` (or the older ``proj:epsg``) in
    item.properties; CDSE keeps it at the ASSET level instead, so fall back to the
    first asset that carries one.
    """
    p = item.properties
    crs = p.get("proj:code") or p.get("proj:epsg")
    if crs is None:
        for asset in item.assets.values():
            crs = asset.extra_fields.get("proj:code") or asset.extra_fields.get(
                "proj:epsg"
            )
            if crs is not None:
                break
    if crs is None:
        return None
    # Canonicalise to "EPSG:<code>". proj:code is already that, proj:epsg is a
    # bare code (int, or a string on some servers) - without this they would be
    # two different Counter keys for ONE projection and fake a multi-CRS area.
    # Kept branchy rather than routed through crs_attr_string: this runs per item
    # and the common cases must not pay for a PROJ database lookup.
    if isinstance(crs, int):
        return f"EPSG:{crs}"
    text = str(crs).strip()
    if text.isdigit():
        return f"EPSG:{text}"
    if text.upper().startswith("EPSG:"):
        return "EPSG:" + text.split(":", 1)[1].strip()
    return crs_attr_string(text)


def _aoi_utm_crs(bbox):
    """Offline UTM CRS for a WGS84 AOI bbox, or None when it cannot be trusted.

    Uses the PROJ database (no network) via geopandas' bbox-CENTRE lookup, so it
    is a geometric preference, not a statement about which data exists. Returns
    None for an antimeridian-wrapping bbox, where the centre longitude is
    meaningless (geopandas takes a plain mean of the bounds for geographic input
    and would land on the opposite side of the planet).
    """
    try:
        minx, miny, maxx, maxy = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if maxx - minx > 180:
        return None
    try:
        from shapely.geometry import box as _box

        gdf = gpd.GeoDataFrame(
            geometry=[_box(minx, miny, maxx, maxy)], crs="EPSG:4326"
        )
        return crs_attr_string(gdf.estimate_utm_crs())
    except Exception:
        return None


def _native_crs_area_shares(items, bbox):
    """Fraction of the AOI (0..1) that each native CRS's tiles actually cover.

    This is what decides how much data has to be reprojected, and scene counts
    are a poor proxy for it: one zone can contribute many revisits over a small
    sliver while another covers the whole AOI from fewer tiles.

    Uses item footprints only (metadata already in hand, no extra query and no
    pixels). One representative footprint is kept per (CRS, tile) - the largest,
    so a partial across-track acquisition does not understate its tile - which
    collapses thousands of items to a handful of geometries before the union.

    Returns {} when geometry is unusable, so the caller can fall back.
    """
    try:
        from shapely.geometry import box as _box, shape as _shape
        from shapely.ops import unary_union
    except Exception:
        return {}
    try:
        minx, miny, maxx, maxy = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return {}

    best = {}
    for it in items or []:
        c = _item_crs(it)
        if c is None or not getattr(it, "geometry", None):
            continue
        try:
            geom = _shape(it.geometry)
        except Exception:
            continue
        key = (c, _item_tile(it))
        if key not in best or geom.area > best[key].area:
            best[key] = geom
    if not best:
        return {}

    per_crs = defaultdict(list)
    for (c, _tile), geom in best.items():
        per_crs[c].append(geom)

    try:
        aoi = (
            gpd.GeoSeries([_box(minx, miny, maxx, maxy)], crs="EPSG:4326")
            .to_crs(_EQUAL_AREA)
            .iloc[0]
        )
        total = aoi.area
        if total <= 0:
            return {}
        shares = {}
        for c, geoms in per_crs.items():
            footprint = (
                gpd.GeoSeries([unary_union(geoms)], crs="EPSG:4326")
                .to_crs(_EQUAL_AREA)
                .iloc[0]
            )
            shares[c] = float(aoi.intersection(footprint).area / total)
        return shares
    except Exception:
        return {}


def _choose_target_crs(crs_counts, bbox, items=None, user_crs=None, q=False):
    """Pick the ONE CRS every tile is loaded into - deterministically.

    ``crs_counts`` is a Counter of the NATIVE CRSs of the matched STAC items, so
    this needs no extra query: the items are already fetched. That is the whole
    point - the cube's projection is now decided from the full item set instead
    of from ``items[0]``, whose position is not guaranteed by the STAC API and
    which was verified to differ BETWEEN sources for the same area and dates
    (element84 -> EPSG:32632, terrabyte/PC -> EPSG:32633).

    Rules:
      * ``user_crs`` given -> honoured exactly, with a warning when no item is
        native to it (every scene then gets warped).
      * one native CRS -> that one.
      * several -> the one whose tiles NATIVELY cover the largest share of the
        AOI, i.e. the choice that reprojects the least data. Scene counts are
        only the fallback when footprints are unusable, because they measure
        revisits rather than coverage. The AOI's own UTM zone breaks ties, and
        is deliberately not the primary rule: MGRS tiles overrun their 6-degree
        zone, so an AOI just past a boundary is often imaged entirely by the
        neighbouring zone and its nominal zone may cover little or none of it.

    Ties are broken by CRS string so the result never depends on item order.
    "Least reprojection" is preferred over "closest central meridian": the two
    can disagree, but the scale-error difference between adjacent zones is on
    the order of 0.05%, far below the cost of an extra resampling pass.
    """
    natives = [c for c in crs_counts if c]

    if user_crs is not None:
        target = crs_attr_string(user_crs)
        if natives and target not in natives and not q:
            print(
                f"Note: no scene is natively in {target} (found "
                f"{', '.join(sorted(natives))}) - every scene will be reprojected.",
                flush=True,
            )
        return target

    if not natives:
        return None
    if len(natives) == 1:
        return natives[0]

    # Shares are rounded to 0.1% before ranking: item footprints are not
    # byte-identical between STAC APIs, and without the rounding a near-tie could
    # rank differently per source and undo the cross-source determinism this
    # function exists to provide.
    shares = _native_crs_area_shares(items, bbox)
    preferred = _aoi_utm_crs(bbox)
    if shares:
        ranked = sorted(
            natives,
            key=lambda c: (-round(shares.get(c, 0.0), 3), c != preferred, str(c)),
        )
    else:
        ranked = sorted(
            natives, key=lambda c: (-crs_counts[c], c != preferred, str(c))
        )
    target = ranked[0]

    if not q:
        def _describe(c):
            share = shares.get(c)
            cover = f", covers {share * 100:.0f}% of the area" if share is not None else ""
            return f"{c} ({crs_counts[c]} scenes{cover})"

        breakdown = ", ".join(_describe(c) for c in ranked)
        n_warped = sum(n for c, n in crs_counts.items() if c and c != target)
        basis = "covers most of your area" if shares else "has the most scenes"
        print(
            f"Note: this area spans {len(natives)} projections [{breakdown}]. "
            f"EVERY scene is kept and mosaicked - building in {target} (it "
            f"{basis}) means {n_warped} of them are warped out of their native "
            "projection (one extra resampling pass; pixel size and area are "
            "slightly distorted far from the target zone's central meridian).",
            flush=True,
        )
    return target


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


# Processing baseline at which ESA introduced the -1000 DN radiometric offset.
# Items on either side of it are radiometrically incompatible and scale_factor
# applies ONE offset per timestep, so a single solar day must not mix the two.
_S2_OFFSET_BASELINE = 4.00


def _s2_offset_era(item):
    """True once the -1000 DN offset is baked into an L1C item's pixels."""
    return _s2_baseline(item) >= _S2_OFFSET_BASELINE


def _s2_granule_key(item):
    """Identity of the ground granule behind an L1C item, baseline-independent.

    ``(tile, datastrip sensing time)``. The ``_S<YYYYMMDDThhmmss>`` token of the
    datastrip id is the only field that survives reprocessing unchanged, which
    is why neither obvious alternative works (both verified live on element84):

      * the item's own ``datetime`` CHANGES between baselines - the N02.06 and
        N05.00 copies of S2B tile 32UNC on 2018-06-15 report 04:05:41 and
        04:15:57, so keying on it leaves both copies in;
      * ``s2:product_uri`` carries the datatake time, which two DIFFERENT
        datastrips of one orbit share (T32UNC 2018-06-26, both 102019), so
        keying on it would merge two real acquisitions into one.

    Falls back to the item's own timestamp when no datastrip id is published.
    """
    tile = _item_tile(item)
    ds = str(
        item.properties.get("s2:datastrip_id")
        or item.properties.get("eopf:datastrip_id")  # cdse
        or ""
    )
    m = re.search(r"_S(\d{8}T\d{6})", ds)
    if m:
        return (tile, m.group(1))
    return (tile, str(item.properties.get("datetime", "")))


def _aoi_coverage(geoms, bbox):
    """Share of the AOI (0..1) covered by the union of the given footprints.

    Returns None when the geometry or bbox is unusable, so callers can fall back
    on something else instead of acting on a bogus 0.
    """
    try:
        from shapely.geometry import box as _box
        from shapely.ops import unary_union

        geoms = [g for g in geoms if g is not None]
        if not geoms:
            return None
        minx, miny, maxx, maxy = [float(v) for v in bbox]
        aoi = (
            gpd.GeoSeries([_box(minx, miny, maxx, maxy)], crs="EPSG:4326")
            .to_crs(_EQUAL_AREA)
            .iloc[0]
        )
        if aoi.area <= 0:
            return None
        footprint = (
            gpd.GeoSeries([unary_union(geoms)], crs="EPSG:4326")
            .to_crs(_EQUAL_AREA)
            .iloc[0]
        )
        return float(aoi.intersection(footprint).area / aoi.area)
    except Exception:
        return None


def _dedup_s2_l1c_items(items, bbox, q=False):
    """Drop redundant Sentinel-2 L1C items WITHOUT dropping ground coverage.

    Several items land on the same solar day for two opposite reasons:

      * they cover DIFFERENT ground - the AOI spans several MGRS tiles, or one
        tile was acquired as two datastrip segments. All of them are needed.
      * they are the SAME granule at two processing baselines, because the
        archive still holds the original next to its reprocessed copy. Only the
        newest is wanted, and letting both in is worse than redundant: the
        -1000 DN offset exists from baseline 04.00 on and scale_factor applies
        a single offset per timestep.

    The previous filter kept exactly one item per DAY, which handled the second
    case but silently deleted the first. Verified against the live element84
    catalogue over an AOI spanning 47TPK/47TPL: 17 of 18 dates lost a whole
    tile, so that half of the AOI loaded as nodata - and through
    get_cloud_layers it produced an empty s2cloudless probability map there.

    Two passes:

      1. per solar day, keep ONE radiometric era. Only reachable on a partially
         reprocessed archive (element84); CDSE/terrabyte are uniform. The era
         covering more of the AOI wins rather than always the newer one:
         measured over three AOIs and six seasons, 33 of 581 mixed-baseline
         (day, tile) groups have a datastrip that was never reprocessed, so
         "newest always" would punch holes of up to 49% of the tile.
      2. per granule (tile + datastrip), keep the highest baseline. This is the
         only place an item is dropped as redundant, and by construction it
         covers the same ground as the item kept in its place.
    """
    per_day = defaultdict(list)
    for it in items:
        per_day[str(it.properties.get("datetime", ""))[:10]].append(it)

    kept = []
    for day in sorted(per_day):
        group = per_day[day]

        eras = {_s2_offset_era(it) for it in group}
        if len(eras) > 1:
            new = [it for it in group if _s2_offset_era(it)]
            old = [it for it in group if not _s2_offset_era(it)]
            cov_new = _aoi_coverage([_shapely_geom(it) for it in new], bbox)
            cov_old = _aoi_coverage([_shapely_geom(it) for it in old], bbox)
            cov_both = _aoi_coverage([_shapely_geom(it) for it in group], bbox)
            # Newer era unless the older one demonstrably covers more ground.
            # The margin keeps footprint-polygon refinements between baselines
            # (~1% differences on identical scenes) from flipping the choice.
            if cov_old is not None and cov_new is not None and cov_old > cov_new + 0.02:
                group, keep_label = old, "pre-04.00"
                cov_keep = cov_old
            else:
                group, keep_label = new, ">=04.00"
                cov_keep = cov_new
            # Only worth telling the user about when the choice actually costs
            # ground. Both eras holding the same scene is the normal case (the
            # archive simply kept the original next to its reprocessed copy)
            # and would otherwise warn on nearly every pre-2022 date.
            _unmeasurable = cov_keep is None or cov_both is None
            if not q and (_unmeasurable or cov_keep < cov_both - 0.02):
                _lost = (
                    "coverage could not be measured"
                    if _unmeasurable
                    else f"{cov_both - cov_keep:.0%} of the AOI is left out"
                )
                print(
                    f"Warning: {day} is published at two processing baselines "
                    f"({sorted({_s2_baseline(i) for i in per_day[day]})}), which "
                    f"carry different radiometric offsets and cannot share one "
                    f"timestep. Keeping the {keep_label} scenes - {_lost}."
                )

        by_granule = defaultdict(list)
        for it in group:
            by_granule[_s2_granule_key(it)].append(it)
        for key in sorted(by_granule, key=lambda k: (str(k[0]), str(k[1]))):
            kept.append(max(by_granule[key], key=_s2_baseline))

    return kept


def _shapely_geom(item):
    """Item footprint as a shapely geometry, or None if it has none/unusable."""
    try:
        from shapely.geometry import shape as _shape

        return _shape(item.geometry) if getattr(item, "geometry", None) else None
    except Exception:
        return None


def _probe_window(daterange, window_days):
    """A short date window ending at the user's range end (or today).

    Deliberately NOT the user's full range: the set of projections available for
    an area is fixed by tile geometry and does not change over time, so probing
    three years of items to learn it is pure waste (measured: 295 s for a large
    AOI, versus ~2 s for a short window, same answer). Anchoring at the range END
    also means a 1-day or 3-day cube still probes a usable window.
    """
    end = None
    if isinstance(daterange, (list, tuple)) and len(daterange) == 2:
        try:
            end = datetime.date.fromisoformat(str(daterange[1])[:10])
        except (ValueError, TypeError):
            end = None
    if end is None:
        end = datetime.date.today()
    start = end - datetime.timedelta(days=int(window_days))
    return [start.isoformat(), end.isoformat()]


def probe_native_crs(mission, polygon, source="element84", daterange=None,
                     window_days=14):
    """Which projections an area's scenes are delivered in. No pixels are read.

    Metadata search only, so it is cheap and does not touch the lazy pipeline.
    Sentinel-2 revisits every ~5 days at the equator and more often at higher
    latitudes, so a 14-day window sees every tile covering the AOI; the window is
    widened twice if the archive has a gap there. No cloud filter is applied -
    this is about tile geometry, not usable imagery.

    Returns a list of dicts ``{"crs", "scenes", "share"}`` sorted by the share of
    the AOI each projection natively covers (the same ranking the build uses),
    or [] when nothing is found.
    """
    _source_aliases = {"e84": "element84", "tb": "terrabyte", "pc": "planetary_computer"}
    source = _source_aliases.get(source, source)

    mission_cat = _CATALOGUES.get(mission)
    if mission_cat is None:
        raise ValueError(f"No STAC catalogue configured for mission {mission!r}.")
    if isinstance(mission_cat, dict):
        if source not in mission_cat:
            raise ValueError(
                f"Unknown source {source!r} for {mission}. "
                f"Valid options: {list(mission_cat.keys())}."
            )
        url, collection = mission_cat[source]
    else:
        url, collection = mission_cat

    bbox = polygon if isinstance(polygon, (list, tuple)) else polygon_2_bbox(polygon)
    if not bbox:
        raise ValueError("Could not derive a bounding box from the polygon input.")

    catalog = pystacclient.open(url)
    items = []
    for days in (window_days, window_days * 4, window_days * 12):
        items = catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=_probe_window(daterange, days),
        ).item_collection()
        if len(items):
            break
    if not len(items):
        return []

    counts = Counter(_item_crs(i) for i in items)
    shares = _native_crs_area_shares(items, bbox)
    natives = [c for c in counts if c]
    return [
        {"crs": c, "scenes": int(counts[c]), "share": shares.get(c)}
        for c in sorted(
            natives, key=lambda c: (-round(shares.get(c, 0.0), 3), str(c))
        )
    ]


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
    crs=None,
    min_footprint_coverage=None,
    clip_raster=None,
    q: bool = False,
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
        items, crs_counts, stac_mission, tiles = _catalogue_search(
            catalog, collection, bbox, daterange, query, mission, source=source
        )
    else:
        start_md, end_md, years_spec = season_spec
        years = _parse_years_spec(years_spec, mission)
        windows = _expand_season_windows(start_md, end_md, years)

        all_items = []
        crs_counts = Counter()
        stac_mission = None
        tiles_set = set()

        for win in windows:
            win_items, win_crs_counts, win_stac_mission, win_tiles = _catalogue_search(
                catalog, collection, bbox, win, query, mission,
                allow_empty=True, source=source,
            )
            if win_items:
                all_items.extend(list(win_items))
                # Merge across season windows so the target CRS is chosen from
                # every scene the cube will hold, not from the first window.
                crs_counts.update(win_crs_counts)
                if stac_mission is None:
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

    # --- Footprint prefilter (pre-load) ---------------------------------------
    # Drop acquisitions that barely graze the AOI while they are still just STAC
    # metadata. This is the ONLY scene filter that runs before stac_load, so it
    # is the only one that saves work rather than trimming the result: a dropped
    # acquisition is never loaded, never has its SCL read and never gets a
    # full-grid slab. Runs before _choose_target_crs on purpose, so the target
    # projection is chosen from the scenes the cube will actually hold.
    #
    # Imported here, not at module scope: auxiliary imports from this module, so
    # a top-level import would be circular.
    footprint_filter = None
    if min_footprint_coverage:
        from .auxiliary import filter_items_by_footprint

        items, footprint_filter = filter_items_by_footprint(
            items,
            polygon,
            min_footprint_coverage,
            coverage_geometry="polygon" if clip_raster else "bbox",
            tile_handling=tile_handling,
            q=q,
        )
        # The tile list is rebuilt from the survivors: a whole MGRS tile can
        # disappear with the acquisitions that only touched the AOI through it.
        if footprint_filter and mission in ("sentinel_2_l2a", "sentinel_2_l1c"):
            _kept_tiles = {_item_tile(it) for it in items}
            _kept_tiles.discard(None)
            if _kept_tiles:
                tiles = np.array(sorted(_kept_tiles))
            crs_counts = Counter(_item_crs(item) for item in items)

    # --- Target projection ----------------------------------------------------
    # ONE CRS for the whole cube (every tile is warped into it by odc-stac), now
    # chosen from the native CRSs of ALL matched items rather than from items[0].
    target_crs = _choose_target_crs(crs_counts, bbox, items=items, user_crs=crs, q=q)

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

    # Sentinel-2 L1C: drop reprocessing duplicates of the same granule, but keep
    # every distinct tile/datastrip of a day (see _dedup_s2_l1c_items - the old
    # one-item-per-day filter deleted whole MGRS tiles on multi-tile AOIs).
    if mission == "sentinel_2_l1c":
        items = _dedup_s2_l1c_items(items, bbox, q=q)

    _load_kwargs = dict(
        bands=bands,
        crs=target_crs,
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

    # Projection provenance, stashed as Dataset attrs for main.py to pick up and
    # record on the finished cube. Same channel as scene_metadata_multiday below:
    # these do not survive main.py's band concat, so they are read there directly.
    if target_crs is not None:
        stac.attrs["target_crs"] = target_crs
    # Provenance for the pre-load prefilter, picked up by main.py: a cube whose
    # date list was trimmed before download must be able to say so, and by how
    # much (the estimate is catalogue-dependent, so the source matters too).
    if footprint_filter:
        stac.attrs["footprint_prefilter"] = footprint_filter

    # --- Per-day mean solar azimuth, from the items already in hand ----------
    # Shadow masking needs one azimuth per solar day and today runs a SECOND
    # STAC search to get it (get_solar_geometry via solar_azimuths_for_days) -
    # measured at ~25% of a lazy shadow build, and it grows with the item count.
    # Every value it wants is already on the items matched above, so they are
    # summarised here for main.py to reuse.
    #
    # Offered ONLY when this search cannot have seen a different item set than
    # that one does, because the value is a per-day MEAN and a different set
    # means a different mean. get_solar_geometry queries the same endpoint and
    # bbox with NO cloud filter, so the sets match only when nothing narrowed
    # ours either:
    #   * no effective cloud query - measured on a 6-month, 2-tile AOI:
    #     max_cc=None and max_cc=100 both give the same 144 items and azimuths
    #     equal to 1e-10 deg, while max_cc=30 leaves 78 items and moves 8 of 43
    #     days by up to 1.33 deg (a tile of a multi-tile day drops out);
    #   * no footprint prefilter, which likewise drops whole acquisitions;
    #   * not season mode, whose windows are not one contiguous range.
    # L2A only: the L1C dedup below rewrites the item list for the same reason.
    # When any of that applies the attr is simply absent and main.py falls back
    # to the search, so the result is unchanged either way.
    if (
        mission == "sentinel_2_l2a"
        and (max_cc is None or float(max_cc) >= 100)
        and not footprint_filter
        and season_spec is None
    ):
        _az_by_day = {}
        for _it in items:
            _p = _it.properties
            _a = _p.get("view:sun_azimuth")
            if _a is None:
                _a = _p.get("s2:mean_solar_azimuth")
            if _a is None:
                continue
            _az_by_day.setdefault(str(_p.get("datetime", ""))[:10], []).append(
                float(_a)
            )
        if _az_by_day:
            stac.attrs["solar_azimuth_by_day"] = {
                _d: float(np.mean(_v)) for _d, _v in _az_by_day.items()
            }
    _natives = sorted(c for c in crs_counts if c)
    # Recorded whenever the scenes are NOT all native to the cube's own CRS, i.e.
    # whenever something had to be reprojected. That covers both a multi-projection
    # area AND a single-projection area where the user asked for a different CRS
    # (e.g. EPSG:3035), which is equally worth telling them about. Stays absent in
    # the ordinary case of one native projection that the cube also uses.
    if _natives and set(_natives) != {target_crs}:
        stac.attrs["native_crs"] = _natives
        # Recomputed here rather than plumbed out of _choose_target_crs: this
        # branch is the rare one, and the union is over a handful of per-tile
        # footprints, not per item.
        _shares = _native_crs_area_shares(items, bbox)
        if _shares:
            stac.attrs["native_crs_share"] = [
                float(round(_shares.get(c, 0.0), 4)) for c in _natives
            ]

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
        # One baseline per LOADED timestep, keyed by solar day (the same day-key
        # convention as extract_scene_metadata). Deliberately NOT an exact
        # timestamp join: odc-stac labels a merged solar day with the FIRST
        # item's sensing time, so matching on the timestamp keeps only one item
        # of a multi-tile day - and the duplicate-timestamp guard that used to
        # follow it then deleted such dates from the cube outright.
        # _dedup_s2_l1c_items has already made every day single-era, so the
        # spread that can remain within a day (e.g. 3.00 next to 3.01) cannot
        # flip the -1000 offset; max() reports the newest.
        per_day_baseline = defaultdict(list)
        for item in items:
            day = str(item.properties.get("datetime", ""))[:10]
            per_day_baseline[day].append(_s2_baseline(item))

        item_times = pd.to_datetime(
            [item.properties["datetime"] for item in items], format="mixed", utc=True
        ).tz_localize(None).to_numpy(dtype="datetime64[ns]")
        item_baselines = np.array([_s2_baseline(item) for item in items], dtype=float)

        values = []
        n_fallback = 0
        for t in stac.time.values:
            vals = per_day_baseline.get(str(pd.Timestamp(t).date()))
            if vals:
                values.append(max(vals))
            else:
                # odc-stac groups by SOLAR day, the item key is the UTC date;
                # near the dateline the two can differ. Take the nearest item
                # in time rather than guessing an offset.
                n_fallback += 1
                nearest = int(
                    np.argmin(np.abs(item_times - np.datetime64(t, "ns")))
                )
                values.append(float(item_baselines[nearest]))
        if n_fallback and not q:
            print(
                f"Warning: {n_fallback} loaded date(s) had no L1C item on the "
                f"matching UTC day (solar-day grouping shifts the date near the "
                f"dateline) - processing baseline taken from the nearest item."
            )

        baselines = xr.DataArray(
            np.array(values, dtype=float),
            dims=["time"],
            coords={"time": stac.time.values},
            name="processing_baseline",
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
# _is_mmdd / _expand_season_windows are imported from stac_processing, shared
# with the custom seasonal composites of calculate_statistics.


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
            return [], Counter(), None, None
        raise ValueError(
            "No scenes found by the given parameters. Please check your polygon's geometry, date range or increase max cloud coverage."
        )

    # Native CRS of EVERY item, not just items[0]: the caller picks the target
    # projection from the full distribution (see _choose_target_crs). Free - the
    # items are already fetched, this is dict lookups over metadata in memory.
    crs_counts = Counter(_item_crs(item) for item in items)
    stac_mission = items[0].to_dict().get("collection")
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

    return items, crs_counts, stac_mission, tiles


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