import datetime
import re

import numpy as np
import xarray as xr


# ==========================================================
# SEASON / DATE WINDOW HELPERS
# ==========================================================
# Shared by the seasonal `daterange` of get_stac_layers (which searches STAC one
# season window per year) and the custom seasonal composites of
# calculate_statistics (which reduces an already-built cube over the same
# windows), so both read "MM-DD" exactly the same way. They live here rather
# than in get_data because get_data pulls odc/pystac/geopandas at import.
_MMDD_RE = re.compile(r"^\d{2}-\d{2}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_mmdd(s) -> bool:
    """Return True if string is in MM-DD format and represents a valid calendar day."""
    if not isinstance(s, str) or not _MMDD_RE.match(s.strip()):
        return False
    mm, dd = map(int, s.strip().split("-"))
    try:
        # Use a leap year to allow 02-29 in case someone needs it
        datetime.date(2000, mm, dd)
    except ValueError:
        return False
    return True


def is_iso_date(s) -> bool:
    """Return True if string is a valid YYYY-MM-DD calendar date."""
    if not isinstance(s, str) or not _ISO_DATE_RE.match(s.strip()):
        return False
    try:
        datetime.date.fromisoformat(s.strip())
    except ValueError:
        return False
    return True


def season_crosses_year(start_md: str, end_md: str) -> bool:
    """True when a MM-DD season starts later in the calendar than it ends, i.e.
    it runs over New Year (e.g. 11-01 .. 03-31)."""
    sm, sd = map(int, start_md.split("-"))
    em, ed = map(int, end_md.split("-"))
    return (sm, sd) > (em, ed)


def expand_season_windows(start_md: str, end_md: str, years):
    """Expand a season (MM-DD .. MM-DD) into per-year concrete ISO windows.

    If start_md is later than end_md (e.g. 11-01 .. 03-31), season crosses year
    boundary and the window is labelled by its START year.
    """
    crosses_year = season_crosses_year(start_md, end_md)

    windows = []
    for y in years:
        start_date = f"{int(y)}-{start_md}"
        end_year = int(y) + 1 if crosses_year else int(y)
        end_date = f"{end_year}-{end_md}"
        windows.append([start_date, end_date])

    return windows


def cloud_mask(stac, mission, keep_clouds=False, keep_layer=False):
    """Build the per-pixel cloud boolean from the scene-classification / QA layer
    and either remove the cloudy pixels (default) or leave the imagery untouched.

    Returns ``(cube, cloud_bool, imaged_bool)``:
      * ``cube``       - cloudy pixels set to NaN (default), OR, when
                         ``keep_clouds`` is True, the cube with pixels intact and
                         only the auxiliary cloud layer dropped.
      * ``cloud_bool`` - the per-pixel cloud mask (time, y, x). It is what lets
                         the caller compute a cloud percentage even in keep-clouds
                         mode (where there are no NaN holes to count). ``None`` for
                         missions without a configured cloud layer.
      * ``imaged_bool``- the per-pixel "was this pixel imaged" boolean (time, y,
                         x), derived from the SAME SCL/QA read: True where the
                         pixel carries a real class (not SCL No-Data / not the QA
                         fill bit). It is the reliable per-scene footprint signal
                         for scene coverage - a swath/orbit gap is SCL 0, caught
                         even when the cube loads gaps as 0 rather than NaN.
                         ``None`` for missions without a configured cloud layer.

    ``keep_clouds`` exists for users (e.g. artists) who want the clouds visible
    but still want a per-scene cloud percentage to filter the fully-clouded
    scenes out. The percentage uses the SAME cloud classes as the masking, so the
    number means the same thing whether the clouds were removed or kept.

    ``keep_layer`` keeps the classification layer (scl / qa_pixel) in the cube
    instead of dropping it after use - set when the user explicitly requested
    it as a band (e.g. to let shadow masking reuse it without re-downloading).
    """
    cfg = _mission_cfg(mission)
    layer_name = cfg[0]
    classes = cfg[1]

    if mission == "sentinel_2_l2a":
        # For Sentinel-2, use the classification directly with isin().
        scl = getattr(stac, layer_name)
        cloud_bool = scl.isin(classes)
        # Imaged = any real SCL class; class 0 (No Data) and NaN mark a gap.
        imaged_bool = scl.notnull() & (scl != 0)
    elif mission == "landsat_c2_l2":
        # For Landsat, qa_pixel is bit-packed. Extract three flags:
        #   dilated_cloud is in bit offset 1,
        #   cirrus      is in bit offset 2,
        #   cloud       is in bit offset 3.
        mask_dilated = ((stac.qa_pixel >> 1) & 1).astype(bool)
        mask_cirrus = ((stac.qa_pixel >> 2) & 1).astype(bool)
        mask_cloud = ((stac.qa_pixel >> 3) & 1).astype(bool)
        # Combine the three flags: a pixel is cloud if any of the flags are True.
        cloud_bool = mask_dilated | mask_cirrus | mask_cloud
        # qa_pixel bit 0 = "Designated Fill" (no-data). Imaged = fill bit clear
        # and not the odc fill value 0. (Untested without a Landsat fixture; the
        # scene-coverage caller falls back to a band read if this is wrong.)
        _qa = stac.qa_pixel.fillna(0).astype("int64")
        imaged_bool = (_qa != 0) & ((_qa & 1) == 0)
    else:
        # If no cloud masking is implemented for the mission, do nothing.
        print(f"No cloud masking configured for mission {mission}")
        return stac, None, None

    if keep_clouds:
        # Keep the imagery exactly as observed (no NaN holes); only the auxiliary
        # cloud layer is dropped below. The percentage is derived from cloud_bool.
        out = stac
    else:
        out = stac.where(~cloud_bool)

    # Drop the cloud/classification band if it exists (it is not spectral
    # data) - unless the user explicitly asked to keep it in the cube.
    if not keep_layer and layer_name is not None and layer_name in out:
        out = out.drop_vars(layer_name)

    return out, cloud_bool, imaged_bool


def build_scl_mask_cube(stac, cloud_bool):
    """Binary SCL cloud-mask time series aligned to the (final) cube grid.

    1 = cloud (the SCL/QA cloud classes), 0 = clear / other. Returned as a
    DataArray named ``Cloud_Stack`` with a single band ``cloud_mask_scl`` so it
    follows the package's Cloud_Stack convention and can be fed to
    ``mask_stac_clouds`` or the cloud / co-registration tools later. This is what
    lets a *keep-clouds* cube (clouds left visible) still be masked or filtered
    after the fact.

    ``cloud_bool`` is the per-pixel boolean returned by :func:`cloud_mask`. It is
    aligned to ``stac`` by label (a clip may have dropped rows/cols) and its time
    is re-stamped to the cube's (floored) time - exactly like
    ``compute_cloud_percentage`` - so the mask lines up pixel-for-pixel and
    date-for-date with the data cube. The cube's ``cloud_percentage`` coord is
    carried over when present.
    """
    cm = cloud_bool.sel(y=stac["y"], x=stac["x"]).assign_coords(time=stac["time"])
    cm = cm.astype("uint8")
    cm = cm.expand_dims(band=["cloud_mask_scl"]).transpose("time", "band", "y", "x")
    cm.name = "Cloud_Stack"
    if "cloud_percentage" in stac.coords:
        cm = cm.assign_coords(
            cloud_percentage=("time", np.asarray(stac["cloud_percentage"].data))
        )
    return cm


def scale_factor(stac, mission, baselines, source=None):
    # Categorical layers (SCL class codes, bit-packed QA words) must keep
    # their raw values: reflectance gain/offset would destroy them (e.g. the
    # PC L2A offset would turn SCL class 4 into (4-1000)*1e-4 = -0.0996).
    # Pop them out, scale the spectral variables, then re-attach as float32
    # (the later band-concat needs one uniform dtype).
    categorical = {}
    if isinstance(stac, xr.Dataset):
        for name in ("scl", "qa_pixel", "qa_radsat", "qa_aerosol", "qa_temp"):
            if name in stac.data_vars:
                categorical[name] = stac[name].astype("float32")
                stac = stac.drop_vars(name)

    stac = _scale_values(stac, mission, baselines, source)

    for name, da in categorical.items():
        stac[name] = da
    return stac


def _scale_values(stac, mission, baselines, source=None):
    # Normalize short source aliases (mirror get_data) so the per-provider
    # Sentinel-2 L2A offset logic below triggers regardless of how it was called.
    _source_aliases = {"e84": "element84", "tb": "terrabyte", "pc": "planetary_computer"}
    source = _source_aliases.get(source, source)

    # L1C: baseline-aware scaling. The -1000 DN radiometric offset was introduced
    # at processing baseline 04.00; reflectance = (DN - 1000)/10000 for >= 04.00,
    # else DN/10000. L1C carries no SCL and the per-scene baseline also matters
    # downstream (s2cloudless), so L1C stays baseline-driven.
    if mission == "sentinel_2_l1c":
        baselines = baselines.astype(float)
        baselines_aligned = baselines.sel(time=stac.time)
        # Cast to float32 up front: keeps the cube float32 (not float64, which
        # integer/float division would otherwise produce) and avoids uint16
        # underflow in (DN - 1000) for dark pixels (DN < 1000).
        stac = stac.astype("float32")
        stac = xr.where(baselines_aligned >= 4.00, (stac - 1000) / 10000, stac / 10000)
        return stac

    cfg = _mission_cfg(mission)
    gain = cfg[2]
    offset = cfg[3]

    # Sentinel-2 L2A: reflectance = DN * 1e-4, minus the PB>=04.00 BOA offset
    # (-1000 DN = -0.1) where it is actually baked into the served pixels. Whether
    # it is present depends ENTIRELY on the provider, verified against real pixels:
    #   * element84 == cdse to 4 dp, and element84 == PC - 1000.
    # Cast to float32 first (raw DNs are uint16; `*gain` would otherwise promote to
    # float64 -> doubled cube, and (DN-1000) would underflow uint16 for DN<1000).
    if mission == "sentinel_2_l2a":
        stac = stac.astype("float32")
        if source == "element84":
            # element84 builds its OWN L2A COGs and harmonizes them: the +1000
            # offset is already removed from the pixels -> nothing to subtract.
            return stac * gain
        if source in ("cdse", "terrabyte"):
            # Raw ESA Collection-1 archive: every scene is reprocessed to PB>=05,
            # so the +1000 offset is present on ALL scenes -> always subtract it.
            return (stac - 1000) * gain
        # planetary_computer: raw, NON-reprocessed rolling archive with mixed
        # baselines. The +1000 offset exists only for PB>=04.00, i.e. acquisitions
        # from 2022-01-25 onward; older scenes (e.g. PB 02.xx) carry no offset.
        cutoff = np.datetime64("2022-01-25")
        return xr.where(stac["time"] >= cutoff, (stac - 1000) * gain, stac * gain)

    # Other missions (Landsat / S1 / DEM): single gain/offset from _mission_cfg.
    return (stac.astype("float32") + offset) * gain


def _mission_cfg(mission):
    # relevant band (cloud), classifications (cloud), gain (scale), offset (scale)
    cfg = {
        "sentinel_2_l2a": ("scl", [8, 9, 10], 1e-4, 0),  # scl: 3 is cloud shadows
        "sentinel_2_l1c": (None, None, 1e-4, -1000),
        "landsat_c2_l2": (
            "qa_pixel",
            [1, 2, 3],
            0.0000275,
            -0.2,
        ),  # qa_pixel: bit-packed values
        "sentinel_1_rtc": (None, None, 1, 0),
        "cop_dem_glo_30": (None, None, 1, 0),
    }
    return cfg[mission]
