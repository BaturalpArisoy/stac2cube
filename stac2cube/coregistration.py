import os
import io
import numpy as np
import xarray as xr
import rioxarray
from .export_cfg import (
    open_cube,
    is_zarr_path,
    _write_zarr,
    _set_compression,
    write_qgis_vrt,
    normalize_stack_name,
    resolve_stack_var,
)
from arosics import COREG
from geoarray import GeoArray
from rasterio.transform import Affine
from rasterio.enums import Resampling
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from tqdm.auto import tqdm


# ----------------------------------------------------------------------
# Helper to suppress noisy AROSICS warnings / prints
# ----------------------------------------------------------------------
@contextmanager
def _suppress_arosics_warnings():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            yield


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _compute_coords(gt, height, width):
    origin_x, pixel_width, _, origin_y, _, pixel_height = gt
    x_coords = origin_x + pixel_width * (np.arange(width) + 0.5)
    y_coords = origin_y + pixel_height * (np.arange(height) + 0.5)
    return y_coords, x_coords


def _get_bounds_from_gt(gt, height, width):
    y_coords, x_coords = _compute_coords(gt, height, width)
    left = np.min(x_coords)
    right = np.max(x_coords)
    bottom = np.min(y_coords)
    top = np.max(y_coords)
    return left, bottom, right, top


def _load_coreg_input(input_obj, stack_name="Time_Series"):
    """
    Accept input as:
      - str path to NetCDF
      - xarray.Dataset containing stack_name
      - xarray.DataArray (stack itself)

    Returns:
      ds (xr.Dataset or None),
      stack (xr.DataArray),
      cloud_pct_da (xr.DataArray or None),
      input_path_str (str or None)
    """
    if isinstance(input_obj, str):
        ds = open_cube(input_obj)
        # open_cube migrated a legacy time-series name; map a legacy stack_name
        # passed by an older script forward too.
        stack_name = resolve_stack_var(ds, stack_name)
        if stack_name not in ds:
            raise KeyError(
                f"Dataset has no variable '{stack_name}'. Found: {list(ds.data_vars)}"
            )
        stack = ds[stack_name]
        input_path_str = input_obj

    elif isinstance(input_obj, xr.Dataset):
        # An in-memory Dataset bypasses open_cube, so migrate it here.
        ds = normalize_stack_name(input_obj)
        input_obj = ds
        stack_name = resolve_stack_var(ds, stack_name)
        if stack_name not in ds:
            raise KeyError(
                f"Dataset has no variable '{stack_name}'. Found: {list(ds.data_vars)}"
            )
        stack = ds[stack_name]
        input_path_str = None

    elif isinstance(input_obj, xr.DataArray):
        ds = None
        stack = normalize_stack_name(input_obj)
        input_path_str = None

    else:
        raise TypeError(
            "input_path must be one of: str (netcdf path), xarray.Dataset, xarray.DataArray"
        )

    cloud_pct_da = None
    if "cloud_percentage" in stack.coords:
        cloud_pct_da = stack.coords["cloud_percentage"]
    elif ds is not None:
        if "cloud_percentage" in ds.coords:
            cloud_pct_da = ds.coords["cloud_percentage"]
        elif "cloud_percentage" in ds.data_vars:
            cloud_pct_da = ds["cloud_percentage"]

    return ds, stack, cloud_pct_da, input_path_str


def _get_geotransform(stack, ds=None):
    """
    Returns GDAL geotransform: [x0, px_w, 0, y0, 0, px_h]
    Tries:
      1) stack.rio.transform()
      2) stack.spatial_ref.GeoTransform
      3) ds.spatial_ref.GeoTransform
      4) derive from x/y coords
    """
    try:
        aff = stack.rio.transform(recalc=False)
        return list(map(float, aff.to_gdal()))
    except Exception:
        pass

    try:
        if hasattr(stack, "spatial_ref") and hasattr(stack.spatial_ref, "GeoTransform"):
            return [float(x) for x in stack.spatial_ref.GeoTransform.split()]
    except Exception:
        pass

    try:
        if (
            ds is not None
            and "spatial_ref" in ds.variables
            and hasattr(ds.spatial_ref, "GeoTransform")
        ):
            return [float(x) for x in ds.spatial_ref.GeoTransform.split()]
    except Exception:
        pass

    if "x" in stack.coords and "y" in stack.coords:
        x = np.asarray(stack.coords["x"].values, dtype=float)
        y = np.asarray(stack.coords["y"].values, dtype=float)
        if x.size < 2 or y.size < 2:
            raise ValueError("Cannot derive transform: x/y coords too short.")
        px_w = float(np.median(np.diff(x)))
        px_h = float(np.median(np.diff(y)))  # often negative
        x0 = float(x[0] - px_w / 2.0)
        y0 = float(y[0] - px_h / 2.0)
        return [x0, px_w, 0.0, y0, 0.0, px_h]

    raise ValueError(
        "Could not determine geotransform. Ensure the DataArray has rio transform/CRS "
        "or comes from a Dataset with 'spatial_ref.GeoTransform'."
    )


def _get_crs_wkt(stack, ds=None):
    try:
        if stack.rio.crs is not None:
            return stack.rio.crs.to_wkt()
    except Exception:
        pass

    try:
        if hasattr(stack, "spatial_ref") and hasattr(stack.spatial_ref, "crs_wkt"):
            return stack.spatial_ref.crs_wkt
    except Exception:
        pass

    try:
        if (
            ds is not None
            and "spatial_ref" in ds.variables
            and hasattr(ds.spatial_ref, "crs_wkt")
        ):
            return ds.spatial_ref.crs_wkt
    except Exception:
        pass

    raise ValueError(
        "Could not determine CRS WKT. Ensure the DataArray has stack.rio.crs set "
        "or provide a Dataset that contains spatial_ref with crs_wkt."
    )


def _auto_output_path(input_path_str, suffix="_coregistered"):
    in_dir, in_name = os.path.split(input_path_str)
    base, ext = os.path.splitext(in_name)
    if not ext:
        ext = ".nc"
    out_name = f"{base}{suffix}{ext}"
    return os.path.join(in_dir, out_name)


def _write_coreg_output(out_ds, out_path, compress=False, vrt=False):
    """Write the co-registered Dataset; the extension picks the format.

    ``*.zarr`` -> streamed Zarr store (export_cfg._write_zarr, same writer as
    export_stac); anything else -> the long-standing plain to_netcdf.

    ``compress`` and ``vrt`` are NetCDF-only and mirror ``export_stac``: zlib
    level 4 on the spatial variables, and the QGIS band-mapping sidecar. Zarr
    always applies its own codec, and a VRT cannot read a Zarr store's pixels
    back, so both are ignored on that path.
    """
    if is_zarr_path(out_path):
        _write_zarr(out_ds, out_path, overwrite=True)
        return

    _set_compression(out_ds, compress)
    out_ds.to_netcdf(out_path)
    if vrt:
        # A sidecar failure must not lose the cube that was just written.
        try:
            print(f"QGIS band-labelled VRT: {write_qgis_vrt(out_path)}")
        except Exception as exc:
            print(f"Note: could not write the QGIS VRT ({exc}). "
                  "The NetCDF is fine.")


def _apply_time_and_cloud_filters(stack, max_cc=None, time_period=None):
    """
    Uses stac2cube.filter_cloud(stack, max_cc) if available.
    Falls back to stack.where(stack.cloud_percentage <= max_cc) if possible.
    time_period: None OR ["YYYY-MM-DD", "YYYY-MM-DD"] OR (start, end)
    """
    out = stack

    # cloud filter
    if max_cc is not None:
        try:
            from stac2cube import filter_cloud

            out = filter_cloud(out, max_cc)
        except Exception:
            # fallback if filter_cloud not importable or fails
            if "cloud_percentage" in out.coords:
                out = out.where(out.cloud_percentage <= float(max_cc), drop=True)

    # time filter
    if time_period is not None and "time" in out.dims:
        if not (isinstance(time_period, (list, tuple)) and len(time_period) == 2):
            raise TypeError("time_period must be None or [start, end] (two elements).")
        start, end = time_period
        out = out.sel(time=slice(start, end))

    return out


# ----------------------------------------------------------------------
# Consensus estimation engine
# ----------------------------------------------------------------------
# Bands acquired natively at 10 m by Sentinel-2. 20-m bands (rededge*,
# nir08, swir16, swir22) are only resampled to the 10-m grid and lack the
# high-frequency content the FFT matcher needs (measured: reliability ~25
# vs ~65 on the same scenes), so auto-selection is restricted to these.
_NATIVE_10M_BANDS = ("nir", "red", "green", "blue")


def _resolve_match_band(band_names, match_band):
    """Return (one_based_index, band_label) of the band used for matching."""
    labels = [str(b) for b in band_names]
    lowered = [s.lower() for s in labels]

    if match_band is None or str(match_band).lower() == "auto":
        for cand in _NATIVE_10M_BANDS:
            if cand in lowered:
                i = lowered.index(cand)
                return i + 1, labels[i]
        raise ValueError(
            "match_band='auto' requires one of the native 10-m bands "
            f"{list(_NATIVE_10M_BANDS)} in the cube; found {labels}. "
            "Pass match_band=<band name> to override explicitly."
        )

    mb = str(match_band).lower()
    if mb not in lowered:
        raise ValueError(
            f"match_band='{match_band}' not found in the cube. Available: {labels}"
        )
    i = lowered.index(mb)
    if mb not in _NATIVE_10M_BANDS:
        warnings.warn(
            f"match_band='{labels[i]}' is not a native 10-m Sentinel-2 band. "
            "Resampled 20-m bands match far worse (low reliability); expect "
            "degraded co-registration quality.",
            stacklevel=3,
        )
    return i + 1, labels[i]


def _grid_candidates(geotransform, height, width, grid_size):
    """Matching-window positions: AROSICS' automatic position plus a
    grid_size x grid_size lattice across the AOI."""
    left, bottom, right, top = _get_bounds_from_gt(geotransform, height, width)
    margin = 1.0 / (grid_size + 1)
    frac_vals = np.linspace(margin, 1.0 - margin, grid_size)
    cands = [("auto", None)]
    for iy, fy in enumerate(frac_vals):
        for ix, fx in enumerate(frac_vals):
            cands.append(
                (
                    f"g{grid_size}x{grid_size}_r{iy}_c{ix}",
                    (left + fx * (right - left), bottom + fy * (top - bottom)),
                )
            )
    return cands


# adaptive escalation: a coarse-stage consensus is accepted outright only
# when it is this unambiguous (measured clear-scene spreads: 0.02-0.09 px)
_COARSE_MIN_SUCCESS = 5
_COARSE_MIN_INLIER_FRACTION = 0.8
_COARSE_MAX_SPREAD_PX = 0.15
_COARSE_N = 10  # coarse-stage window budget (matches the old 3x3 lattice + auto)

# cloud-aware window placement: AROSICS tolerates NO nodata inside the
# matching window (measured: ~10 NaN pixels in a 256 px window already make
# the match fail), so on partially cloudy scenes blind lattice positions
# almost all die and good scenes get dropped. Windows are therefore placed
# only where BOTH images are fully clear, scanned on a dense candidate
# lattice with an integral image (cost is negligible next to one COREG call).
_WIN_PX = 256  # AROSICS default matching-window size (win_size_XY)
_CLEAR_PAD_PX = 8  # margin: grid alignment (1 px) + integer-shift search
_PLACEMENT_STRIDE_PX = 64
_MIN_SEP_PX = 128  # selected window centers at least half a window apart


def _integral_image(mask2d):
    ii = np.zeros((mask2d.shape[0] + 1, mask2d.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = mask2d.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    return ii


def _box_count(ii, cy, cx, half):
    return (
        ii[cy + half, cx + half]
        - ii[cy - half, cx + half]
        - ii[cy + half, cx - half]
        + ii[cy - half, cx - half]
    )


def _clear_window_positions(
    ref_finite,
    tgt_finite,
    geotransform,
    n_max,
    win_px=_WIN_PX,
    pad_px=_CLEAR_PAD_PX,
    stride_px=_PLACEMENT_STRIDE_PX,
    min_sep_px=_MIN_SEP_PX,
):
    """Candidate window positions where BOTH images are fully clear
    (no NaN in the padded matching-window box).

    Returns [(label, (map_x, map_y)), ...] ordered by farthest-point
    sampling from the AOI center, so any prefix of the list is itself a
    well-spread subset (used directly as the adaptive coarse stage).
    Returns [] when the AOI is smaller than a padded window or no fully
    clear position exists.
    """
    h, w = ref_finite.shape
    half = win_px // 2 + pad_px
    if h < 2 * half or w < 2 * half:
        return []

    both = ref_finite & tgt_finite
    ii = _integral_image(both)
    full = (2 * half) ** 2

    cys = np.arange(half, h - half + 1, stride_px)
    cxs = np.arange(half, w - half + 1, stride_px)
    pool = [
        (cy, cx)
        for cy in cys
        for cx in cxs
        if _box_count(ii, cy, cx, half) == full
    ]
    if not pool:
        return []

    pts = np.asarray(pool, dtype=float)
    center = np.array([h / 2.0, w / 2.0])
    seed = int(np.argmin(((pts - center) ** 2).sum(axis=1)))
    selected = [seed]
    dist = np.hypot(*(pts - pts[seed]).T)
    while len(selected) < min(n_max, len(pool)):
        i = int(np.argmax(dist))
        if dist[i] < min_sep_px:
            break
        selected.append(i)
        dist = np.minimum(dist, np.hypot(*(pts - pts[i]).T))

    left, top = geotransform[0], geotransform[3]
    px_w, px_h = geotransform[1], geotransform[5]
    out = []
    for k, i in enumerate(selected):
        cy, cx = pool[i]
        out.append((f"cw{k}_y{cy}_x{cx}", (left + cx * px_w, top + cy * px_h)))
    return out


def _place_candidates(ref_finite, tgt_finite, geotransform, grid_candidates):
    """Window positions for one scene pair: fully clear positions first
    (farthest-point ordered), topped up with blind lattice positions that
    keep their distance from the clear ones. Falls back to the blind
    lattice entirely when no clear position exists (small AOIs, hopeless
    scenes) - AROSICS' own window shrinking then gets its chance, exactly
    the pre-placement behavior.

    Returns (candidates, n_clear).
    """
    n_full = len(grid_candidates)
    clear = _clear_window_positions(ref_finite, tgt_finite, geotransform, n_full)
    if not clear:
        return grid_candidates, 0
    if len(clear) >= n_full:
        return clear, len(clear)

    left, top = geotransform[0], geotransform[3]
    px_w, px_h = geotransform[1], geotransform[5]
    sel_px = np.array(
        [((wp[1] - top) / px_h, (wp[0] - left) / px_w) for _l, wp in clear]
    )
    cands = list(clear)
    for label, wp in grid_candidates:
        if len(cands) >= n_full:
            break
        if wp is None:
            continue
        p = np.array([(wp[1] - top) / px_h, (wp[0] - left) / px_w])
        if np.hypot(*(sel_px - p).T).min() >= _MIN_SEP_PX:
            cands.append((label, wp))
    return cands, len(clear)


def _estimate_windows(ref_geoArr, tgt_geoArr, candidates, band_idx, min_win_px):
    """Estimation-only AROSICS at every candidate window (no warping).

    Windows whose ACTUAL matching window ends up below min_win_px in
    either dimension are discarded (AROSICS clips windows near AOI edges;
    tiny windows measured wildly unreliable).

    Returns (xs, ys, reliabilities) as float arrays (possibly empty).
    """
    xs, ys, rel = [], [], []
    with _suppress_arosics_warnings():
        for _label, wp in candidates:
            try:
                kwargs = dict(
                    align_grids=True, q=True, r_b4match=band_idx, s_b4match=band_idx
                )
                if wp is not None:
                    kwargs["wp"] = wp
                cr = COREG(ref_geoArr, tgt_geoArr, **kwargs)
                cr.calculate_spatial_shifts()
                if cr.success is not True or cr.shift_reliability is None:
                    continue
                win_y, win_x = cr.matchBox.imDimsYX
                if min(win_x, win_y) < min_win_px:
                    continue
                xs.append(float(cr.x_shift_px))
                ys.append(float(cr.y_shift_px))
                rel.append(float(cr.shift_reliability))
            except (RuntimeError, ValueError, AssertionError, AttributeError):
                continue
    return np.asarray(xs), np.asarray(ys), np.asarray(rel)


def _consensus_from_estimates(xs, ys, rel, n_attempted):
    """Robust consensus over per-window shift estimates:
      reliability-weighted median -> inliers within max(0.2 px, 3*MAD)
      -> consensus = reliability-weighted mean of the inliers.

    A single high-reliability outlier window (e.g. on a cloud edge or a
    migrated river bar) gets outvoted instead of winning outright, which
    is what poisoned the old argmax-based chaining.

    Returns dict(x, y, n_ok, n_inliers, spread, n_attempted) or None.
    """
    if xs.size == 0:
        return None

    def _wmedian(values, weights):
        order = np.argsort(values)
        cum = np.cumsum(weights[order])
        return values[order][np.searchsorted(cum, cum[-1] / 2.0)]

    med_x = _wmedian(xs, rel)
    med_y = _wmedian(ys, rel)
    dist = np.hypot(xs - med_x, ys - med_y)
    inliers = dist <= max(0.2, 3.0 * float(np.median(dist)))
    n_inl = int(inliers.sum())
    if n_inl == 0:
        return None
    cons_x = float(np.average(xs[inliers], weights=rel[inliers]))
    cons_y = float(np.average(ys[inliers], weights=rel[inliers]))
    spread = float(np.median(np.hypot(xs[inliers] - cons_x, ys[inliers] - cons_y)))
    return {
        "x": cons_x,
        "y": cons_y,
        "n_ok": int(xs.size),
        "n_inliers": n_inl,
        "spread": spread,
        "n_attempted": int(n_attempted),
    }


def _consensus_shift(
    ref_geoArr,
    tgt_geoArr,
    candidates,
    band_idx,
    min_win_px=64,
    coarse_candidates=None,
):
    """Estimate one scene's shift as the consensus over many window positions.

    When coarse_candidates is given (adaptive escalation), the coarse set
    is evaluated first and accepted outright only when the result is
    unambiguous (enough successes, >=80% of them agreeing, tight spread).
    Otherwise the remaining full-grid windows are evaluated as well and
    ALL estimates are pooled into one consensus - escalation adds voters,
    it never discards the coarse evidence.

    Returns dict(x, y, n_ok, n_inliers, spread, n_attempted, escalated)
    or None.
    """
    if coarse_candidates is None:
        xs, ys, rel = _estimate_windows(
            ref_geoArr, tgt_geoArr, candidates, band_idx, min_win_px
        )
        cons = _consensus_from_estimates(xs, ys, rel, len(candidates))
        if cons is not None:
            cons["escalated"] = False
        return cons

    xs, ys, rel = _estimate_windows(
        ref_geoArr, tgt_geoArr, coarse_candidates, band_idx, min_win_px
    )
    cons = _consensus_from_estimates(xs, ys, rel, len(coarse_candidates))
    if (
        cons is not None
        and cons["n_ok"] >= _COARSE_MIN_SUCCESS
        and cons["n_inliers"] >= _COARSE_MIN_INLIER_FRACTION * cons["n_ok"]
        and cons["spread"] <= _COARSE_MAX_SPREAD_PX
    ):
        cons["escalated"] = False
        return cons

    # escalate: evaluate only the positions not already tried, pool all
    coarse_wps = {wp for _label, wp in coarse_candidates if wp is not None}
    extra = [
        c for c in candidates if c[1] is not None and c[1] not in coarse_wps
    ]
    xs2, ys2, rel2 = _estimate_windows(
        ref_geoArr, tgt_geoArr, extra, band_idx, min_win_px
    )
    xs = np.concatenate([xs, xs2])
    ys = np.concatenate([ys, ys2])
    rel = np.concatenate([rel, rel2])
    cons = _consensus_from_estimates(
        xs, ys, rel, len(coarse_candidates) + len(extra)
    )
    if cons is not None:
        cons["escalated"] = True
    return cons


# fraction-based agreement thresholds ("auto"): calibrated to reproduce the
# validated absolute defaults at the 7x7 grid (3 and 8 of 50 windows), but
# scaling with however many windows a scene actually attempted - absolute
# counts silently change meaning when the window budget changes (same class
# of bug as the fixed reliability threshold that froze small-AOI references)
_AUTO_KEEP_FRACTION = 0.06
_AUTO_PROMOTE_FRACTION = 0.16
_INLIER_FLOOR = 3


def _resolve_inlier_threshold(value, n_attempted, fraction):
    """'auto' -> max(floor, fraction of the windows attempted for THIS
    scene); an integer -> itself (legacy absolute behavior)."""
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return max(_INLIER_FLOOR, int(np.ceil(fraction * n_attempted)))
    return int(value)


def _check_inlier_param(value, name):
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return "auto"
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= 1 or 'auto'.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be an integer >= 1 or 'auto'.")
    return value


def _warp_scene_yxb(da_yxb, shift_x_px, shift_y_px, geotransform, crs_wkt):
    """Apply one translation to a (y, x, band) scene by shifting its
    coordinates and resampling (cubic) back onto the original grid.

    Verified pixel-identical (RMSE 0.0) to AROSICS
    correct_shifts(align_grids=True) for the same shift; the coordinates
    must be moved (rioxarray derives the source grid from coords, a
    written transform alone is ignored).
    """
    da = da_yxb.transpose("band", "y", "x")
    template = da.rio.write_crs(crs_wkt).rio.write_transform(
        Affine.from_gdal(*geotransform)
    )
    px_w, px_h = geotransform[1], geotransform[5]
    da = da.assign_coords(
        x=da.x.values + shift_x_px * px_w,
        y=da.y.values + shift_y_px * px_h,
    )
    da = da.rio.write_crs(crs_wkt).rio.write_nodata(np.nan)
    out = da.rio.reproject_match(template, resampling=Resampling.cubic)
    return out.assign_coords(x=template.x.values, y=template.y.values)


def _mask_nodata_zeros(da_yxb):
    """Set pixels where ALL bands are exactly 0 (loader nodata gaps) to NaN
    so the warp treats them as nodata instead of bleeding zeros into
    neighbors. Genuine zero values in individual bands are preserved."""
    allzero = (da_yxb == 0).all(dim="band")
    return da_yxb.where(~allzero)


def _load_cloud_mask(mask_obj, times, height, width):
    """Load a binary cloud mask cube (1 = cloud) and align it to the
    spectral cube's time axis.

    Accepts a path (NetCDF/Zarr with 'Cloud_Stack' or a single data
    variable), an xr.Dataset, or an xr.DataArray. The mask file may
    contain MORE dates than the cube (the builder's mask export is not
    filtered by scene_cloud_coverage); every cube date must be present.

    Returns a (time, y, x) DataArray of 0/1.
    """
    if isinstance(mask_obj, str):
        ds = open_cube(mask_obj)
    elif isinstance(mask_obj, xr.Dataset):
        ds = mask_obj
    elif isinstance(mask_obj, xr.DataArray):
        ds = None
    else:
        raise TypeError(
            "cloud_mask must be a file path, xarray.Dataset or xarray.DataArray"
        )

    if ds is not None:
        if "Cloud_Stack" in ds:
            da = ds["Cloud_Stack"]
        else:
            cands = [v for v in ds.data_vars if v != "spatial_ref"]
            if len(cands) != 1:
                raise KeyError(
                    "cloud_mask dataset needs a 'Cloud_Stack' variable "
                    f"(found: {list(ds.data_vars)})"
                )
            da = ds[cands[0]]
    else:
        da = mask_obj

    if "band" in da.dims:
        da = da.isel(band=0)

    if (da.sizes.get("y"), da.sizes.get("x")) != (height, width):
        raise ValueError(
            "cloud_mask grid does not match the cube: "
            f"mask {da.sizes.get('x')}x{da.sizes.get('y')} px vs "
            f"cube {width}x{height} px. Both must cover the same AOI at "
            "the same resolution."
        )

    missing = np.setdiff1d(times, da.time.values)
    if missing.size:
        miss = ", ".join(np.datetime_as_string(m, unit="D") for m in missing[:10])
        raise ValueError(
            f"cloud_mask is missing {missing.size} of the cube's dates "
            f"(e.g. {miss}). Export the mask for the same query."
        )

    return da.sel(time=times).transpose("time", "y", "x").load()


def _auto_anchor_time(data, times, band_idx, cloud_lookup):
    """Pick the best chain anchor scene automatically.

    Candidates: <=10% cloud and <20% missing pixels (falling back to all
    scenes when nothing qualifies). Among them, prefer scenes in the
    CENTRAL HALF of the time range - the chain runs bidirectionally from
    the anchor, so a central anchor halves the worst-case chain length
    and the error accumulation (measured: mid-series anchor gave the
    smallest shifts and best cube). The most textured scene (median
    gradient of the matching band) of the preferred window wins; texture
    is what the matcher locks onto, and clouds/haze destroy it."""
    scored = []
    for t in times:
        band = data.sel(time=t).isel(band=band_idx - 1).values
        gy, gx = np.gradient(band)
        grad = np.hypot(gy, gx)
        finite = np.isfinite(grad)
        texture = float(np.nanmedian(grad[finite])) if finite.any() else 0.0
        nan_frac = 1.0 - float(finite.mean())
        scored.append((t, cloud_lookup(t), texture, nan_frac))

    eligible = [
        s for s in scored if (s[1] is None or s[1] <= 10.0) and s[3] < 0.2
    ]
    if not eligible:
        eligible = scored

    t0, t1 = times[0], times[-1]
    quarter = (t1 - t0) / 4
    central = [s for s in eligible if t0 + quarter <= s[0] <= t1 - quarter]
    if central:
        eligible = central
    return max(eligible, key=lambda s: s[2])[0]


def _estimate_shifts_pass(
    data,  # (time, y, x, band) DataArray, loaded
    times,
    geotransform,
    crs_wkt,
    grid_candidates,  # blind lattice: window budget + placement fallback
    band_idx,
    mode,  # "composite" | "anchor"
    composite_window_days,
    anchor_time,  # scene the chain is anchored at (mode="anchor")
    min_inliers_keep,  # int or "auto" (fraction-based, per-scene)
    min_inliers_update_ref,  # int or "auto"
    max_cloud_update_ref,
    cloud_lookup,  # callable time -> float or None
    min_win_px,
    desc,
    adaptive=False,  # coarse-first scan with escalation
):
    """One estimation pass over the (raw or pre-shifted) cube.

    Chains on UNWARPED scenes: the running reference is the last trusted
    scene plus its accumulated shift, so translations compose by addition
    and the reference is never resampled.

    Window positions are chosen PER SCENE PAIR: fully clear positions in
    both images first (AROSICS fails on any NaN inside the window, so on
    partially cloudy scenes blind lattice positions almost all die), blind
    lattice positions as fallback/top-up. The budget per scene is
    len(grid_candidates).

    mode="anchor" pins the chain at anchor_time (shift 0) and chains
    BIDIRECTIONALLY: forward in time from the anchor, then backward from
    the anchor, so any scene of the series can be the reference.
    mode="composite" matches every scene forward against the median of
    the first composite_window_days days.

    Returns (shifts, dropped, stats):
      shifts: {time -> (abs_x_px, abs_y_px)} for kept scenes
      dropped: [(time, reason), ...]
      stats: {time -> consensus dict}
    """
    shifts, stats, dropped = {}, {}, []

    if mode == "composite":
        first_time = times[0]
        end_time = first_time + np.timedelta64(int(composite_window_days), "D")
        subset = data.sel(time=slice(first_time, end_time))
        if subset.sizes["time"] == 0:
            subset = data
        master = subset.median(dim="time", skipna=True)
        master = master.where(master != 0, np.nan)
        ref_arr = master.values
        legs = [list(range(0, len(times)))]
    elif mode == "anchor":
        matches = np.where(times == anchor_time)[0]
        if matches.size == 0:
            raise ValueError(
                f"anchor scene {np.datetime_as_string(anchor_time, 'D')} "
                "is not in the (filtered) time series."
            )
        a = int(matches[0])
        ref_arr = data.sel(time=times[a]).values
        shifts[times[a]] = (0.0, 0.0)
        legs = [
            list(range(a + 1, len(times))),  # forward in time
            list(range(a - 1, -1, -1)),  # backward in time
        ]
    else:
        raise ValueError("mode must be 'composite' or 'anchor'")

    anchor_geoArr = GeoArray(ref_arr, geotransform=geotransform, projection=crs_wkt)
    anchor_finite = np.isfinite(ref_arr[:, :, band_idx - 1])

    progress = tqdm(total=len(times), initial=len(times) - sum(len(l) for l in legs),
                    desc=desc, unit="scene")
    for leg in legs:
        # every leg starts over from the anchor/composite reference
        ref_geoArr = anchor_geoArr
        ref_finite = anchor_finite
        ref_abs = (0.0, 0.0)
        for idx in leg:
            t = times[idx]
            tgt_arr = data.sel(time=t).values
            tgt_geoArr = GeoArray(
                tgt_arr, geotransform=geotransform, projection=crs_wkt
            )
            tgt_finite = np.isfinite(tgt_arr[:, :, band_idx - 1])

            candidates, n_clear = _place_candidates(
                ref_finite, tgt_finite, geotransform, grid_candidates
            )
            coarse_candidates = (
                candidates[:_COARSE_N]
                if adaptive and len(candidates) > _COARSE_N
                else None
            )

            cons = _consensus_shift(
                ref_geoArr,
                tgt_geoArr,
                candidates,
                band_idx,
                min_win_px=min_win_px,
                coarse_candidates=coarse_candidates,
            )
            progress.update(1)
            if cons is None:
                dropped.append(
                    (t, "no successful matching windows"
                        if n_clear else "no fully clear matching window vs. "
                        "the reference, blind windows all failed")
                )
                continue
            keep_thr = _resolve_inlier_threshold(
                min_inliers_keep, cons["n_attempted"], _AUTO_KEEP_FRACTION
            )
            if cons["n_inliers"] < keep_thr:
                dropped.append(
                    (t, f"only {cons['n_inliers']} agreeing window(s), "
                        f"{keep_thr} needed")
                )
                continue

            abs_x = ref_abs[0] + cons["x"]
            abs_y = ref_abs[1] + cons["y"]
            shifts[t] = (abs_x, abs_y)
            cons["n_clear"] = n_clear
            stats[t] = cons

            cp = cloud_lookup(t)
            promote_thr = _resolve_inlier_threshold(
                min_inliers_update_ref, cons["n_attempted"], _AUTO_PROMOTE_FRACTION
            )
            update_ref = cons["n_inliers"] >= promote_thr and (
                max_cloud_update_ref is None or cp is None or cp <= max_cloud_update_ref
            )
            if update_ref:
                ref_geoArr = tgt_geoArr
                ref_finite = tgt_finite
                ref_abs = (abs_x, abs_y)
    progress.close()

    return shifts, dropped, stats


def _edge_roughness(data_tyxb, band_idx):
    """Internal cube-quality metric for auto-iteration: mean |second
    temporal difference| of the matching band over EDGE pixels (top-decile
    spatial gradient of the temporal median). Misregistration makes
    land-cover boundary pixels flicker between classes, so this drops as
    alignment improves; it is insensitive to uniform seasonal change."""
    v = data_tyxb.isel(band=band_idx - 1).values  # (time, y, x)
    # pixels that are NaN on every date (persistent cloud mask / nodata)
    # legitimately yield NaN means; silence numpy's empty-slice warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(v, axis=0)
        gy, gx = np.gradient(med)
        grad = np.hypot(gy, gx)
        valid = np.isfinite(grad)
        if not valid.any():
            return np.nan
        edge = grad >= np.nanpercentile(grad[valid], 90)
        d2 = np.abs(v[:-2] - 2 * v[1:-1] + v[2:])
        rough = np.nanmean(d2, axis=0)
        return float(np.nanmean(rough[edge]))


# ----------------------------------------------------------------------
# Main function (sliding-grid, consensus)
# ----------------------------------------------------------------------
def coregister_cube(
    input_path,  # str | xr.Dataset | xr.DataArray
    output_path=None,
    stack_name="Time_Series",
    first_scene_mode="composite",
    composite_window_days=30,
    grid_size=7,
    match_band="auto",
    min_inliers_keep="auto",
    min_inliers_update_ref="auto",
    max_cloud_update_ref=20.0,
    max_cc=None,
    time_period=None,
    iteration=1,
    min_win_px=64,
    cloud_mask=None,
    adaptive=True,
    compress=False,
    vrt=False,
    **deprecated,
):
    """Co-register a Sentinel-2 data cube scene-to-scene (AROSICS global).

    Redesigned engine (2026-07): per scene, shifts are ESTIMATED at up to
    grid_size^2 + 1 window positions and combined into a robust consensus
    (outlier windows are outvoted instead of argmax-selected). Window
    positions are CLOUD-AWARE: chosen per scene pair where both images are
    fully clear (AROSICS fails on any NaN inside a window, so blind lattice
    positions die on partially cloudy scenes), with the blind lattice as
    fallback. Scenes chain by summing translations against unwarped
    references; the data is warped exactly ONCE at the end (cubic),
    regardless of `iteration`.

    Parameters
    ----------
    first_scene_mode : "composite" | "first" | "auto" | "YYYY-MM-DD"
        How the chain's reference anchor is chosen. "first" anchors at the
        first scene; "composite" matches everything against the median of
        the first composite_window_days days; "auto" picks the scene with
        the most texture on the matching band among low-cloud scenes
        (snow/haze/clouds reduce texture, which is what makes a bad
        reference); a date string anchors at the nearest scene to that
        date (inspect the cube visually first to pick a clean one). With
        "auto" or a date the chain runs BIDIRECTIONALLY from the anchor.
    match_band : "auto" or band name.
        Band used for matching. "auto" picks the first available native
        10-m band in the order nir, red, green, blue (20-m bands match
        far worse; measured).
    min_inliers_keep : "auto" or int
        A scene is dropped when fewer window positions agree on its shift.
        "auto" = max(3, 6% of the windows attempted for that scene), which
        reproduces the validated absolute default (3 of 50) at the 7x7
        grid but scales with any window budget. An integer is absolute.
    min_inliers_update_ref : "auto" or int
        A scene becomes the reference for the next scene only when at
        least this many windows agree (replaces the old reliability
        thresholds, which did not transfer across AOI sizes).
        "auto" = max(3, 16% of attempted windows) (8 of 50 at the 7x7
        grid). An integer is absolute.
    max_cloud_update_ref : float or None
        Scenes cloudier than this never become the reference.
    iteration : int or "auto"
        Estimation passes. Pass k re-estimates residual shifts on an
        in-memory shifted copy; the OUTPUT is always the original data
        warped once by the total shift (the old behavior resampled the
        cube once per iteration, blurring it).
        "auto" measures the cube after every pass (edge roughness of the
        matching band) and only keeps a refinement pass when it improves
        the cube by more than 2%; otherwise the pass is discarded and
        iteration stops (max 5 passes). With the consensus estimator,
        pass 1 is usually already converged (measured), so "auto"
        typically keeps 1 pass and proves that a second was not needed.
    min_win_px : int
        Discard candidate windows whose actual matching window (after
        AROSICS clips it to the AOI) is smaller than this in either
        dimension.
    cloud_mask : path | xr.Dataset | xr.DataArray or None
        Binary cloud mask cube (1 = cloud, e.g. the builder's SCL mask
        export) for co-registering an UNMASKED (clouds kept) cube: the
        shifts are estimated on an in-memory cloud-masked copy - exactly
        what the masked workflow would estimate - while the exported
        scenes keep their clouds. The mask file may contain more dates
        than the cube; every cube date must be present in it. If the
        cube itself lacks a cloud_percentage coordinate, per-scene cloud
        percentages are derived from the mask for the reference rule.
    adaptive : bool
        Adaptive window escalation (default True): each scene is first
        measured at a coarse set (the 10 best-spread positions) and the
        full window budget runs only when that result is not unambiguous
        (a failure, disagreeing windows, or a loose spread). Clear scenes
        resolve at the coarse stage; cloudy/difficult scenes automatically
        get the full effort. Set False to always use the full budget.

    Deprecated and ignored (accepted for old configs/GUIs):
    min_reliability_keep, min_reliability_update_ref.

    Returns (out_ds, output_path).
    """
    _DEPRECATED = {"min_reliability_keep", "min_reliability_update_ref"}
    unknown = set(deprecated) - _DEPRECATED
    if unknown:
        raise TypeError(f"coregister_cube() got unexpected arguments: {sorted(unknown)}")
    if deprecated:
        warnings.warn(
            f"{sorted(set(deprecated) & _DEPRECATED)} are deprecated and ignored: "
            "scene keep/reference decisions now use consensus inlier counts "
            "(min_inliers_keep / min_inliers_update_ref).",
            stacklevel=2,
        )

    AUTO_MAX_PASSES = 5
    AUTO_MIN_IMPROVE = 0.02  # a pass must improve edge roughness by > 2%

    auto_iteration = isinstance(iteration, str) and iteration.lower() == "auto"
    if auto_iteration:
        iteration = AUTO_MAX_PASSES
    else:
        if isinstance(iteration, bool) or not isinstance(iteration, (int, np.integer)):
            raise TypeError("iteration must be an integer >= 1 or 'auto'.")
        iteration = int(iteration)
        if iteration < 1:
            raise ValueError("iteration must be an integer >= 1 (cannot be 0).")

    min_inliers_keep = _check_inlier_param(min_inliers_keep, "min_inliers_keep")
    min_inliers_update_ref = _check_inlier_param(
        min_inliers_update_ref, "min_inliers_update_ref"
    )

    # ------------------------------------------------------------------
    # Load + filter
    # ------------------------------------------------------------------
    stac, masked_stack, cloud_pct_da, input_path_str = _load_coreg_input(
        input_path, stack_name=stack_name
    )
    input_crs_attr = masked_stack.attrs.get("crs", None)
    filtered = _apply_time_and_cloud_filters(
        masked_stack, max_cc=max_cc, time_period=time_period
    )

    crs_wkt = _get_crs_wkt(filtered, ds=stac)
    filtered = filtered.rio.write_crs(crs_wkt, inplace=True)
    geotransform = _get_geotransform(filtered, ds=stac)

    times = filtered.time.values
    if times.size == 0:
        raise ValueError("No scenes left after applying max_cc/time_period filters.")

    band_idx, band_label = _resolve_match_band(filtered.band.values, match_band)

    height = filtered.sizes["y"]
    width = filtered.sizes["x"]
    if min(height, width) < 256:
        print(
            f"Warning: the AOI is only {width}x{height} px. Matching accuracy "
            "is texture-limited on small areas (measured ~0.3 px residual on "
            "a 287x124 px AOI); a larger AOI gives better co-registration."
        )

    candidates = _grid_candidates(geotransform, height, width, grid_size)

    def _cloud_lookup(t):
        if cloud_pct_da is None:
            return None
        try:
            return (
                float(cloud_pct_da.sel(time=t))
                if "time" in cloud_pct_da.dims
                else float(cloud_pct_da)
            )
        except Exception:
            return None

    # one in-memory copy in (time, y, x, band) order for fast per-scene access
    data = filtered.transpose("time", "y", "x", "band").load()

    # optional cloud mask: estimate on a masked copy, warp the original.
    mask_da = None
    if cloud_mask is not None:
        mask_da = _load_cloud_mask(cloud_mask, times, height, width)
        if cloud_pct_da is None or "time" not in getattr(cloud_pct_da, "dims", ()):
            # derive per-scene cloud percentage from the mask so the
            # update-reference rule still works on keep-clouds cubes
            cloud_pct_da = (mask_da.mean(dim=("y", "x")) * 100.0).rename(
                "cloud_percentage"
            )

    est_source = data.where(mask_da == 0) if mask_da is not None else data

    # resolve first_scene_mode into (pass-1 mode, anchor scene)
    fsm = "composite" if first_scene_mode is None else str(first_scene_mode)
    if fsm == "composite":
        pass1_mode, anchor_time = "composite", None
    elif fsm == "first":
        pass1_mode, anchor_time = "anchor", times[0]
    elif fsm.lower() == "auto":
        anchor_time = _auto_anchor_time(est_source, times, band_idx, _cloud_lookup)
        pass1_mode = "anchor"
        cp_a = _cloud_lookup(anchor_time)
        print(
            "Auto-selected reference scene: "
            f"{np.datetime_as_string(anchor_time, 'D')}"
            + (f" (cloud {cp_a:.1f}%)" if cp_a is not None else "")
            + " - most textured low-cloud scene on the matching band."
        )
    else:
        try:
            query = np.datetime64(fsm)
        except Exception:
            raise ValueError(
                "first_scene_mode must be 'first', 'composite', 'auto' or a "
                f"date 'YYYY-MM-DD'; got {first_scene_mode!r}."
            )
        anchor_time = times[int(np.argmin(np.abs(times - query)))]
        pass1_mode = "anchor"
        print(
            f"Reference scene for '{fsm}': "
            f"{np.datetime_as_string(anchor_time, 'D')} (nearest available scene)."
        )

    print(
        f"Co-registration: {times.size} scenes, matching on band "
        f"'{band_label}', cloud-aware window placement, "
        + (
            f"adaptive scan ({_COARSE_N} coarse -> "
            f"{len(candidates)} window positions/scene)"
            if adaptive and len(candidates) > _COARSE_N
            else f"up to {len(candidates)} window positions/scene"
        )
        + ", "
        + (
            f"auto iterations (max {AUTO_MAX_PASSES})"
            if auto_iteration
            else f"{iteration} estimation pass(es)"
        )
        + (", clouds masked for estimation only" if mask_da is not None else "")
        + "."
    )

    # ------------------------------------------------------------------
    # Estimation passes (no warping of the output data here)
    # ------------------------------------------------------------------
    total_shifts = {}  # time -> (x_px, y_px)
    dropped_all = []
    last_stats = {}
    est_data = est_source
    est_times = times
    mode = pass1_mode
    quality_prev = None

    for it in range(1, iteration + 1):
        shifts, dropped, stats = _estimate_shifts_pass(
            est_data,
            est_times,
            geotransform,
            crs_wkt,
            candidates,
            band_idx,
            mode,
            composite_window_days,
            anchor_time,
            min_inliers_keep,
            min_inliers_update_ref,
            max_cloud_update_ref,
            _cloud_lookup,
            min_win_px,
            desc=(
                f"Estimating shifts (pass {it}"
                + ("/auto)" if auto_iteration else f"/{iteration})")
            ),
            adaptive=adaptive,
        )

        cand_times = np.array([t for t in est_times if t in shifts])
        if cand_times.size == 0:
            raise RuntimeError("No scenes were kept. Output stack would be empty.")
        cand_totals = dict(total_shifts)
        for t, (sx, sy) in shifts.items():
            px, py = cand_totals.get(t, (0.0, 0.0))
            cand_totals[t] = (px + sx, py + sy)

        # shifted in-memory copy: input of the next pass, and in auto mode
        # the object the quality guard measures. The final OUTPUT is still
        # warped once from the ORIGINAL data.
        shifted_data = None
        if auto_iteration or it < iteration:
            shifted = []
            for t in cand_times:
                da = _warp_scene_yxb(
                    est_source.sel(time=t), *cand_totals[t], geotransform, crs_wkt
                ).transpose("y", "x", "band")
                shifted.append(da.assign_coords(time=t))
            # coords="different" (today's default, pinned explicitly so the
            # planned xarray default change cannot alter behavior): per-scene
            # scalar coords like cloud_percentage differ and must be
            # concatenated along time
            shifted_data = xr.concat(
                shifted, dim="time", coords="different", compat="equals"
            )

        # auto mode: accept a refinement pass only when it measurably
        # improves the cube. All refinement strategies tested re-measure
        # estimation noise once the consensus has converged (usually after
        # pass 1), so the guard is on cube quality itself, not on the size
        # of the corrections.
        if auto_iteration:
            q = _edge_roughness(shifted_data, band_idx)
            if it == 1:
                q0 = _edge_roughness(est_source.sel(time=cand_times), band_idx)
                print(
                    f"Pass 1: edge roughness {q0:.4f} -> {q:.4f} "
                    f"({(q0 - q) / q0:+.1%})"
                )
            else:
                improve = (quality_prev - q) / quality_prev if quality_prev else 0.0
                print(f"Pass {it}: edge roughness {q:.4f} ({improve:+.1%} vs previous)")
                if improve <= AUTO_MIN_IMPROVE:
                    print(
                        f"Auto-iteration: converged - pass {it} did not improve "
                        f"the cube, keeping {it - 1} pass(es)."
                    )
                    break  # discard this pass entirely
            quality_prev = q

        # accept the pass
        total_shifts = cand_totals
        est_times = cand_times
        dropped_all.extend(dropped)
        last_stats.update(stats)

        if it < iteration:
            est_data = shifted_data
            if mode == "composite":
                # refinement passes need a concrete anchor scene
                mode = "anchor"
                anchor_time = est_times[0]

    # ------------------------------------------------------------------
    # Single final warp from the original data
    # ------------------------------------------------------------------
    corrected_images = []
    for t in tqdm(est_times, desc="Applying shifts (single warp)", unit="scene"):
        scene = _mask_nodata_zeros(data.sel(time=t))
        da = _warp_scene_yxb(scene, *total_shifts[t], geotransform, crs_wkt)
        corrected_images.append(da.assign_coords(time=t))

    corrected_stack = xr.concat(
        corrected_images, dim="time", coords="different", compat="equals"
    ).transpose("time", "band", "y", "x")
    corrected_stack = corrected_stack.rio.write_crs(crs_wkt, inplace=True)
    if input_crs_attr is not None:
        corrected_stack.attrs["crs"] = input_crs_attr
    corrected_stack.name = "Time_Series"

    out_ds = xr.Dataset({"Time_Series": corrected_stack})
    if stac is not None and "spatial_ref" in stac.variables:
        out_ds["spatial_ref"] = stac["spatial_ref"]
    if cloud_pct_da is not None and "time" in getattr(cloud_pct_da, "dims", ()):
        out_ds = out_ds.assign_coords(
            cloud_percentage=cloud_pct_da.sel(time=corrected_stack.time.values)
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    times_out = corrected_stack.time.values
    print("\nCo-registration summary")
    print("-----------------------")
    print(
        f"Original (after max_cc/time_period): {len(times)} scenes from "
        f"{np.datetime_as_string(times[0], 'D')} to {np.datetime_as_string(times[-1], 'D')}"
    )
    reason_by_time = {t: r for t, r in dropped_all}
    print(
        "Scenes excluded after co-registration:",
        len(reason_by_time),
    )
    print(f"Scenes remaining in the co-registered cube: {len(times_out)}")

    if reason_by_time:
        print("Excluded dates:")
        for ts in sorted(reason_by_time):
            ds_ = np.datetime_as_string(ts, unit="D")
            cp = _cloud_lookup(ts)
            cloud_txt = f", cloud {cp:.1f}%" if cp is not None else ""
            print(f"  {ds_} ({reason_by_time[ts]}{cloud_txt})")

    if last_stats:
        inl = np.array([s["n_inliers"] for s in last_stats.values()], dtype=float)
        spr = np.array([s["spread"] for s in last_stats.values()], dtype=float)
        print(
            f"\nWindow agreement of kept scenes: mean {inl.mean():.1f} inlier "
            f"windows/scene (min {int(inl.min())}), median spread {np.median(spr):.2f} px"
        )
        if adaptive:
            n_esc = sum(1 for s in last_stats.values() if s.get("escalated"))
            print(
                f"Adaptive scan: {len(last_stats) - n_esc}/{len(last_stats)} scenes "
                f"resolved at the coarse stage, {n_esc} escalated to the full budget"
            )
        n_cl = np.array(
            [s.get("n_clear", 0) for s in last_stats.values()], dtype=float
        )
        print(
            f"Window placement: mean {n_cl.mean():.1f} fully clear window "
            f"positions/scene ({int((n_cl == 0).sum())} scene(s) fell back to "
            f"blind windows)"
        )
        mags = np.array([np.hypot(*total_shifts[t]) for t in est_times])
        print(
            f"Applied shifts: mean {mags.mean():.2f} px, max {mags.max():.2f} px "
            f"(single cubic warp; data resampled once)"
        )

    print("\nS2 co-registration is completed!")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    final_out_path = output_path
    if final_out_path is None and input_path_str is not None:
        final_out_path = _auto_output_path(input_path_str, suffix="_cr")

    if final_out_path is not None:
        _write_coreg_output(out_ds, final_out_path, compress=compress, vrt=vrt)
        print(f"\nCo-registered cube written to: {final_out_path}")
    else:
        print(
            "\nNo output_path provided and input was not a file path -> skipping export."
        )

    return out_ds, final_out_path


# ----------------------------------------------------------------------
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go


def _load_stac(path, stack_name="Time_Series"):
    with open_cube(path) as ds:
        return ds[stack_name].load()


def _band_label(stac, name):
    b = stac.coords["band"].values
    if b.dtype.kind in ("U", "S", "O"):
        bl = np.array([str(x).lower() for x in b])
        m = np.where(bl == name.lower())[0]
        if m.size:
            return b[m[0]]
    raise KeyError(
        f"band='{name}' not found. Available bands: {list(stac.coords['band'].values)}"
    )


def _pick_rgb(stac):
    b = stac.coords["band"].values
    if b.dtype.kind in ("U", "S", "O"):
        bl = np.array([str(x).lower() for x in b])

        def pick(cands):
            for c in cands:
                m = np.where(bl == c)[0]
                if m.size:
                    return b[m[0]]
            return None

        r = pick(["red", "r", "b04"])
        g = pick(["green", "g", "b03"])
        bb = pick(["blue", "b", "b02"])
        if r is not None and g is not None and bb is not None:
            return stac.sel(band=[r, g, bb])

    # fallback: first 3 bands
    return stac.isel(band=[0, 1, 2])


def _stretch_to_uint8(rgb_yxb, p2=2, p98=98):
    arr = rgb_yxb.values.astype("float32")  # (y,x,3)
    lo = np.nanpercentile(arr, p2, axis=(0, 1))
    hi = np.nanpercentile(arr, p98, axis=(0, 1))
    img = (arr - lo) / (hi - lo + 1e-12)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _to_dmy(time_values):
    """Convert a numpy datetime64 array to a list of 'dd.mm.yyyy' strings."""
    return [
        "{2}.{1}.{0}".format(*np.datetime_as_string(t, unit="D").split("-"))
        for t in time_values
    ]


def spectral_profiler(
    before_path, after_path, band="ndvi",
    stack_name="Time_Series", rgb_time="first"
):
    stac_b = _load_stac(before_path, stack_name)
    stac_a = _load_stac(after_path, stack_name)

    band_label = str(band)
    band_b = _band_label(stac_b, band_label)
    band_a = _band_label(stac_a, band_label)

    # RGB for click map (from BEFORE)
    rgb = _pick_rgb(stac_b)
    if rgb_time == "median":
        rgb_base = rgb.median("time", skipna=True).transpose("y", "x", "band")
    else:
        rgb_base = rgb.isel(time=0).transpose("y", "x", "band")

    rgb_img = _stretch_to_uint8(rgb_base)

    # Plotly image widget (clickable)
    fig_img = go.FigureWidget(data=[go.Image(z=rgb_img)])
    fig_img.update_layout(
        title="Click a pixel / Use zoom tools --->",
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
        width=650,
    )

    # NDVI time-series widget
    fig_ts = go.FigureWidget()
    fig_ts.add_scatter(
        name="before (non-registered)",
        x=[],
        y=[],
        mode="lines",
        line=dict(color="gold", width=3),
    )
    fig_ts.add_scatter(
        name="after (co-registered)",
        x=[],
        y=[],
        mode="lines",
        line=dict(color="blue", width=3),
    )
    fig_ts.update_layout(
        title=dict(
            text=f"{band_label.upper()} Spectral Profile",
            x=0.5,
            xanchor="center",
            pad=dict(t=10, b=20),
        ),
        xaxis=dict(
            title="time",
            tickangle=-45,   # <-- incline labels so they don't overlap
            type="category", # <-- treat x as categorical strings, not numbers
        ),
        yaxis_title=band_label.upper(),
        margin=dict(l=40, r=10, t=110, b=80),  # extra bottom margin for angled labels
        height=450,
        width=650,
        legend=dict(
            orientation="h",
            xanchor="left",
            x=0,
            yanchor="top",
            y=1.08,
        ),
    )

    out = widgets.Output()

    x_vals = stac_b.coords["x"].values
    y_vals = stac_b.coords["y"].values

    def update_from_rowcol(row, col):
        # map indices -> map coords from BEFORE cube
        x0 = float(x_vals[col])
        y0 = float(y_vals[row])

        s_b = stac_b.sel(band=band_b).sel(x=x0, y=y0, method="nearest")
        s_a = stac_a.sel(band=band_a).sel(x=x0, y=y0, method="nearest")

        with fig_ts.batch_update():
            fig_ts.data[0].x = _to_dmy(s_b.time.values)  # <-- converted to dd.mm.yyyy
            fig_ts.data[0].y = s_b.values
            fig_ts.data[1].x = _to_dmy(s_a.time.values)  # <-- converted to dd.mm.yyyy
            fig_ts.data[1].y = s_a.values

        with out:
            out.clear_output(wait=True)
            print(f"Clicked pixel: row={row}, col={col} | x={x0}, y={y0}")

    # Click handler
    def handle_click(trace, points, state):
        # robust extraction of clicked coordinates
        if hasattr(points, "xs") and points.xs and hasattr(points, "ys") and points.ys:
            col = int(np.clip(round(points.xs[0]), 0, rgb_img.shape[1] - 1))
            row = int(np.clip(round(points.ys[0]), 0, rgb_img.shape[0] - 1))
        elif hasattr(points, "point_inds") and points.point_inds:
            # flattened index fallback
            ind = int(points.point_inds[0])
            row = ind // rgb_img.shape[1]
            col = ind % rgb_img.shape[1]
        else:
            return

        update_from_rowcol(row, col)

    fig_img.data[0].on_click(handle_click)

    # initialize with center pixel so plot is not empty
    update_from_rowcol(rgb_img.shape[0] // 2, rgb_img.shape[1] // 2)

    # Stack the click map and the time-series plot vertically so both fit inside
    # a narrow GUI panel (side-by-side pushed the graph past the panel border).
    ui = widgets.VBox([fig_img, fig_ts, out])
    display(ui)
    return ui
