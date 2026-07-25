import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display, clear_output
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image, ImageDraw, ImageFont
import re


# ==========================================================
# COORDS: extent + origin for projected coords (UTM)
# ==========================================================
def _get_extent_and_origin(stac_mode: xr.DataArray):
    """
    Build extent for imshow() from projected x/y coordinates (e.g., UTM).
    Returns (extent, origin) so the image is shown north-up.
    """
    x = stac_mode["x"].values
    y = stac_mode["y"].values

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    # If y is decreasing (common in rasters), origin should be "upper"
    origin = "upper" if y[0] > y[-1] else "lower"
    extent = [xmin, xmax, ymin, ymax]

    return extent, origin


# ==========================================================
# LAZY DETECTION (Dask-backed)
# ==========================================================
def _is_lazy_xarray(da: xr.DataArray) -> bool:
    """
    True if the DataArray is backed by a lazy array (typically dask).
    """
    data = da.data
    # Avoid hard dependency on dask
    return hasattr(data, "compute") and not isinstance(data, np.ndarray)


# ==========================================================
# BAND SELECTION
# ==========================================================
def _select_mode(stac: xr.DataArray, display_mode: str) -> xr.DataArray:
    dm = str(display_mode).lower().strip()

    if dm == "rgb":
        return stac.sel(band=["red", "green", "blue"])

    elif dm == "false_color":
        # Classic CIR: NIR, Red, Green (we map to RGB later)
        return stac.sel(band=["nir", "red", "green"])

    elif dm in ["ndvi", "ndwi"]:
        return stac.sel(band=dm)

    else:
        raise ValueError(f"Unknown display_mode: {display_mode}")


# ==========================================================
# OPTIONAL CROP (projected coords)
# ==========================================================
def _apply_crop(stac_mode: xr.DataArray, crop):
    """
    crop = (xmin, xmax, ymin, ymax) in projected coords
    Handles ascending or descending y.
    """
    if crop is None:
        return stac_mode

    xmin, xmax, ymin, ymax = crop
    y0, y1 = float(stac_mode.y.values[0]), float(stac_mode.y.values[-1])

    if y0 > y1:
        # descending y
        return stac_mode.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    else:
        # ascending y
        return stac_mode.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))


# ==========================================================
# MISSING FRAME DETECTION
# ==========================================================
def _missing_frame(
    arr: np.ndarray, nan_fraction_thresh=0.9, variance_thresh=1e-12,
    min_finite=None, finite_mask=None
) -> bool:
    """
    Robust missing test:
    - too few valid pixels
    - or near-constant frame (all zeros / no signal)

    ``min_finite`` is an absolute number of finite pixels below which the frame
    counts as missing. When given it REPLACES the whole-array NaN-fraction
    test, which is meaningless for a sparse AOI: a polygon following a river
    covers a few percent of its bounding box, so every clipped frame is >90%
    NaN by construction and the fraction test would call the whole cube
    missing. Callers pass a fraction of the cube's own valid-pixel footprint
    here (see _FootprintTracker), which is the same footprint-relative logic
    compute_cloud_percentage already uses.

    ``finite_mask`` lets a caller that has already computed ``np.isfinite(arr)``
    hand it over instead of having it recomputed here (_FootprintTracker needs
    the same mask to track the footprint, so it was being built twice per
    redraw - ~19 ms of the frame budget on a large cube).
    """
    if arr.size == 0:
        return True

    if finite_mask is None:
        finite_mask = np.isfinite(arr)
    n_finite = int(finite_mask.sum())
    if n_finite == 0:
        return True

    if min_finite is None:
        if 1.0 - n_finite / arr.size >= nan_fraction_thresh:
            return True
    elif n_finite < max(int(min_finite), _MIN_ABS_FINITE):
        return True

    if np.nanvar(arr[finite_mask]) <= variance_thresh:
        return True

    return False


# A frame carrying fewer valid pixels than this is missing whatever the
# footprint says: it guards the first-frame bootstrap of _FootprintTracker,
# where the running maximum is still 0 and every threshold would be 0.
_MIN_ABS_FINITE = 64

# Share of the observed footprint a frame must reach to count as imaged.
_FOOTPRINT_MISSING_FRAC = 0.05


class _FootprintTracker:
    """Running valid-pixel footprint of a cube, learned from rendered frames.

    The AOI footprint is not known up front without reading the whole cube, so
    it is approximated by the largest finite-pixel count seen so far, per frame
    shape (a 2D single-band frame and a 3D RGB frame of the same cube have
    different counts). The maximum only grows, so the estimate converges to the
    true footprint as soon as one well-imaged scene has been displayed.

    This keeps the two cases the NaN-fraction test used to cover: a fully
    cloud-masked or empty scene still has ~no valid pixels and reads missing,
    while a sparse-AOI scene is judged against the AOI, not the bounding box.
    """

    def __init__(self, frac: float = _FOOTPRINT_MISSING_FRAC):
        self._max_finite = {}
        self._frac = float(frac)

    def is_missing(self, arr: np.ndarray, variance_thresh=1e-12) -> bool:
        if arr is None or arr.size == 0:
            return True
        key = arr.shape
        # Computed once and passed on: _missing_frame needs the very same mask.
        finite_mask = np.isfinite(arr)
        n_finite = int(finite_mask.sum())
        seen = max(self._max_finite.get(key, 0), n_finite)
        self._max_finite[key] = seen
        return _missing_frame(
            arr,
            variance_thresh=variance_thresh,
            min_finite=self._frac * seen,
            finite_mask=finite_mask,
        )


# ==========================================================
# SCALING POLICY
# ==========================================================
def _get_scaling_policy(data_stac: xr.DataArray, display_mode: str):
    dm = str(display_mode).lower().strip()

    if dm in ["rgb", "false_color"]:
        return {
            "rgb_p_low": 2,
            "rgb_p_high": 98,
            "rgb_auto_gain": True,
            "rgb_target_luma": 0.38,
            "rgb_gain_min": 0.9,
            "rgb_gain_max": 1.25,
            "rgb_gamma": 1.0,
        }

    # NDVI/NDWI: the index range, used as the fallback only.
    #
    # These used to be percentiles of the WHOLE cube whenever it was not lazy,
    # which read every scene of every band just to pick two numbers - the same
    # trap the cloud percentage had. The viewer now derives vmin/vmax from the
    # displayed frame and the stretch slider (see _render_current), so the
    # cube-wide read bought nothing and is gone. What is left is the honest
    # default for a normalized difference index, and it costs no I/O.
    return {"vmin": -1.0, "vmax": 1.0}


# ==========================================================
# RGB NORMALIZATION (per-frame, robust)
# ==========================================================
# Pixels the percentile stretch is estimated from. The stretch only needs the
# shape of the histogram, and a regular subsample of a scene reproduces it: at
# this size the 2nd/98th percentiles of a full 8.5 MP frame and of its stride-4
# subsample agree to ~2e-4 on a 0-1 scale, i.e. below one uint8 step. Every
# DISPLAYED pixel is still stretched and drawn at full resolution - only the two
# bounds are estimated from the sample.
_STRETCH_SAMPLE_PX = 500_000


def _stretch_bounds(band: np.ndarray, p_low: float, p_high: float):
    """(lo, hi) percentile bounds of one band, estimated from a subsample.

    Both percentiles come from a SINGLE np.nanpercentile call: each call sorts
    the array, so asking for them separately sorted the same data twice.

    Non-finite values are dropped from the sample rather than from the whole
    band: sanitizing the full array allocated a copy per band (34 MB each on a
    large frame) purely to feed the percentile. +/-inf pixels are excluded from
    the displayed image anyway by the caller's `valid` mask, so keeping them in
    the clip below changes nothing.
    """
    npix = band.size
    if npix > _STRETCH_SAMPLE_PX:
        step = int(np.ceil(np.sqrt(npix / _STRETCH_SAMPLE_PX)))
        sample = band[::step, ::step]
    else:
        sample = band

    sample = sample[np.isfinite(sample)]
    if sample.size:
        lo, hi = np.nanpercentile(sample, [p_low, p_high])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return float(lo), float(hi)

    # Degenerate sample (all non-finite, or flat). A sparse AOI can land a
    # subsample entirely outside the valid area, so fall back to the full band
    # before giving up - correctness first, the slow path is the rare one.
    full = band[np.isfinite(band)]
    if full.size:
        lo, hi = np.nanpercentile(full, [p_low, p_high])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return float(lo), float(hi)
    return None


def _normalize_rgb_frame(rgb_yxb: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    """
    Per-frame robust RGB normalization:
    - percentile clip per band (bounds estimated from a subsample, see
      _stretch_bounds - the displayed pixels are all still stretched)
    - valid pixels only if all 3 bands finite
    - invalid pixels -> neutral gray (prevents random red/blue speckles)
    Returns float RGB in [0,1]
    """
    rgb = rgb_yxb.astype(np.float32, copy=False)

    valid = np.all(np.isfinite(rgb), axis=2)
    out = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        band = rgb[:, :, i]

        bounds = _stretch_bounds(band, p_low, p_high)
        if bounds is None:
            out[:, :, i] = 0.5
            continue
        lo, hi = bounds

        out[:, :, i] = (np.clip(band, lo, hi) - lo) / (hi - lo)

    out[~valid] = 0.5
    # In-place: the values are already inside [0, 1] by construction (a clipped
    # band mapped through (v-lo)/(hi-lo), plus the 0.5 fill), so this only
    # guards against float edge cases - no reason to allocate a second full
    # frame (100 MB on a large cube) to do it.
    return np.clip(out, 0, 1, out=out)


def _rgb_to_uint8(
    rgb_01: np.ndarray, gamma: float = 1.0, gain: float = 1.0
) -> np.ndarray:
    """
    Convert RGB float [0,1] -> uint8 with gain + optional gamma.
    """
    rgb_01 = np.clip(rgb_01, 0, 1)
    rgb_01 = np.clip(rgb_01 * gain, 0, 1)

    if gamma is not None and gamma > 0 and gamma != 1.0:
        rgb_01 = np.power(rgb_01, gamma)

    return (rgb_01 * 255).astype(np.uint8)


# ==========================================================
# NDVI/NDWI -> RGB image
# ==========================================================
def _nd_to_rgb_uint8(
    data: np.ndarray, cmap_name: str, vmin: float, vmax: float
) -> np.ndarray:
    """
    Convert a 2D NDVI/NDWI array into an RGB uint8 image using a colormap.
    """
    import matplotlib.colors as colors

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # matplotlib.cm.get_cmap was removed in 3.9; matplotlib.colormaps works in 3.5+
    try:
        import matplotlib as mpl

        cmap = mpl.colormaps[cmap_name]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(data))  # float RGBA in [0,1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


# ==========================================================
# FRAME RENDERING (single time index)
# ==========================================================
def _render_frame_as_uint8(
    stac_mode: xr.DataArray, display_mode: str, idx: int, scaling, raw=None,
    is_missing=None
):
    """
    Returns a uint8 RGB image.
    Lazy-safe: computes ONLY the selected time slice.

    ``raw`` optionally carries the already-computed pixel values of this slice
    ((y, x, band) for rgb/false_color, (y, x) for ndvi/ndwi) so a caller-side
    frame cache can restyle a scene without re-reading the data.

    ``is_missing`` optionally replaces the default missing-frame test with a
    footprint-aware one (see _FootprintTracker.is_missing).
    """
    dm = str(display_mode).lower().strip()
    if is_missing is None:
        is_missing = _missing_frame

    if dm in ["rgb", "false_color"]:
        if raw is not None:
            rgb = raw
        else:
            frame = stac_mode.isel(time=idx).transpose("y", "x", "band")
            rgb = frame.values  # lazy -> computes only this slice

        if is_missing(rgb):
            return None

        # For false_color, stac_mode bands are [nir, red, green] already.
        # That means the "rgb" array here is actually (R=nir, G=red, B=green) — perfect.
        rgb01 = _normalize_rgb_frame(
            rgb,
            p_low=scaling.get("rgb_p_low", 2),
            p_high=scaling.get("rgb_p_high", 98),
        )

        gain = 1.0
        if scaling.get("rgb_auto_gain", True):
            luma = float(np.mean(rgb01))
            target = float(scaling.get("rgb_target_luma", 0.38))
            if np.isfinite(luma) and luma > 1e-6:
                gain = target / luma
                gain = float(
                    np.clip(
                        gain,
                        scaling.get("rgb_gain_min", 0.9),
                        scaling.get("rgb_gain_max", 1.25),
                    )
                )

        return _rgb_to_uint8(
            rgb01,
            gamma=scaling.get("rgb_gamma", 1.0),
            gain=gain,
        )

    # NDVI / NDWI
    if raw is not None:
        data = raw
    else:
        frame = stac_mode.isel(time=idx)
        data = frame.values  # lazy -> computes only this slice

    if is_missing(data):
        return None

    cmap = "RdYlGn" if dm == "ndvi" else "Blues"
    return _nd_to_rgb_uint8(
        data, cmap_name=cmap, vmin=scaling["vmin"], vmax=scaling["vmax"]
    )


# ---------------- HELPERS ----------------
def _band_key(da: xr.DataArray, name: str):
    if "band" not in da.coords:
        raise KeyError("No 'band' coordinate found on the DataArray.")
    b = da.coords["band"].values
    bl = np.array([str(x).lower() for x in b])
    m = np.where(bl == name.lower())[0]
    if m.size == 0:
        raise KeyError(f"band '{name}' not found. Available: {list(b)}")
    return b[m[0]]


def _stretch_uint8(a2d: np.ndarray, p_low=2, p_high=98, gamma=1.0):
    a = a2d.astype("float32", copy=False)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, dtype=np.uint8)

    lo = np.nanpercentile(a[finite], p_low)
    hi = np.nanpercentile(a[finite], p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(a[finite])
        hi = np.nanmax(a[finite])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros(a.shape, dtype=np.uint8)

    x = (a - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    if gamma and gamma != 1.0:
        x = np.power(x, 1.0 / float(gamma))
    x[~np.isfinite(x)] = 0.0
    return (x * 255.0 + 0.5).astype(np.uint8)


def _norm_uint8(a2d: np.ndarray, vmin: float, vmax: float):
    a = a2d.astype("float32", copy=False)
    x = (a - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    x[~np.isfinite(x)] = 0.0
    return (x * 255.0 + 0.5).astype(np.uint8)


def _load_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _resample_lanczos():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def _format_date_ddmmyyyy(t) -> str:
    if isinstance(t, np.datetime64):
        s = np.datetime_as_string(t, unit="D")
    else:
        s = str(t)

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{d}.{mo}.{y}"

    m = re.search(r"(\d{4})/(\d{2})/(\d{2})", s)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{d}.{mo}.{y}"

    return s[:10]


def _add_top_label(
    im: Image.Image, txt: str, font_scale=0.03, font_min=14, font_max=48, bar_pad=None
):
    """
    Adds a top label bar WITHOUT covering the image.
    Returns a new image with extra height (bar + original image).
    """
    w, h = im.size

    # font size based on image width
    fs = int(round(w * float(font_scale)))
    fs = max(font_min, min(font_max, fs))
    font = _load_font(fs)

    if bar_pad is None:
        bar_pad = max(6, int(fs * 0.35))

    # measure text (works across Pillow versions)
    tmp = Image.new("RGB", (1, 1), (0, 0, 0))
    draw_tmp = ImageDraw.Draw(tmp)

    while True:
        try:
            bbox = draw_tmp.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = draw_tmp.textsize(txt, font=font)

        if tw <= w - 2 * bar_pad or fs <= 10:
            break
        fs -= 2
        font = _load_font(fs)
        bar_pad = max(6, int(fs * 0.35))

    bar_h = th + 2 * bar_pad

    # create new canvas (bar on top, image below)
    out = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
    out.paste(im, (0, bar_h))

    draw = ImageDraw.Draw(out)

    # bar background (already black, but keep explicit)
    draw.rectangle([0, 0, w, bar_h], fill=(0, 0, 0))

    # centered text in the bar
    x = (w - tw) // 2
    y = (bar_h - th) // 2
    draw.text((x, y), txt, fill=(255, 255, 255), font=font)

    return out


# ---------------- COLOR LUTS (no matplotlib) ----------------
def _make_piecewise_lut(points):
    """
    points: list of (pos, (r,g,b)) where pos in [0,255]
    returns lut shape (256,3) uint8
    """
    pts = sorted(points, key=lambda x: x[0])
    lut = np.zeros((256, 3), dtype=np.float32)
    for (p0, c0), (p1, c1) in zip(pts[:-1], pts[1:]):
        p0 = int(p0)
        p1 = int(p1)
        if p1 <= p0:
            continue
        c0 = np.array(c0, dtype=np.float32)
        c1 = np.array(c1, dtype=np.float32)
        t = np.linspace(0, 1, p1 - p0 + 1)[:, None]
        lut[p0 : p1 + 1] = c0 + (c1 - c0) * t
    # fill before first and after last
    lut[: pts[0][0]] = np.array(pts[0][1], dtype=np.float32)
    lut[pts[-1][0] :] = np.array(pts[-1][1], dtype=np.float32)
    return np.clip(lut + 0.5, 0, 255).astype(np.uint8)


# NDVI: brown -> yellow -> green -> dark green
_NDVI_LUT = _make_piecewise_lut(
    [
        (0, (70, 40, 20)),  # -1
        (128, (220, 200, 80)),  # 0
        (180, (90, 190, 70)),  # ~0.4
        (255, (0, 90, 0)),  # 1
    ]
)

# NDWI: land (tan/gray) -> light cyan -> blue -> dark blue
_NDWI_LUT = _make_piecewise_lut(
    [
        (0, (120, 95, 70)),  # -1
        (128, (210, 210, 210)),  # 0
        (170, (170, 235, 245)),  # ~0.33
        (220, (60, 150, 225)),  # ~0.72
        (255, (0, 50, 130)),  # 1
    ]
)


def _apply_lut(u8: np.ndarray, lut: np.ndarray, nodata_mask: np.ndarray | None = None):
    rgb = lut[u8]  # (y,x,3)
    if nodata_mask is not None and nodata_mask.any():
        rgb = rgb.copy()
        rgb[nodata_mask] = 0
    return rgb


# ---------------- FRAME MAKER ----------------
def make_frame(
    da: xr.DataArray,
    t,
    display_mode="rgb",  # "rgb" | "false_color" | "ndvi" | "ndwi" | "band" | "custom"
    max_width=None,
    label=True,
    # label sizing (normally leave)
    font_scale=0.03,
    font_min=14,
    font_max=48,
    bar_pad=None,
    # stretch defaults (rgb/false-color/band/custom)
    p_low=2,
    p_high=98,
    gamma=1.1,
    # index ranges
    ndvi_range=(-1.0, 1.0),
    ndwi_range=(-1.0, 1.0),
    # display_mode="band": name of the band to render in grey levels
    band=None,
    # display_mode="custom": (r, g, b) band names, QGIS-style free mapping
    rgb_bands=None,
):
    mode = str(display_mode).lower().strip()

    if mode == "rgb":
        r_key = _band_key(da, "red")
        g_key = _band_key(da, "green")
        b_key = _band_key(da, "blue")

        R = da.sel(time=t, band=r_key).transpose("y", "x").compute().values
        G = da.sel(time=t, band=g_key).transpose("y", "x").compute().values
        B = da.sel(time=t, band=b_key).transpose("y", "x").compute().values

        rgb = np.dstack(
            [
                _stretch_uint8(R, p_low, p_high, gamma),
                _stretch_uint8(G, p_low, p_high, gamma),
                _stretch_uint8(B, p_low, p_high, gamma),
            ]
        )
        im = Image.fromarray(rgb, mode="RGB")

    elif mode == "false_color":
        # Classic CIR: NIR -> R, Red -> G, Green -> B
        nir_key = _band_key(da, "nir")
        red_key = _band_key(da, "red")
        grn_key = _band_key(da, "green")

        N = da.sel(time=t, band=nir_key).transpose("y", "x").compute().values
        R = da.sel(time=t, band=red_key).transpose("y", "x").compute().values
        G = da.sel(time=t, band=grn_key).transpose("y", "x").compute().values

        rgb = np.dstack(
            [
                _stretch_uint8(N, p_low, p_high, gamma),
                _stretch_uint8(R, p_low, p_high, gamma),
                _stretch_uint8(G, p_low, p_high, gamma),
            ]
        )
        im = Image.fromarray(rgb, mode="RGB")

    elif mode == "ndvi":
        # NDVI = (NIR - RED) / (NIR + RED)
        nir_key = _band_key(da, "nir")
        red_key = _band_key(da, "red")

        N = (
            da.sel(time=t, band=nir_key)
            .transpose("y", "x")
            .compute()
            .values.astype("float32", copy=False)
        )
        R = (
            da.sel(time=t, band=red_key)
            .transpose("y", "x")
            .compute()
            .values.astype("float32", copy=False)
        )

        denom = N + R
        ndvi = np.divide(N - R, denom, out=np.full_like(N, np.nan), where=(denom != 0))
        nodata = ~np.isfinite(ndvi)

        u = _norm_uint8(ndvi, ndvi_range[0], ndvi_range[1])
        rgb = _apply_lut(u, _NDVI_LUT, nodata_mask=nodata)
        im = Image.fromarray(rgb, mode="RGB")

    elif mode == "ndwi":
        # NDWI (McFeeters) = (GREEN - NIR) / (GREEN + NIR)
        grn_key = _band_key(da, "green")
        nir_key = _band_key(da, "nir")

        G = (
            da.sel(time=t, band=grn_key)
            .transpose("y", "x")
            .compute()
            .values.astype("float32", copy=False)
        )
        N = (
            da.sel(time=t, band=nir_key)
            .transpose("y", "x")
            .compute()
            .values.astype("float32", copy=False)
        )

        denom = G + N
        ndwi = np.divide(G - N, denom, out=np.full_like(G, np.nan), where=(denom != 0))
        nodata = ~np.isfinite(ndwi)

        u = _norm_uint8(ndwi, ndwi_range[0], ndwi_range[1])
        rgb = _apply_lut(u, _NDWI_LUT, nodata_mask=nodata)
        im = Image.fromarray(rgb, mode="RGB")

    elif mode == "band":
        # Single band in grey levels with per-frame percentile stretch.
        if band is None or str(band).strip() == "":
            raise ValueError("display_mode='band' requires the 'band' argument.")
        if "band" in da.dims:
            key = _band_key(da, str(band))
            A = da.sel(time=t, band=key).transpose("y", "x").compute().values
        else:
            # Band-less array (single layer time series): render it directly.
            A = da.sel(time=t).transpose("y", "x").compute().values
        u = _stretch_uint8(A, p_low, p_high, gamma)
        im = Image.fromarray(u, mode="L").convert("RGB")

    elif mode == "custom":
        # Free channel mapping: any band to R, G and B (QGIS logic).
        if not rgb_bands or len(rgb_bands) != 3:
            raise ValueError(
                "display_mode='custom' requires rgb_bands=(r_band, g_band, b_band)."
            )
        keys = [_band_key(da, str(b)) for b in rgb_bands]
        chans = [
            da.sel(time=t, band=k).transpose("y", "x").compute().values
            for k in keys
        ]
        rgb = np.dstack([_stretch_uint8(c, p_low, p_high, gamma) for c in chans])
        im = Image.fromarray(rgb, mode="RGB")

    else:
        raise ValueError(
            "display_mode must be one of: 'rgb', 'false_color', 'ndvi', 'ndwi', "
            "'band', 'custom'"
        )

    # optional downscale only
    if max_width is not None and im.width > max_width:
        new_h = int(im.height * (max_width / im.width))
        im = im.resize((max_width, new_h), resample=_resample_lanczos())

    # label
    if label:
        im = _add_top_label(
            im,
            _format_date_ddmmyyyy(t),
            font_scale=font_scale,
            font_min=font_min,
            font_max=font_max,
            bar_pad=bar_pad,
        )

    return im


# ---------------- GIF SAVER ----------------
def _bands_needed_for_mode(display_mode, band=None, rgb_bands=None):
    """Band names required to render a single frame in the given display mode."""
    mode = str(display_mode).lower().strip()
    if mode == "band":
        return [str(band)] if band else []
    if mode == "custom":
        return [str(b) for b in (rgb_bands or [])]
    return {
        "rgb": ["red", "green", "blue"],
        "false_color": ["nir", "red", "green"],
        "ndvi": ["nir", "red"],
        "ndwi": ["green", "nir"],
    }.get(mode, [])


def _materialize_for_gif(da: xr.DataArray, display_mode, band=None, rgb_bands=None):
    """
    Load the bands needed for the animation into memory ONCE.

    When the cube has been edited in the data-cube editor (cloud filter or a
    time/band slice), the working array carries a lazy dask graph whose time
    axis was selected with a non-contiguous (fancy) index. Pulling a single
    time slice out of that graph forces dask to re-read and re-evaluate the
    whole source time-chunk from disk, so building a GIF frame-by-frame
    re-reads the cube hundreds of times and appears to hang. A freshly loaded
    cube has a plain "read one slice" graph and stays fast, which is why the
    problem only shows up after editing.

    Computing the needed bands a single time reads each on-disk chunk once and
    leaves an in-memory array that make_frame can slice instantly.
    """
    is_lazy = getattr(da, "chunks", None) is not None
    if not is_lazy:
        return da

    sub = da
    if "band" in da.dims:
        wanted = _bands_needed_for_mode(display_mode, band=band, rgb_bands=rgb_bands)
        keys = []
        for name in wanted:
            try:
                k = _band_key(da, name)
            except KeyError:
                # Let make_frame raise the precise, user-facing band error.
                return da.load()
            # Custom RGB may map the same band to several channels; sel() with
            # duplicate keys would duplicate the band axis.
            if k not in keys:
                keys.append(k)
        if keys:
            sub = da.sel(band=keys)
    return sub.load()


def save_timeseries_gif(
    da: xr.DataArray,
    out_path="timeseries.gif",
    fps=2,  # yes: higher fps = faster animation
    display_mode="rgb",
    max_width=None,
    label=True,
    band=None,        # display_mode="band": band name to render in grey levels
    rgb_bands=None,   # display_mode="custom": (r, g, b) band names
    p_low=2,
    p_high=98,
):
    da = _materialize_for_gif(da, display_mode, band=band, rgb_bands=rgb_bands)
    times = list(da.coords["time"].values)
    frames = [
        make_frame(
            da,
            t,
            display_mode=display_mode,
            max_width=max_width,
            label=label,
            band=band,
            rgb_bands=rgb_bands,
            p_low=p_low,
            p_high=p_high,
        )
        for t in times
    ]
    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], loop=0, duration=duration_ms
    )
    return out_path


# ==========================================================
# PLOTLY (ZOOM/PAN) RENDERING HELPERS
# ==========================================================
# The zoomable viewer sends real pixel values to the browser as a go.Image
# trace rather than a pre-scaled picture. That is deliberate: plotly.js applies
# crisp-edges (image-rendering: pixelated) styling to image and heatmap traces
# ONLY - a layout image or any plain <img> gets browser-smoothed, which turns
# Sentinel-2 pixels into mush exactly when you zoom in to look at them. The
# cost is payload: plotly 6 serialises uint8 arrays as base64 (~1.33x raw), so
# a 1000x1000 RGB scene is ~4 MB per redraw. _MAX_DISPLAY_PX bounds that.
_MAX_DISPLAY_PX = 2500

# On-screen height per figsize inch for the zoomable view. Above matplotlib's
# 100 dpi on purpose: square-pixel lock means height caps how big the map can
# be drawn, so a larger value fills more of a wide panel.
_ZOOM_PX_PER_INCH = 115


def _zoom_frame_uint8(img: np.ndarray, max_px: int = _MAX_DISPLAY_PX):
    """uint8 image (HxW grey, HxWx3 RGB, HxWx4 RGBA) -> (HxWx3 uint8, shrunk).

    go.Image needs three channels, so grey is replicated and any alpha is
    dropped. Downsampling is nearest-neighbour and reported back to the caller
    rather than done silently, because a downsampled frame no longer shows
    true sensor pixels - which is the whole point of this renderer.
    """
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)

    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3]
    else:
        raise ValueError(
            f"Cannot render an image of shape {arr.shape}; expected 2D grey "
            "or 3D RGB/RGBA."
        )

    h, w = rgb.shape[:2]
    if max(h, w) <= max_px:
        return np.ascontiguousarray(rgb), False

    scale = max_px / float(max(h, w))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    # NEAREST: never invent pixel values that were not in the data.
    pil = Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), Image.NEAREST)
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8)), True


def interactive_time_view(
    stac: xr.DataArray,
    widget_type: str = "slider",  # "slider" or "dropdown" (for time)
    figsize=(8, 8),
    crop=None,
    modes=("rgb", "false_color", "ndvi", "ndwi"),
    return_time_widget: bool = False,
    renderer: str = "static",  # "static" (matplotlib) or "interactive" (plotly)
    max_display_px: int = _MAX_DISPLAY_PX,
    show_grid: bool = True,
    show_coords: bool = False,
    height: int = None,
):
    """
    Interactive cube viewer with three visualization sections:

      1) Presets     - RGB / False color / NDVI / NDWI composites (default).
      2) Single band - any band of the cube shown in grey levels, with a
                       percentile min/max stretch so a few outlier pixels
                       cannot blind the whole scene.
      3) Custom RGB  - free channel mapping: pick any band for R, G and B
                       (QGIS logic), with the same percentile stretch.

    A time slider or date dropdown is shared by all sections. Lazy
    (dask-backed) cubes stay lazy: only the currently displayed scene is
    computed.

    Also accepts a single image without a 'time' dimension (e.g. a temporal
    composite such as a median layer): the time control is hidden and the
    frame is titled with the layer name instead of a date.

    renderer
    --------
    "static"      : matplotlib (default). Draws a fresh PNG per interaction.
                    Lowest overhead and the smallest notebook footprint, but
                    the frame is a flat picture: no zoom, no pan.
    "interactive" : plotly FigureWidget. Same pixels and the same stretch
                    logic, drawn into a live figure with box zoom, scroll
                    zoom and pan over projected-coordinate axes. The zoomed
                    area is deliberately kept across date/band changes, so a
                    feature can be tracked through the time series without
                    re-zooming ("Reset view" restores the full extent).

    Both renderers share one rendering path, so a scene looks the same in
    either; only the presentation layer differs.

    The zoomable view sends true pixel values (a go.Image trace), which
    plotly.js renders with crisp edges - zooming in shows individual sensor
    pixels as hard squares rather than a smoothed blur. The price is payload:
    roughly 1.33x the raw uint8 size per redraw (a 1000x1000 RGB scene is
    ~4 MB), so it is heavier per scene and per saved notebook than "static".

    max_display_px
        Longest edge sent to the browser (default 2500). Above it the frame is
        downsampled nearest-neighbour, and zooming then no longer shows true
        sensor pixels - a note says so when it happens. Raise it to inspect a
        large cube at full resolution, at the cost of a heavier notebook.
    show_grid
        Draw the (deliberately faint grey) coordinate grid. Zoomable view only.
    show_coords
        Zoomable view only, default False: show Easting/Northing axis titles
        and tick labels. Off because the labels take a large share of a screen
        panel, and this renderer is for exploring pixels rather than for
        producing figures. The static renderer always keeps its labelled axes,
        so screenshots for papers and docs are unaffected either way.

    height
        Zoomable view only: figure height in pixels, overriding the figsize
        default. This is the direct size control, and it matters because
        square pixels are locked: the map can never be drawn wider than it is
        tall, so on a panel wider than it is tall the height is what caps the
        image and any remaining width shows as side margin. Filling a wide
        panel completely therefore needs a height close to the panel's width
        (e.g. height=1200), which is a genuine geometric trade-off, not a bug
        - the alternative would be stretching pixels out of square.

    Note that hiding the labels is cosmetic only - the image stays
    georeferenced and zoom/pan still operate in projected coordinates.
    """
    renderer = str(renderer).lower().strip()
    if renderer in ("plotly", "zoom", "interactive"):
        renderer = "interactive"
    elif renderer in ("matplotlib", "static"):
        renderer = "static"
    else:
        raise ValueError(
            f"renderer must be 'static' or 'interactive', got {renderer!r}"
        )

    if renderer == "interactive":
        try:
            import plotly.graph_objects as go  # noqa: F401
        except Exception as e:
            raise ImportError(
                "renderer='interactive' needs plotly. Install it "
                "(`pip install plotly`) or use renderer='static'."
            ) from e
    has_time = "time" in stac.dims
    static_label = None
    if not has_time:
        static_label = str(stac.name) if stac.name is not None else "Composite"
        # Length-1 time axis (no coordinate values, so no fake date) lets the
        # per-frame rendering pipeline below be reused unchanged.
        stac = stac.expand_dims("time")

    stac_c = _apply_crop(stac, crop)
    extent, origin = _get_extent_and_origin(stac_c)

    # Missing frames are judged against the cube's own valid-pixel footprint,
    # not against the bounding box, so a sparse AOI (a river polygon covering a
    # few percent of its bbox) is not mistaken for a cube of empty scenes.
    _footprint = _FootprintTracker()

    if has_time:
        time_values = pd.to_datetime(stac_c.time.values)
        n_time = int(stac_c.time.size)
    else:
        time_values = None
        n_time = 1

    # ------------------------------------------------------------------
    # Band inventory and section availability
    # ------------------------------------------------------------------
    if "band" in stac_c.dims:
        band_names = [str(b) for b in stac_c.coords["band"].values]
    else:
        band_names = []
    band_lower = [b.lower() for b in band_names]

    # A preset is offered only when the cube actually carries the bands it
    # needs; that way a sliced cube (e.g. no 'blue') simply hides RGB instead
    # of erroring on click.
    preset_needs = {
        "rgb": ["red", "green", "blue"],
        "false_color": ["nir", "red", "green"],
        "ndvi": ["ndvi"],
        "ndwi": ["ndwi"],
    }
    available_modes = [
        m
        for m in modes
        if m in preset_needs and all(b in band_lower for b in preset_needs[m])
    ]

    section_options = []
    if available_modes:
        section_options.append(("Presets", "preset"))
    # Single band works for any cube; a band-less array is treated as one layer.
    section_options.append(("Single band", "band"))
    if len(band_names) >= 2:
        section_options.append(("Custom RGB", "custom"))

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    section_w = widgets.ToggleButtons(
        options=section_options,
        value=section_options[0][1],
        style={"button_width": "120px"},
        layout=widgets.Layout(margin="0 0 2px 0"),
    )

    preset_labels = {
        "rgb": "RGB (true color)",
        "false_color": "False color (NIR-R-G)",
        "ndvi": "NDVI",
        "ndwi": "NDWI",
    }
    mode_dd = widgets.Dropdown(
        options=[(preset_labels.get(m, m), m) for m in available_modes]
        or [("RGB (true color)", "rgb")],
        value="rgb" if "rgb" in available_modes else (
            available_modes[0] if available_modes else "rgb"
        ),
        description="Preset:",
        layout=widgets.Layout(width="280px"),
    )

    single_options = band_names or [
        str(stac.name) if stac.name is not None else "layer"
    ]
    band_dd = widgets.Dropdown(
        options=single_options,
        value=single_options[0],
        description="Band:",
        layout=widgets.Layout(width="240px"),
    )

    def _default_band(name, fallback_idx):
        if name in band_lower:
            return band_names[band_lower.index(name)]
        return band_names[min(fallback_idx, len(band_names) - 1)]

    if band_names:
        r0 = _default_band("red", 0)
        g0 = _default_band("green", 1)
        b0 = _default_band("blue", 2)
    else:
        r0 = g0 = b0 = single_options[0]

    chan_layout = widgets.Layout(width="190px")
    chan_style = {"description_width": "24px"}
    r_dd = widgets.Dropdown(options=band_names or single_options, value=r0,
                            description="R:", layout=chan_layout, style=chan_style)
    g_dd = widgets.Dropdown(options=band_names or single_options, value=g0,
                            description="G:", layout=chan_layout, style=chan_style)
    b_dd = widgets.Dropdown(options=band_names or single_options, value=b0,
                            description="B:", layout=chan_layout, style=chan_style)

    stretch_w = widgets.FloatRangeSlider(
        value=(2.0, 98.0),
        min=0.0,
        max=100.0,
        step=0.5,
        description="Stretch (%):",
        continuous_update=False,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
    )
    if widget_type == "slider":
        time_w = widgets.IntSlider(
            min=0,
            max=n_time - 1,
            step=1,
            value=0,
            description="Time",
            continuous_update=False,
            layout=widgets.Layout(width="600px"),
        )
    elif widget_type == "dropdown":
        if has_time:
            options = [(t.strftime("%d-%m-%Y"), i) for i, t in enumerate(time_values)]
        else:
            options = [(static_label, 0)]
        time_w = widgets.Dropdown(
            options=options,
            value=0,
            description="Date:",
            layout=widgets.Layout(width="300px"),
        )
    else:
        raise ValueError("widget_type must be 'slider' or 'dropdown'")

    if not has_time:
        # Single image: nothing to scrub through.
        time_w.layout.display = "none"

    out = widgets.Output()

    # Section-specific control rows; only the active one is visible.
    preset_box = widgets.HBox([mode_dd])
    band_box = widgets.HBox([band_dd])
    custom_box = widgets.HBox(
        [r_dd, g_dd, b_dd], layout=widgets.Layout(gap="8px")
    )
    stretch_box = widgets.VBox([stretch_w], layout=widgets.Layout(gap="0px"))

    def _sync_section_visibility():
        sec = section_w.value
        preset_box.layout.display = "" if sec == "preset" else "none"
        band_box.layout.display = "" if sec == "band" else "none"
        custom_box.layout.display = "" if sec == "custom" else "none"
        # The stretch drives every section, presets included: a preset's
        # contrast is as much a viewing choice as a single band's.
        stretch_box.layout.display = ""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    # Cache preset band selection + scaling so switching dates does not redo it.
    preset_cache = {}

    def _get_mode_state(mode: str):
        mode = str(mode).lower().strip()
        if mode not in preset_cache:
            stac_mode = _select_mode(stac_c, mode)
            preset_cache[mode] = {
                "stac_mode": stac_mode,
                "scaling": _get_scaling_policy(stac_mode, mode),
            }
        return preset_cache[mode]

    # Raw-frame cache: interactions that only restyle already-seen pixels
    # (moving the stretch slider, revisiting a date, remixing custom RGB
    # channels) must not re-read the data. Values are the computed float32
    # pixels of one scene, so revisits cost a dict lookup instead of a dask
    # compute / network read. LRU-bounded by entry count and total bytes.
    _frame_cache = {}
    _FRAME_CACHE_MAX_ENTRIES = 16
    _FRAME_CACHE_MAX_BYTES = 256 * 1024 * 1024

    def _cache_get(key):
        if key in _frame_cache:
            _frame_cache[key] = _frame_cache.pop(key)  # move to end (LRU)
            return _frame_cache[key]
        return None

    def _cache_put(key, val):
        _frame_cache[key] = val
        total = sum(v.nbytes for v in _frame_cache.values())
        while len(_frame_cache) > 1 and (
            total > _FRAME_CACHE_MAX_BYTES
            or len(_frame_cache) > _FRAME_CACHE_MAX_ENTRIES
        ):
            oldest = next(iter(_frame_cache))
            if oldest == key:
                break
            total -= _frame_cache.pop(oldest).nbytes
        return val

    def _get_band_frames(names, idx):
        """Raw 2D float32 frames for the given band names at time idx.

        Cached per band; the bands not in the cache are computed together in
        ONE dask compute, so a custom RGB frame costs one read instead of
        three sequential ones (dask parallelizes the band reads).
        """
        missing = [
            n for n in dict.fromkeys(names) if ("band", n, idx) not in _frame_cache
        ]
        if missing:
            if "band" in stac_c.dims:
                keys = [_band_key(stac_c, n) for n in missing]
                block = (
                    stac_c.sel(band=keys)
                    .isel(time=idx)
                    .transpose("y", "x", "band")
                    .values
                )
                block = np.asarray(block, dtype="float32")
                for j, n in enumerate(missing):
                    _cache_put(("band", n, idx), block[:, :, j])
            else:
                # Band-less array: every pseudo-band name is the layer itself.
                arr = np.asarray(
                    stac_c.isel(time=idx).transpose("y", "x").values,
                    dtype="float32",
                )
                for n in missing:
                    _cache_put(("band", n, idx), arr)
        return [_cache_get(("band", n, idx)) for n in names]

    def _get_preset_raw(mode, idx, st):
        """Raw pixels for a preset frame at time idx, cached."""
        key = ("preset", mode, idx)
        raw = _cache_get(key)
        if raw is None:
            if mode in ("rgb", "false_color"):
                raw = (
                    st["stac_mode"]
                    .isel(time=idx)
                    .transpose("y", "x", "band")
                    .values
                )
            else:
                raw = st["stac_mode"].isel(time=idx).values
            raw = _cache_put(key, np.asarray(raw, dtype="float32"))
        return raw

    def _render_current(idx):
        """Returns (image, title_suffix, cmap). image=None means missing frame."""
        sec = section_w.value
        p_lo, p_hi = (float(v) for v in stretch_w.value)
        if p_hi <= p_lo:
            p_lo, p_hi = 2.0, 98.0

        if sec == "preset":
            mode = mode_dd.value
            st = _get_mode_state(mode)
            raw = _get_preset_raw(mode, idx, st)
            # The slider overrides the preset's own defaults, so a preset
            # stretches exactly like the single-band and custom-RGB sections.
            # RGB presets take the percentiles directly; NDVI/NDWI are drawn
            # through a colormap, so the percentiles are first turned into the
            # vmin/vmax that colormap needs - measured on THIS frame, which is
            # what makes the slider do the same thing in every section.
            scaling = dict(st["scaling"])
            if mode in ("rgb", "false_color"):
                scaling.update(rgb_p_low=p_lo, rgb_p_high=p_hi)
            else:
                bounds = _stretch_bounds(np.asarray(raw), p_lo, p_hi)
                if bounds is not None:
                    scaling["vmin"], scaling["vmax"] = bounds
            img = _render_frame_as_uint8(
                st["stac_mode"], mode, idx, scaling, raw=raw,
                is_missing=_footprint.is_missing,
            )
            suffix = {
                "ndvi": " (NDVI)",
                "ndwi": " (NDWI)",
                "false_color": " (False color)",
            }.get(mode, "")
            return img, suffix, None

        if sec == "band":
            suffix = f" - {band_dd.value}"
            arr = _get_band_frames([str(band_dd.value)], idx)[0]
            if _footprint.is_missing(arr):
                return None, suffix, None
            return _stretch_uint8(arr, p_lo, p_hi), suffix, "gray"

        # Custom RGB
        names = [str(r_dd.value), str(g_dd.value), str(b_dd.value)]
        suffix = f" - R:{names[0]} G:{names[1]} B:{names[2]}"
        chans = _get_band_frames(names, idx)
        if all(_footprint.is_missing(c) for c in chans):
            return None, suffix, None
        rgb = np.dstack([_stretch_uint8(c, p_lo, p_hi) for c in chans])
        return rgb, suffix, None

    def _fmt_axes(ax):
        ax.set_xlabel("Easting (10³ m)")
        ax.set_ylabel("Northing (10⁴ m)")
        ax.tick_params(axis="x", rotation=45)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/1000:.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/10000:.0f}"))
        ax.xaxis.offsetText.set_visible(False)
        ax.yaxis.offsetText.set_visible(False)

    def _frame_title(idx, suffix):
        if has_time:
            return time_values[idx].strftime("%d-%m-%Y") + suffix
        return str(static_label) + suffix

    def _plot_static():
        idx = int(time_w.value)

        with out:
            clear_output(wait=True)

            try:
                img, suffix, cmap = _render_current(idx)
            except Exception as e:
                print(f"View not available: {e}")
                return

            title = _frame_title(idx, suffix)

            fig, ax = plt.subplots(figsize=figsize)

            if img is None:
                ax.text(0.5, 0.5, "Missing Data", fontsize=16,
                        ha="center", va="center")
                ax.set_axis_off()
                ax.set_title(title, fontsize=14)
                plt.show()
                plt.close(fig)
                return

            if cmap is not None:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=255,
                          interpolation="nearest", extent=extent, origin=origin)
            else:
                ax.imshow(img, interpolation="nearest",
                          extent=extent, origin=origin)
            ax.set_title(title, fontsize=14)
            _fmt_axes(ax)

            plt.tight_layout()
            plt.show()
            plt.close(fig)

    # ------------------------------------------------------------------
    # Zoomable renderer (plotly)
    # ------------------------------------------------------------------
    # Built lazily so the static path never imports plotly.
    fig_w = None
    reset_btn = None

    # Pixel geometry straight from the cube's own coordinates. dx/dy carry the
    # step INCLUDING its sign, so a descending-y cube (dy < 0) is placed
    # north-up without flipping any array.
    _xs = np.asarray(stac_c["x"].values, dtype="float64")
    _ys = np.asarray(stac_c["y"].values, dtype="float64")
    _nx_full, _ny_full = int(_xs.size), int(_ys.size)
    _dx_full = float(_xs[1] - _xs[0]) if _nx_full > 1 else 1.0
    _dy_full = float(_ys[1] - _ys[0]) if _ny_full > 1 else -1.0

    # x0/y0 are the OUTER EDGE of the first pixel, not its centre: the trace
    # spans dx*nx from there, so this is what makes the drawn footprint equal
    # the raster's true footprint. Using the centre (what px.imshow does) puts
    # the image half a pixel off, and in opposite directions for ascending vs
    # descending y, so the same ground would land differently for two cubes
    # that only differ in row order.
    _x0_edge = float(_xs[0] - _dx_full / 2.0)
    _y0_edge = float(_ys[0] - _dy_full / 2.0)

    # Drawn span of the whole raster; used for the initial view and for Reset.
    _x_edges = sorted([_x0_edge, _x0_edge + _dx_full * _nx_full])
    _y_edges = sorted([_y0_edge, _y0_edge + _dy_full * _ny_full])

    def _build_plotly_figure():
        import plotly.graph_objects as go

        fig = go.FigureWidget()
        # Placeholder image; _plot_interactive fills in the real pixels.
        fig.add_trace(
            go.Image(
                z=np.zeros((1, 1, 3), dtype=np.uint8),
                x0=_x0_edge,
                dx=_dx_full,
                y0=_y0_edge,
                dy=_dy_full,
                hoverinfo="skip",
            )
        )

        # Grid: present for orientation but deliberately quiet, so it reads as
        # a background reference instead of competing with the imagery.
        #
        # Axis titles and tick labels are OFF by default here. This view is for
        # exploring pixels on screen, and the Easting/Northing furniture eats a
        # large share of the panel. The static renderer keeps its full labelled
        # axes untouched - that is the one used for figures in papers and docs.
        grid_kw = {
            "showgrid": bool(show_grid),
            "gridcolor": "rgba(140,140,140,0.22)",
            "gridwidth": 1,
            "zeroline": False,
            "showline": False,
            "showticklabels": bool(show_coords),
            "ticks": "outside" if show_coords else "",
            "tickcolor": "rgba(140,140,140,0.45)",
            "tickfont": {"color": "#6b7280", "size": 10},
        }
        # constraintoward: the square-pixel lock has to give up either range or
        # domain somewhere, and plotly parks the slack at the axis START by
        # default - which is what put all the empty space on the right. Centre
        # it so whatever is left over is split evenly and reads as deliberate.
        fig.update_xaxes(
            range=list(_x_edges),
            title_text="Easting (m)" if show_coords else None,
            constraintoward="center",
            **grid_kw,
        )
        # scaleanchor keeps one map unit square, so a pixel stays a square and
        # shapes are not distorted while zooming. No 'constrain' is set: the
        # default range-constraint is what plotly itself uses with scaleanchor
        # and it survives fast scroll-zoom better than constrain="domain".
        fig.update_yaxes(
            range=list(_y_edges),
            title_text="Northing (m)" if show_coords else None,
            scaleanchor="x",
            scaleratio=1,
            constraintoward="middle",
            **grid_kw,
        )
        # Reclaim the space the labels were using; without them the panel only
        # needs room for the title.
        margin = (
            {"l": 70, "r": 20, "t": 50, "b": 60}
            if show_coords
            else {"l": 8, "r": 8, "t": 40, "b": 8}
        )
        # Width is left to the container (autosize) so the map uses the full
        # width of the panel instead of a fixed box.
        #
        # Height is the real size control here. With square pixels locked, the
        # drawn map can only be as wide as it is tall, so on a panel wider than
        # it is tall HEIGHT is what caps the image and leaves side space. Hence
        # 115 px/inch rather than matplotlib's 100: it makes the map noticeably
        # bigger and eats into that gap. Pass a taller figsize to push further.
        fig.update_layout(
            autosize=True,
            width=None,
            height=int(height) if height else int(figsize[1] * _ZOOM_PX_PER_INCH),
            margin=margin,
            dragmode="pan",
            plot_bgcolor="white",
            title={"text": "", "x": 0.5, "xanchor": "center"},
        )
        # Scroll-to-zoom is off by default in plotly; on a map view it is the
        # interaction users reach for first. "responsive" lets the figure
        # re-fit when the browser window or panel is resized.
        fig._config = {
            "scrollZoom": True,
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        }
        return fig

    def _plot_interactive():
        idx = int(time_w.value)

        try:
            img, suffix, cmap = _render_current(idx)
        except Exception as e:
            with out:
                clear_output(wait=True)
                print(f"View not available: {e}")
            return

        title = _frame_title(idx, suffix)

        with out:
            clear_output(wait=True)

        if img is None:
            with fig_w.batch_update():
                fig_w.data[0].visible = False
                fig_w.layout.title.text = title + "  -  Missing Data"
            return

        z, shrunk = _zoom_frame_uint8(img, max_px=max_display_px)
        if shrunk:
            with out:
                print(
                    f"Note: this scene is larger than {max_display_px} px, so "
                    "the display is downsampled and zooming will NOT show true "
                    "sensor pixels. Raise max_display_px (heavier notebook) or "
                    "crop the cube to inspect them. Data itself is untouched."
                )

        # A downsampled frame covers the same ground with fewer, bigger pixels,
        # so the step has to grow to keep the raster georeferenced.
        h, w = z.shape[:2]
        dx = _dx_full * _nx_full / float(w)
        dy = _dy_full * _ny_full / float(h)

        # Only the pixels and the title change: axis ranges are left untouched
        # on purpose so a zoomed-in area stays put when the user steps through
        # dates or swaps bands.
        with fig_w.batch_update():
            tr = fig_w.data[0]
            tr.z = z
            tr.x0, tr.dx = _x0_edge, dx
            tr.y0, tr.dy = _y0_edge, dy
            tr.visible = True
            fig_w.layout.title.text = title

    def _reset_view(_btn):
        with fig_w.batch_update():
            fig_w.layout.xaxis.range = list(_x_edges)
            fig_w.layout.yaxis.range = list(_y_edges)

    plot_current = _plot_static if renderer == "static" else _plot_interactive

    def _on_section_change(_change):
        _sync_section_visibility()
        plot_current()

    def _on_control_change(_change):
        plot_current()

    section_w.observe(_on_section_change, names="value")
    for w in (mode_dd, band_dd, r_dd, g_dd, b_dd, stretch_w, time_w):
        w.observe(_on_control_change, names="value")

    _sync_section_visibility()

    controls = widgets.VBox(
        [section_w, preset_box, band_box, custom_box, stretch_box, time_w],
        layout=widgets.Layout(gap="6px"),
    )

    if renderer == "static":
        display(widgets.VBox([controls, out]))
    else:
        fig_w = _build_plotly_figure()
        reset_btn = widgets.Button(
            description="Reset view",
            icon="refresh",
            tooltip="Zoom back out to the full cube extent",
            layout=widgets.Layout(width="130px"),
        )
        reset_btn.on_click(_reset_view)
        zoom_hint = widgets.HTML(
            "<div style='font-size:11px; color:#6b7280; margin-left:4px;'>"
            "Scroll to zoom, drag to pan, double-click to reset. The zoom "
            "is kept when you change date or bands, so you can follow one "
            "spot through the time series."
            "</div>"
        )
        # A FigureWidget has no settable ipywidgets width: `.layout` is
        # plotly's figure layout, and `_widget_layout` is that same figure
        # layout synced to the frontend - neither is CSS. So the widget's own
        # DOM node keeps its natural width and an autosized plot has nothing
        # to expand into, no matter how wide the wrapper is. `add_class` does
        # sync (via _dom_classes), so a stylesheet is the way in.
        fig_w.add_class("s2c-zoom-fig")
        fig_css = widgets.HTML(
            "<style>"
            ".s2c-zoom-fig, .s2c-zoom-fig > div, .s2c-zoom-fig .js-plotly-plot,"
            ".s2c-zoom-fig .plot-container, .s2c-zoom-fig .svg-container"
            "{width:100% !important; max-width:100% !important;}"
            "</style>"
        )
        fig_box = widgets.Box(
            [fig_w], layout=widgets.Layout(width="100%", overflow="hidden")
        )
        display(
            widgets.VBox(
                [
                    fig_css,
                    controls,
                    widgets.HBox([reset_btn, zoom_hint]),
                    fig_box,
                    out,
                ],
                layout=widgets.Layout(width="100%"),
            )
        )

    plot_current()

    if return_time_widget:
        # opt-in accessor for embedding GUIs (e.g. "pick a reference scene"):
        # time_w.value is the integer index into the cube's time axis.
        return time_w


# ==========================================================
# CLOUD-MASK COMPARISON VIEWER (ARD manual workflow)
# ==========================================================
def _cloud_band_label(name: str):
    """Friendly (label, value) pair for a Cloud_Stack band name."""
    s = str(name)
    if s.lower() == "cloud_prob":
        return ("Cloud probability (%)", s)
    m = re.search(r"(\d+)\s*$", s)
    if s.lower().startswith("cloud_mask") and m:
        return (f"Binary mask ≥ {m.group(1)}%", s)
    return (s, s)


def _is_probability_band(name: str) -> bool:
    return "prob" in str(name).lower()


def interactive_cloud_overlay_view(
    spectral: xr.DataArray,
    cloud: xr.DataArray,
    widget_type: str = "dropdown",  # kept for API symmetry; date is always a dropdown
    figsize=(13, 6),
    default_opacity: float = 0.5,
    mask_color=(1.0, 0.15, 0.15),
    prob_cmap: str = "turbo",
):
    """
    Side-by-side cloud-mask comparison viewer for the ARD manual workflow.

    Left panel : clean RGB reference (unchanged true-color scene).
    Right panel: the same RGB with the selected cloud band overlaid, so the user
                 can judge which real pixels a probability/threshold removes and
                 pick the threshold that fits their scene.

    Controls:
      - Date dropdown          (shared time axis of both cubes)
      - Cloud band dropdown     (cloud_prob or any binary cloud_mask_*)
      - Overlay opacity slider  (keeps the underlying RGB texture visible)

    Overlay rules:
      - cloud_prob   : translucent heat colormap; per-pixel alpha grows with
                       probability so clear sky stays RGB and cloudy areas glow.
                       A 0-100% colorbar is drawn.
      - cloud_mask_* : binary; masked (==1) pixels get a translucent solid wash,
                       clear pixels stay fully RGB. Title reports % masked.

    Both cubes are expected to be pre-aligned on the same time and y/x grid
    (the GUI handler validates and slices them to the common dates before call).
    """
    import matplotlib as mpl
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    if "band" not in cloud.dims:
        raise ValueError("Cloud cube has no 'band' dimension.")

    # RGB rendering shares the exact pipeline used by interactive_time_view.
    rgb_mode = _select_mode(spectral, "rgb")
    extent, origin = _get_extent_and_origin(rgb_mode)
    scaling = _get_scaling_policy(rgb_mode, "rgb")
    time_values = pd.to_datetime(rgb_mode.time.values)

    # Same footprint-relative missing test as interactive_time_view.
    _footprint = _FootprintTracker()

    # Cloud band options: probability first, then masks by ascending threshold.
    band_names = [str(b) for b in cloud["band"].values]

    def _sort_key(n):
        if _is_probability_band(n):
            return (0, 0)
        m = re.search(r"(\d+)\s*$", n)
        return (1, int(m.group(1)) if m else 0)

    band_names = sorted(band_names, key=_sort_key)
    band_options = [_cloud_band_label(n) for n in band_names]

    date_w = widgets.Dropdown(
        options=[(t.strftime("%d-%m-%Y"), i) for i, t in enumerate(time_values)],
        value=0,
        description="Date:",
        layout=widgets.Layout(width="260px"),
    )
    band_w = widgets.Dropdown(
        options=band_options,
        value=band_names[0],
        description="Cloud band:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="320px", margin="0 0 0 48px"),
    )
    opacity_w = widgets.FloatSlider(
        value=float(default_opacity),
        min=0.1,
        max=0.9,
        step=0.05,
        description="Overlay opacity:",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="360px"),
    )
    # One percentile stretch shared by BOTH panels (same control as the other
    # viewers): on cloudy scenes the bright clouds own the top percentiles and
    # push the land into the dark bottom of the range. Lowering the high
    # percentile saturates the clouds and gives the range back to the ground.
    stretch_w = widgets.FloatRangeSlider(
        value=(2.0, 98.0),
        min=0.0,
        max=100.0,
        step=0.5,
        description="Stretch (%):",
        continuous_update=False,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
    )
    # Map layout control. "Vertical" stacks the two panels so each map spans the
    # whole output cell (biggest option on a laptop screen); "Horizontal" keeps
    # them side-by-side.
    view_w = widgets.ToggleButtons(
        options=[("Horizontal", "normal"), ("Vertical", "full")],
        value="normal",
        description="Map Layout:",
        style={"description_width": "initial", "button_width": "110px"},
    )

    out = widgets.Output()

    # Raw-frame cache: restyling interactions (stretch / opacity / layout) must
    # not re-read the cube. The spectral cube may be lazy (dask-backed), so
    # without this every slider move would re-read three bands from disk.
    # Small FIFO bound keeps memory flat on long time series.
    _raw_cache = {}
    _RAW_CACHE_MAX = 8

    def _get_rgb_raw(idx):
        raw = _raw_cache.get(idx)
        if raw is None:
            raw = np.asarray(
                rgb_mode.isel(time=idx).transpose("y", "x", "band").values,
                dtype="float32",
            )
            while len(_raw_cache) >= _RAW_CACHE_MAX:
                _raw_cache.pop(next(iter(_raw_cache)))
            _raw_cache[idx] = raw
        return raw

    def _fmt_axis(ax):
        ax.set_xlabel("Easting (10³ m)")
        ax.set_ylabel("Northing (10⁴ m)")
        ax.tick_params(axis="x", rotation=45)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/1000:.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/10000:.0f}"))
        ax.xaxis.offsetText.set_visible(False)
        ax.yaxis.offsetText.set_visible(False)

    def plot_current():
        idx = int(date_w.value)
        sel = band_w.value
        alpha = float(opacity_w.value)
        p_lo, p_hi = (float(v) for v in stretch_w.value)
        if p_hi <= p_lo:
            p_lo, p_hi = 2.0, 98.0
        view = view_w.value
        t = rgb_mode.time.values[idx]
        date_str = time_values[idx].strftime("%d-%m-%Y")

        with out:
            clear_output(wait=True)

            rgb_img = _render_frame_as_uint8(
                rgb_mode,
                "rgb",
                idx,
                dict(scaling, rgb_p_low=p_lo, rgb_p_high=p_hi),
                raw=_get_rgb_raw(idx),
                is_missing=_footprint.is_missing,
            )

            # "full" stacks the panels vertically so each map fills the output
            # width; "normal" keeps them side-by-side at the base size.
            if view == "full":
                fig, (ax_l, ax_r) = plt.subplots(2, 1, figsize=(11, 15))
            else:
                fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

            if rgb_img is None:
                for ax in (ax_l, ax_r):
                    ax.text(0.5, 0.5, "Missing Data", fontsize=16,
                            ha="center", va="center")
                    ax.set_axis_off()
                fig.suptitle(date_str, fontsize=14)
                plt.tight_layout()
                plt.show()
                plt.close(fig)
                return

            # Left: clean RGB reference
            ax_l.imshow(rgb_img, interpolation="nearest", extent=extent, origin=origin)
            ax_l.set_title(f"{date_str}  —  RGB (reference)", fontsize=13)
            _fmt_axis(ax_l)

            # Right: RGB + cloud overlay
            ax_r.imshow(rgb_img, interpolation="nearest", extent=extent, origin=origin)

            carr = (
                cloud.sel(time=t, band=sel)
                .transpose("y", "x")
                .values.astype("float32", copy=False)
            )
            finite = np.isfinite(carr)
            label = dict(band_options).get(sel, sel)

            if _is_probability_band(sel):
                norm01 = np.clip(carr / 100.0, 0.0, 1.0)
                cmap = mpl.colormaps[prob_cmap]
                rgba = cmap(norm01)  # (H, W, 4) float
                a = alpha * norm01
                a[~finite] = 0.0
                rgba[..., 3] = a
                ax_r.imshow(rgba, interpolation="nearest", extent=extent, origin=origin)

                sm = ScalarMappable(norm=Normalize(vmin=0, vmax=100), cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax_r, fraction=0.046, pad=0.04)
                cbar.set_label("Cloud probability (%)")
                ax_r.set_title(f"{date_str}  —  {label} overlay", fontsize=13)
            else:
                masked = finite & (carr >= 0.5)
                overlay = np.zeros(carr.shape + (4,), dtype="float32")
                overlay[..., 0] = mask_color[0]
                overlay[..., 1] = mask_color[1]
                overlay[..., 2] = mask_color[2]
                overlay[..., 3] = np.where(masked, alpha, 0.0)
                ax_r.imshow(overlay, interpolation="nearest", extent=extent, origin=origin)

                valid = int(finite.sum())
                pct = (100.0 * masked.sum() / valid) if valid else 0.0
                ax_r.set_title(
                    f"{date_str}  —  {label}  ({pct:.1f}% masked)", fontsize=13
                )

            _fmt_axis(ax_r)
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    def _on_change(_change):
        plot_current()

    date_w.observe(_on_change, names="value")
    band_w.observe(_on_change, names="value")
    opacity_w.observe(_on_change, names="value")
    stretch_w.observe(_on_change, names="value")
    view_w.observe(_on_change, names="value")

    controls = widgets.HBox(
        [date_w, band_w],
        layout=widgets.Layout(gap="32px"),
    )
    controls2 = widgets.HBox(
        [
            opacity_w,
            widgets.VBox([stretch_w], layout=widgets.Layout(gap="0px")),
        ],
        layout=widgets.Layout(gap="32px", align_items="flex-start"),
    )
    controls3 = widgets.HBox(
        [view_w],
        layout=widgets.Layout(align_items="center"),
    )
    display(widgets.VBox([controls, controls2, controls3, out]))
    plot_current()
