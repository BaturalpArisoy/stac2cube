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
    arr: np.ndarray, nan_fraction_thresh=0.9, variance_thresh=1e-12
) -> bool:
    """
    Robust missing test:
    - mostly NaN
    - or near-constant frame (all zeros / no signal)
    """
    if arr.size == 0:
        return True

    nan_frac = np.mean(~np.isfinite(arr))
    if nan_frac >= nan_fraction_thresh:
        return True

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True

    if np.nanvar(finite) <= variance_thresh:
        return True

    return False


# ==========================================================
# SCALING POLICY
# ==========================================================
def _get_scaling_policy(data_stac: xr.DataArray, display_mode: str):
    dm = str(display_mode).lower().strip()
    lazy = _is_lazy_xarray(data_stac)

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

    # NDVI/NDWI
    if lazy:
        return {"vmin": -1.0, "vmax": 1.0}
    else:
        vals = data_stac.values
        vmin = float(np.nanpercentile(vals, 2))
        vmax = float(np.nanpercentile(vals, 98))
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
        return {"vmin": vmin, "vmax": vmax}


# ==========================================================
# RGB NORMALIZATION (per-frame, robust)
# ==========================================================
def _normalize_rgb_frame(rgb_yxb: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    """
    Per-frame robust RGB normalization:
    - percentile clip per band
    - valid pixels only if all 3 bands finite
    - invalid pixels -> neutral gray (prevents random red/blue speckles)
    Returns float RGB in [0,1]
    """
    rgb = rgb_yxb.astype(np.float32, copy=False)

    valid = np.all(np.isfinite(rgb), axis=2)
    out = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        band = rgb[:, :, i]
        band = np.where(np.isfinite(band), band, np.nan)

        lo = float(np.nanpercentile(band, p_low))
        hi = float(np.nanpercentile(band, p_high))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[:, :, i] = 0.5
            continue

        band = np.clip(band, lo, hi)
        out[:, :, i] = (band - lo) / (hi - lo)

    out[~valid] = 0.5
    return np.clip(out, 0, 1)


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
    stac_mode: xr.DataArray, display_mode: str, idx: int, scaling
):
    """
    Returns a uint8 RGB image.
    Lazy-safe: computes ONLY the selected time slice.
    """
    dm = str(display_mode).lower().strip()

    if dm in ["rgb", "false_color"]:
        frame = stac_mode.isel(time=idx).transpose("y", "x", "band")
        rgb = frame.values  # lazy -> computes only this slice

        if _missing_frame(rgb):
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
    frame = stac_mode.isel(time=idx)
    data = frame.values  # lazy -> computes only this slice

    if _missing_frame(data):
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
    display_mode="rgb",  # "rgb" | "false_color" | "ndvi" | "ndwi"
    max_width=None,
    label=True,
    # label sizing (normally leave)
    font_scale=0.03,
    font_min=14,
    font_max=48,
    bar_pad=None,
    # rgb/false-color stretch defaults
    p_low=2,
    p_high=98,
    gamma=1.1,
    # index ranges
    ndvi_range=(-1.0, 1.0),
    ndwi_range=(-1.0, 1.0),
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

    else:
        raise ValueError(
            "display_mode must be one of: 'rgb', 'false_color', 'ndvi', 'ndwi'"
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
def _bands_needed_for_mode(display_mode):
    """Band names required to render a single frame in the given display mode."""
    mode = str(display_mode).lower().strip()
    return {
        "rgb": ["red", "green", "blue"],
        "false_color": ["nir", "red", "green"],
        "ndvi": ["nir", "red"],
        "ndwi": ["green", "nir"],
    }.get(mode, [])


def _materialize_for_gif(da: xr.DataArray, display_mode):
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
        wanted = _bands_needed_for_mode(display_mode)
        keys = []
        for name in wanted:
            try:
                keys.append(_band_key(da, name))
            except KeyError:
                # Let make_frame raise the precise, user-facing band error.
                return da.load()
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
):
    da = _materialize_for_gif(da, display_mode)
    times = list(da.coords["time"].values)
    frames = [
        make_frame(da, t, display_mode=display_mode, max_width=max_width, label=label)
        for t in times
    ]
    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], loop=0, duration=duration_ms
    )
    return out_path


def interactive_time_view(
    stac: xr.DataArray,
    widget_type: str = "slider",  # "slider" or "dropdown" (for time)
    figsize=(8, 8),
    crop=None,
    modes=("rgb", "false_color", "ndvi", "ndwi"),
):
    """
    Interactive viewer with TWO controls:
      1) display_mode dropdown (default RGB)
      2) time slider or date dropdown

    Uses your existing pipeline: _select_mode, _apply_crop, _get_extent_and_origin,
    _get_scaling_policy, _render_frame_as_uint8.
    """
    # UI
    mode_options = []
    for m in modes:
        label = {
            "rgb": "RGB",
            "false_color": "False color",
            "ndvi": "NDVI",
            "ndwi": "NDWI",
        }.get(m, m)
        mode_options.append((label, m))

    mode_dd = widgets.Dropdown(
        options=mode_options,
        value="rgb" if "rgb" in modes else modes[0],
        description="Display Mode:",
        layout=widgets.Layout(width="260px"),
    )

    out = widgets.Output()

    # Cache per mode to avoid recomputing selection/scaling each time
    cache = {}

    def _get_mode_state(mode: str):
        mode = str(mode).lower().strip()
        if mode in cache:
            return cache[mode]

        stac_mode = _select_mode(stac, mode)
        stac_mode = _apply_crop(stac_mode, crop)

        extent, origin = _get_extent_and_origin(stac_mode)
        time_values = pd.to_datetime(stac_mode.time.values)
        n = stac_mode.time.size
        scaling = _get_scaling_policy(stac_mode, mode)

        cache[mode] = {
            "stac_mode": stac_mode,
            "extent": extent,
            "origin": origin,
            "time_values": time_values,
            "n": n,
            "scaling": scaling,
        }
        return cache[mode]

    # Time widget (created once, updated if needed)
    # We build it after we know n from default mode
    try:
        s0 = _get_mode_state(mode_dd.value)
    except Exception as e:
        with out:
            clear_output(wait=True)
            print(f"Error initializing mode '{mode_dd.value}': {e}")
        display(widgets.VBox([mode_dd, out]))
        return

    n0 = s0["n"]
    time_values0 = s0["time_values"]

    if widget_type == "slider":
        time_w = widgets.IntSlider(
            min=0,
            max=n0 - 1,
            step=1,
            value=0,
            description="Time",
            layout=widgets.Layout(width="800px"),
        )
    elif widget_type == "dropdown":
        options = [(t.strftime("%d-%m-%Y"), i) for i, t in enumerate(time_values0)]
        time_w = widgets.Dropdown(
            options=options,
            value=0,
            description="Date:",
            layout=widgets.Layout(width="300px"),
        )
    else:
        raise ValueError("widget_type must be 'slider' or 'dropdown'")

    def _set_time_widget_options(state):
        """Update time widget to match current mode's time axis (usually identical)."""
        n = state["n"]
        tv = state["time_values"]

        if widget_type == "slider":
            time_w.max = n - 1
            if time_w.value > n - 1:
                time_w.value = n - 1
        else:
            opts = [(t.strftime("%d-%m-%Y"), i) for i, t in enumerate(tv)]
            time_w.options = opts
            if time_w.value > n - 1:
                time_w.value = n - 1

    def plot_current():
        mode = mode_dd.value
        idx = int(time_w.value)

        with out:
            clear_output(wait=True)

            try:
                state = _get_mode_state(mode)
            except Exception as e:
                print(f"Mode '{mode}' not available: {e}")
                return

            # keep time widget in sync (in case mode has different n)
            if idx > state["n"] - 1:
                idx = state["n"] - 1
                time_w.value = idx

            stac_mode = state["stac_mode"]
            extent = state["extent"]
            origin = state["origin"]
            time_values = state["time_values"]
            scaling = state["scaling"]

            img = _render_frame_as_uint8(stac_mode, mode, idx, scaling)
            fig, ax = plt.subplots(figsize=figsize)

            # Title
            title = time_values[idx].strftime("%d-%m-%Y")
            if mode == "ndvi":
                title += " (NDVI)"
            elif mode == "ndwi":
                title += " (NDWI)"
            elif mode == "false_color":
                title += " (False color)"

            if img is None:
                ax.text(0.5, 0.5, "Missing Data", fontsize=16, ha="center", va="center")
                ax.set_axis_off()
                plt.show()
                plt.close(fig)
                return

            ax.imshow(img, interpolation="nearest", extent=extent, origin=origin)
            ax.set_title(title, fontsize=14)

            ax.set_xlabel("Easting (10³ m)")
            ax.set_ylabel("Northing (10⁴ m)")
            ax.tick_params(axis="x", rotation=45)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/1000:.0f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/10000:.0f}"))
            ax.xaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

            plt.tight_layout()
            plt.show()
            plt.close(fig)

    def _on_mode_change(change):
        # update time widget if needed + redraw
        try:
            state = _get_mode_state(change["new"])
            _set_time_widget_options(state)
        except Exception:
            pass
        plot_current()

    def _on_time_change(change):
        plot_current()

    mode_dd.observe(_on_mode_change, names="value")
    time_w.observe(_on_time_change, names="value")

    display(widgets.VBox([mode_dd, time_w, out]))
    plot_current()


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
        view = view_w.value
        t = rgb_mode.time.values[idx]
        date_str = time_values[idx].strftime("%d-%m-%Y")

        with out:
            clear_output(wait=True)

            rgb_img = _render_frame_as_uint8(rgb_mode, "rgb", idx, scaling)

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
    view_w.observe(_on_change, names="value")

    controls = widgets.HBox(
        [date_w, band_w],
        layout=widgets.Layout(gap="32px"),
    )
    controls2 = widgets.HBox(
        [opacity_w, view_w],
        layout=widgets.Layout(gap="32px", align_items="center"),
    )
    display(widgets.VBox([controls, controls2, out]))
    plot_current()
