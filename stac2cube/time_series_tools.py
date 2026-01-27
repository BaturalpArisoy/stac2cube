import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display, clear_output
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image

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
    if display_mode == "rgb":
        return stac.sel(band=["red", "green", "blue"])
    elif display_mode in ["ndvi", "ndwi"]:
        return stac.sel(band=display_mode)
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
def _missing_frame(arr: np.ndarray, nan_fraction_thresh=0.9, variance_thresh=1e-12) -> bool:
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
    """
    RGB:
      - per-frame percentile normalization
      - auto exposure gain (controlled, not too bright)
      - avoids color speckles by handling invalid pixels consistently

    NDVI/NDWI:
      - lazy: fixed vmin/vmax (no global compute)
      - eager: global percentiles (computed once)
    """
    lazy = _is_lazy_xarray(data_stac)

    if display_mode == "rgb":
        return {
            "rgb_p_low": 2,
            "rgb_p_high": 98,
            "rgb_auto_gain": True,
            "rgb_target_luma": 0.38,  # lower = slightly darker
            "rgb_gain_min": 0.9,
            "rgb_gain_max": 1.25,
            "rgb_gamma": 1.0,         # keep off (1.0)
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


def _rgb_to_uint8(rgb_01: np.ndarray, gamma: float = 1.0, gain: float = 1.0) -> np.ndarray:
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
def _nd_to_rgb_uint8(data: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> np.ndarray:
    """
    Convert a 2D NDVI/NDWI array into an RGB uint8 image using a colormap.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(data))  # float RGBA in [0,1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


# ==========================================================
# FRAME RENDERING (single time index)
# ==========================================================
def _render_frame_as_uint8(stac_mode: xr.DataArray, display_mode: str, idx: int, scaling):
    """
    Returns a uint8 RGB image.
    Lazy-safe: computes ONLY the selected time slice.
    """

    if display_mode == "rgb":
        frame = stac_mode.isel(time=idx).transpose("y", "x", "band")
        rgb = frame.values  # lazy -> computes only this slice

        if _missing_frame(rgb):
            return None

        rgb01 = _normalize_rgb_frame(
            rgb,
            p_low=scaling.get("rgb_p_low", 2),
            p_high=scaling.get("rgb_p_high", 98),
        )

        # controlled auto exposure
        gain = 1.0
        if scaling.get("rgb_auto_gain", True):
            luma = float(np.mean(rgb01))
            target = float(scaling.get("rgb_target_luma", 0.38))

            if np.isfinite(luma) and luma > 1e-6:
                gain = target / luma
                gain = float(np.clip(
                    gain,
                    scaling.get("rgb_gain_min", 0.9),
                    scaling.get("rgb_gain_max", 1.25),
                ))

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

    cmap = "RdYlGn" if display_mode == "ndvi" else "Blues"
    return _nd_to_rgb_uint8(data, cmap_name=cmap, vmin=scaling["vmin"], vmax=scaling["vmax"])


# ==========================================================
# PUBLIC API 1: INTERACTIVE TIME VIEW (lazy-safe)
# ==========================================================
def interactive_time_view(
    stac: xr.DataArray,
    display_mode: str,
    widget_type: str = "slider",
    figsize=(8, 8),
    crop=None,  # optional projected crop
):
    """
    Lazy-safe interactive viewer.
    - computes ONLY selected time slice
    - shows projected UTM axes (Easting/Northing) with nice short ticks
    """
    stac_mode = _select_mode(stac, display_mode)
    stac_mode = _apply_crop(stac_mode, crop)

    extent, origin = _get_extent_and_origin(stac_mode)
    time_values = pd.to_datetime(stac_mode.time.values)
    n = stac_mode.time.size

    scaling = _get_scaling_policy(stac_mode, display_mode)
    out = widgets.Output()

    def plot_idx(idx: int):
        with out:
            clear_output(wait=True)

            img = _render_frame_as_uint8(stac_mode, display_mode, idx, scaling)
            fig, ax = plt.subplots(figsize=figsize)

            title = time_values[idx].strftime("%d-%m-%Y")
            if display_mode == "ndvi":
                title += " (NDVI)"
            elif display_mode == "ndwi":
                title += " (NDWI)"

            if img is None:
                ax.text(0.5, 0.5, "Missing Data", fontsize=16, ha="center", va="center")
                ax.set_axis_off()
                plt.show()
                plt.close(fig)
                return

            ax.imshow(img, interpolation="nearest", extent=extent, origin=origin)
            ax.set_title(title, fontsize=14)

            # short coords like you wanted
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

    if widget_type == "slider":
        w = widgets.IntSlider(
            min=0,
            max=n - 1,
            step=1,
            value=0,
            description="Time",
            layout=widgets.Layout(width="800px"),
        )
    elif widget_type == "dropdown":
        options = [(t.strftime("%d-%m-%Y"), i) for i, t in enumerate(time_values)]
        w = widgets.Dropdown(
            options=options,
            value=0,
            description="Date:",
            layout=widgets.Layout(width="300px"),
        )
    else:
        raise ValueError("widget_type must be 'slider' or 'dropdown'")

    display(w, out)
    widgets.interact(plot_idx, idx=w)


# ==========================================================
# PUBLIC API 2: GIF EXPORT (no ffmpeg, lazy-safe)
# ==========================================================
def generate_animation(
    stac: xr.DataArray,
    output_path: str,
    display_mode: str,
    frame_interval_ms: int = 700,
    crop=None,                 # optional: (xmin, xmax, ymin, ymax)
    max_size: int | None = 900,# resize longest side to this (smaller file). None = no resize
    colors: int = 128,         # palette size (64/128/256). Lower = smaller file
    dither: bool = False,      # dithering makes file bigger + noisy; False is cleaner for EO
):
    """
    Streaming GIF export (no PNGs, no ffmpeg).
    - lazy-safe: computes one frame at a time
    - small file size controls: max_size + colors

    output_path: user-defined path, will append .gif if missing
    """
    if frame_interval_ms <= 0:
        raise ValueError("frame_interval_ms must be > 0")

    # Ensure output folder exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not output_path.lower().endswith(".gif"):
        output_path += ".gif"

    stac_mode = _select_mode(stac, display_mode)
    stac_mode = _apply_crop(stac_mode, crop)

    n = stac_mode.time.size
    scaling = _get_scaling_policy(stac_mode, display_mode)

    frames: list[Image.Image] = []

    for i in range(n):
        arr = _render_frame_as_uint8(stac_mode, display_mode, i, scaling)
        if arr is None:
            continue

        img = Image.fromarray(arr, mode="RGB")

        # optional resize for file size
        if max_size is not None:
            w, h = img.size
            longest = max(w, h)
            if longest > max_size:
                scale = max_size / longest
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        # convert to palette image to reduce GIF size
        dith = Image.FLOYDSTEINBERG if dither else Image.NONE
        img = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=int(colors), dither=dith)

        frames.append(img)

    if len(frames) == 0:
        raise RuntimeError("No valid frames found. GIF cannot be created.")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_interval_ms,
        loop=0,
        optimize=True,
    )

    print(f"GIF saved: {output_path}  (frames={len(frames)}, colors={colors}, max_size={max_size})")
