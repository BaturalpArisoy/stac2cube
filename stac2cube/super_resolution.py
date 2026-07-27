import os
import io
import sys
import ast
import contextlib
import mlstac
import torch
import torch.nn.functional as F
import xarray as xr
import numpy as np
import rioxarray
from affine import Affine
from tqdm.auto import tqdm

from .get_spectral_indices import calculate_spectral_index
from .export_cfg import open_cube, is_zarr_path, _write_zarr, resolve_stack_var


@contextlib.contextmanager
def suppress_tqdm():
    with contextlib.redirect_stderr(io.StringIO()):
        yield

# ==========================================================
# SEN2SR model configs (labels in YOUR cube)
# ==========================================================
RGBN_REQUIRED = {"blue", "green", "red", "nir"}

# SEN2SRLite expects this 10-band stack (inputs are assumed already on the same grid, i.e., your cube grid)
FULL_REQUIRED = {
    "blue", "green", "red", "nir",
    "rededge1", "rededge2", "rededge3",
    "nir08", "swir16", "swir22",
}

# SEN2SR-required *model input order* (IMPORTANT)
# The standalone RGBN model takes the raw CNN input order (R,G,B,N - its
# mlm.json is correct). The full model's referencex4 wrapper instead expects
# Sentinel-2 NATURAL order (B02..B12): it hard-codes positions [0,1,2,6] as
# the 10-m natives and [3,4,5,7,8,9] as the 20-m bands, and extracts R,G,B,N
# for the internal CNN itself. Its mlm.json documents the internal CNN order
# (RGBN-first), which is WRONG for the wrapper: feeding that order sent native
# nir through the destructive 20-m branch (decimate+re-SR; lowpass corr with
# true nir 0.836 vs 0.998) and rededge3 through the RGBN branch. Verified via
# hard-constraint fingerprints on the model's own example data (2026-07-09).
RGBN_MODEL_ORDER = ["red", "green", "blue", "nir"]
FULL_MODEL_ORDER = [
    "blue", "green", "red",
    "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "swir16", "swir22",
]

# SEN2SRLite_Reference_RSWIR_x2 (20-m bands -> 10-m) input order: natural
# order, same as FULL_MODEL_ORDER (referencex2 hard-codes the same positional
# split as referencex4; its mlm.json band list is wrong the same way).
X2_MODEL_ORDER = list(FULL_MODEL_ORDER)


def _bands_in_cube_order(da: xr.DataArray, band_set: set[str]) -> list[str]:
    """Return bands present in da, preserving da.band order."""
    cube_order = [str(b) for b in da.coords["band"].values]
    return [b for b in cube_order if b in band_set]


def _validate_required_bands(available: set[str], required: set[str], model_type: str) -> None:
    missing = sorted(required - available)
    if missing:
        req = ", ".join(sorted(required))
        miss = ", ".join(missing)
        have = ", ".join(sorted(available))
        raise ValueError(
            f"model_type='{model_type}' requires these bands in the input cube:\n"
            f"  required: {req}\n"
            f"  missing:  {miss}\n"
            f"  present:  {have}\n\n"
            "Please rebuild your initial data cube including the missing bands."
        )


def _resolve_model_path(model_dir: str, candidates: list[str]) -> str:
    """
    Try resolving model folders robustly:
      1) <this_module_dir>/<model_dir>/<candidate>
      2) <cwd>/<model_dir>/<candidate>
    Falls back to the first candidate under cwd-style path.
    """
    bases = [
        os.path.join(os.path.dirname(__file__), model_dir),
        model_dir,
    ]
    for base in bases:
        for name in candidates:
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
    return os.path.join(model_dir, candidates[0])


def affine_from_xy_centers(x: np.ndarray, y: np.ndarray) -> Affine:
    x = np.asarray(x)
    y = np.asarray(y)
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    c = float(x[0] - dx * 0.5)
    f = float(y[0] - dy * 0.5)
    return Affine(dx, 0.0, c, 0.0, dy, f)


def _copy_time_coords(src: xr.DataArray, dst: xr.DataArray) -> xr.DataArray:
    """
    Copy all 1D time-dependent coordinates (except 'time' itself) from src to dst.
    This preserves things like cloud_percentage(time).
    """
    if "time" not in src.dims or "time" not in dst.dims:
        return dst

    for name, coord in src.coords.items():
        if name == "time":
            continue
        if coord.dims == ("time",) and name in src.coords:
            dst = dst.assign_coords({name: ("time", src[name].values)})
    return dst


def _extract_cf_crs_and_geotransform(ds: xr.Dataset, data_var: str):
    """
    Try to recover CRS+transform the same way CF NetCDFs store it for GIS.
    Priority:
      1) Find grid mapping var (via data_var attrs['grid_mapping']) and pull WKT + GeoTransform
      2) Fall back to ds/data_var attrs
      3) If no GeoTransform, build transform from x/y coords
    """
    da = ds[data_var]

    gm_name = da.attrs.get("grid_mapping", None)
    if gm_name is None and "spatial_ref" in ds.variables:
        gm_name = "spatial_ref"

    crs_wkt = None
    geotransform = None

    if gm_name is not None and gm_name in ds.variables:
        gm = ds[gm_name]

        for k in ("spatial_ref", "crs_wkt", "WKT", "proj_wkt", "esri_pe_string"):
            v = gm.attrs.get(k, None)
            if isinstance(v, str) and v.strip():
                crs_wkt = v
                break

        gt = gm.attrs.get("GeoTransform", None) or gm.attrs.get("geotransform", None)
        if isinstance(gt, str):
            parts = [p for p in gt.replace(",", " ").split() if p]
            if len(parts) == 6:
                geotransform = [float(p) for p in parts]
        elif isinstance(gt, (list, tuple)) and len(gt) == 6:
            geotransform = [float(p) for p in gt]

    if crs_wkt is None:
        crs_wkt = da.attrs.get("crs_wkt", None) or da.attrs.get("spatial_ref", None)

    tf = None
    if geotransform is not None:
        c, a, b, f, d, e = geotransform
        tf = Affine(a, b, c, d, e, f)
    else:
        if "x" in da.coords and "y" in da.coords:
            tf = affine_from_xy_centers(da["x"].values, da["y"].values)

    return crs_wkt, tf


def dilate_mask_2d(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Dilate a 2D boolean mask by `radius_px` pixels using max-pooling (fast, no scipy).
    mask: (H, W) bool
    """
    if radius_px <= 0:
        return mask

    m = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]  # (1,1,H,W)
    k = 2 * radius_px + 1
    m_dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius_px)
    return (m_dil[0, 0] > 0).cpu().numpy()


def _dilate_mask_t(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Dilate a 2D boolean mask tensor by `radius_px` pixels on its own device."""
    if radius_px <= 0:
        return mask
    m = mask.to(torch.float32)[None, None, :, :]
    k = 2 * radius_px + 1
    m_dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius_px)
    return m_dil[0, 0] > 0


# ==========================================================
# Optimized replacement for sen2sr.predict_large
# ==========================================================
PATCH_SIZE = 128  # SEN2SRLite hard-constraint masks are baked for 128-px patches


def _patch_positions(dim: int, chunk: int, step: int) -> list[int]:
    """Start positions of chunk-sized windows covering [0, dim), clamped to bounds."""
    if dim <= chunk:
        return [0]
    positions = [min(p, dim - chunk) for p in range(0, dim, step)]
    return list(dict.fromkeys(positions))  # dedupe clamped tail, keep order


def predict_large_batched(
    model: torch.nn.Module,
    X: torch.Tensor,
    overlap: int = 32,
    batch_size: int = 8,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Drop-in replacement for sen2sr.predict_large with the same stitching
    semantics, but:
      - runs under torch.inference_mode()
      - batches patches through the model instead of one forward per patch
      - one device->host transfer per batch instead of per patch
      - supports non-square inputs (any H, W >= 128)
      - optional mixed-precision autocast on CUDA
      - halves the batch and retries on CUDA OOM

    Returns the super-resolved image as a float32 CPU tensor (C_out, H*r, W*r).
    """
    device = X.device
    _, H, W = X.shape
    step = PATCH_SIZE - overlap
    positions = [
        (y, x)
        for y in _patch_positions(H, PATCH_SIZE, step)
        for x in _patch_positions(W, PATCH_SIZE, step)
    ]

    use_autocast = autocast_dtype is not None and device.type == "cuda"
    output = None
    res_n = skip = 0

    with torch.inference_mode():
        i = 0
        bs = max(1, int(batch_size))
        while i < len(positions):
            batch_pos = positions[i : i + bs]
            batch = torch.stack(
                [X[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE] for (y, x) in batch_pos]
            )
            try:
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        result = model(batch)
                    result = result.float()
                    if not torch.isfinite(result).all():
                        # fp16 overflows to NaN/Inf inside the model's FFT on
                        # very bright patches (snow, clouds); redo the batch in
                        # full precision instead of leaving holes in the output
                        result = model(batch)
                else:
                    result = model(batch)
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                del batch
                torch.cuda.empty_cache()
                continue

            if output is None:
                res_n = result.shape[-1] // PATCH_SIZE
                skip = overlap * res_n // 2
                output = torch.zeros(
                    (result.shape[1], H * res_n, W * res_n),
                    dtype=torch.float32,
                    device="cpu",
                )

            result = result.cpu()  # single transfer for the whole batch
            for k, (y, x) in enumerate(batch_pos):
                # keep everything except the leading half-overlap margin, which
                # was already written by the previous (overlapping) patch
                sy = 0 if y == 0 else skip
                sx = 0 if x == 0 else skip
                output[
                    :,
                    y * res_n + sy : (y + PATCH_SIZE) * res_n,
                    x * res_n + sx : (x + PATCH_SIZE) * res_n,
                ] = result[k, :, sy:, sx:]
            i += len(batch_pos)

    return output


def superresolve_single_time(
    da,
    crs_wkt,
    transform,
    model,
    device,
    bands_to_use,
    model_band_order,
    old_res=10.0,
    new_res=2.5,
    nan_pixel_buffer=8,
    edge_crop_px=8,
    batch_size=8,
    autocast_dtype=None,
):
    """
    Super-resolve a SINGLE time slice.

    Order:
      1) SR full image
      2) Crop edges in HR (edge_crop_px)
      3) Apply NaN buffer in HR (nan_pixel_buffer)
    """
    da = da.sel(band=bands_to_use)
    da.rio.set_spatial_dims("x", "y", inplace=True)

    # This is the "cube order" for the subset you selected (used to restore at the end)
    orig_band_order = da.band.values

    orig_attrs = dict(da.attrs)
    orig_attrs.pop("transform", None)
    orig_attrs.pop("grid_mapping", None)

    time_coord = da.coords.get("time", None)

    # --- reorder into SEN2SR-required input order ---
    new_order = list(model_band_order)
    da_reordered = da.sel(band=new_order).transpose("band", "y", "x")

    # --- pad each axis independently to at least one 128-px patch ---
    # (the batched predictor handles non-square inputs, so no square padding:
    #  that used to waste up to ~2.5x compute on elongated AOIs)
    ny, nx = da_reordered.sizes["y"], da_reordered.sizes["x"]
    pad_y = max(PATCH_SIZE - ny, 0)
    pad_x = max(PATCH_SIZE - nx, 0)
    pad_dict = {
        "y": (pad_y // 2, pad_y - pad_y // 2),
        "x": (pad_x // 2, pad_x - pad_x // 2),
    }

    # ============================================================
    # 1) READ ONCE, BUILD LR NAN/CLOUD MASK + INFERENCE INPUT
    # ============================================================
    arr_lr = da_reordered.to_numpy().astype(np.float32, copy=False)
    arr_lr = np.pad(
        arr_lr, ((0, 0), pad_dict["y"], pad_dict["x"]), constant_values=0.0
    )
    mask_lr = np.isnan(arr_lr).any(axis=0)

    # ============================================================
    # 2) INFERENCE
    # ============================================================
    X = torch.from_numpy(arr_lr).to(device)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    superX = predict_large_batched(
        model=model,
        X=X,
        overlap=32,
        batch_size=batch_size,
        autocast_dtype=autocast_dtype,
    )

    # ============================================================
    # 3) UPSAMPLE LR MASK TO HR (stays on `device` until applied)
    # ============================================================
    scale = int(round(old_res / new_res))  # 10m -> 2.5m => 4
    mask_lr_t = torch.from_numpy(mask_lr.astype(np.float32))[None, None, :, :].to(device)
    mask_hr_t = F.interpolate(mask_lr_t, scale_factor=scale, mode="nearest")
    mask_hr = mask_hr_t[0, 0].bool()

    # ============================================================
    # 4) BUILD FULL HR TRANSFORM
    # ============================================================
    pad_y_top_hr = pad_dict["y"][0] * scale
    pad_x_left_hr = pad_dict["x"][0] * scale

    scaled_tf = transform * Affine.scale(1 / scale, 1 / scale)
    full_tf = scaled_tf * Affine.translation(-pad_x_left_hr, -pad_y_top_hr)

    # ============================================================
    # 5) CROP PADDING BACK TO ORIGINAL HR EXTENT
    # ============================================================
    orig_h_lr = da_reordered.sizes["y"]
    orig_w_lr = da_reordered.sizes["x"]

    y0, y1 = pad_y_top_hr, pad_y_top_hr + orig_h_lr * scale
    x0, x1 = pad_x_left_hr, pad_x_left_hr + orig_w_lr * scale

    superX = superX[:, y0:y1, x0:x1]
    mask_hr = mask_hr[y0:y1, x0:x1]
    cropped_tf = full_tf * Affine.translation(x0, y0)

    # ============================================================
    # 6) EDGE CROP IN HR (BEFORE NAN BUFFER)
    # ============================================================
    if edge_crop_px > 0:
        H = superX.shape[1]
        W = superX.shape[2]
        if 2 * edge_crop_px < H and 2 * edge_crop_px < W:
            superX = superX[:, edge_crop_px:-edge_crop_px, edge_crop_px:-edge_crop_px]
            mask_hr = mask_hr[edge_crop_px:-edge_crop_px, edge_crop_px:-edge_crop_px]
            cropped_tf = cropped_tf * Affine.translation(edge_crop_px, edge_crop_px)

    # ============================================================
    # 7) APPLY NAN BUFFER IN HR (AFTER EDGE CROP)
    # ============================================================
    if nan_pixel_buffer > 0:
        mask_hr = _dilate_mask_t(mask_hr, radius_px=int(nan_pixel_buffer))

    superX = superX.contiguous()
    superX.masked_fill_(mask_hr.cpu().unsqueeze(0), float("nan"))

    # ============================================================
    # 8) BUILD FINAL XARRAY OUTPUT (already cropped)
    # ============================================================
    arr = superX.numpy()
    var_name = da.name or "Time_Series"

    da_hr = xr.DataArray(
        arr,
        dims=("band", "y", "x"),
        coords={
            "band": new_order,
            "y": np.arange(arr.shape[1]),
            "x": np.arange(arr.shape[2]),
        },
        name=var_name,
    )
    da_hr.attrs = orig_attrs
    if time_coord is not None:
        da_hr = da_hr.assign_coords(time=time_coord)

    ds_tmp = da_hr.to_dataset(name=var_name)
    ds_tmp.rio.set_spatial_dims("x", "y", inplace=True)
    ds_tmp.rio.write_crs(crs_wkt, inplace=True)
    ds_tmp.rio.write_transform(cropped_tf, inplace=True)

    # restore "cube order" for the selected subset
    ds_tmp = ds_tmp.sel(band=orig_band_order)

    # assign world x/y centers from cropped_tf
    W = ds_tmp.sizes["x"]
    H = ds_tmp.sizes["y"]
    xs = cropped_tf.c + cropped_tf.a * (np.arange(W) + 0.5)
    ys = cropped_tf.f + cropped_tf.e * (np.arange(H) + 0.5)

    ds_tmp = ds_tmp.assign_coords(
        x=("x", xs.astype(np.float64)),
        y=("y", ys.astype(np.float64)),
    )

    da_super = ds_tmp[var_name]
    da_super.rio.set_spatial_dims("x", "y", inplace=True)
    da_super.rio.write_crs(crs_wkt, inplace=True)
    da_super.rio.write_transform(cropped_tf, inplace=True)

    da_super.attrs.update(orig_attrs)
    #da_super.attrs["status"] = "super-resolved"

    return da_super


def super_resolve_cube(
    input_path,
    output_path: str | None = None,
    var_name="Time_Series",
    nan_pixel_buffer: int | None = None,  # NaN-mask dilation in OUTPUT pixels; None = per-mode default
    model_type: str | None = None,  # None | "rgbn" | "full_spectral" | "20to10"
    model_dir: str | None = None,   # folder containing the 'model' dir (e.g. interactive/)
    batch_size: int | None = None,  # patches per forward pass; None = auto
    precision: str = "auto",        # "auto" | "fp32" | "fp16" | "bf16"
    compress: bool = False,         # zlib-compress the output NetCDF (lossless, ~10x slower export)
    pack_to_int16: bool = True,     # CF scale/offset int16 packing (near-lossless, 2x smaller, fast)
):
    """
    Super-resolve full cube.

    model_type:
      - None (default): auto-detect using attrs["spectral_bands"]
          * if spectral_bands are ONLY within [blue, green, red, nir] -> rgbn
          * otherwise -> full_spectral
      - "rgbn": use SEN2SRLite-RGBN (10-m RGBN -> 2.5-m)
      - "full_spectral": use SEN2SRLite (requires all 10 bands; -> 2.5-m)
      - "20to10": use SEN2SRLite_Reference_RSWIR_x2. Enhances the six 20-m
        bands (rededge1, rededge2, rededge3, nir08, swir16, swir22) to true
        10-m detail, guided by the four native 10-m bands. Requires the same
        10 bands as "full_spectral", stored on a 10-m grid (the cube must be
        built at 10-m resolution). The 10-m bands pass through unchanged; the
        output grid is identical to the input grid (no pixel-size change).

    nan_pixel_buffer:
      Radius (in OUTPUT pixels) by which the NaN/cloud mask is dilated to
      remove SR halo artifacts around masked areas. None (default) picks 8 for
      the 2.5-m modes and 2 for "20to10" - the same 20-m physical buffer in
      both cases.

    batch_size:
      Number of 128x128 patches super-resolved per model call. None picks a
      size from free GPU memory (falls back automatically on CUDA OOM); on CPU
      a small batch is used.

    precision:
      - "auto" (default): full fp32 everywhere. fp16 is NOT used by default
        because it overflows to NaN on very bright patches (snow, clouds).
      - "fp32": same as "auto".
      - "fp16"/"bf16": force the given autocast dtype on CUDA. Any batch
        whose output comes back non-finite is automatically redone in fp32,
        so this no longer punches holes in bright scenes.

    compress:
      Write the output with chunked zlib compression (lossless). Off by
      default: zlib is the slow part of the export (roughly 10x slower write
      for only ~1.25x further shrink on top of int16 packing). Turn on when
      disk size matters more than export time (e.g. archival). Any NetCDF
      reader (xarray, GDAL, QGIS) handles it transparently.

    pack_to_int16:
      Store values as int16 with CF scale_factor/add_offset (the same scheme
      Sentinel-2 uses natively) for a ~2x size reduction at almost no time
      cost. The scale is derived from the actual data range; if the range is
      too wide to keep at least 1e-4 precision (coarser than S2's own
      quantization), packing is skipped automatically and float32 is written
      instead. Readers unpack transparently. Set False for bit-exact float32
      output.

    Output container:
      The output_path extension picks the format: ``*.zarr`` -> Zarr store,
      anything else -> NetCDF (unchanged behavior). With no output_path the
      input's own extension is kept (zarr in -> zarr out). ``compress`` is
      NetCDF-only (Zarr always applies its default lossless codec). Zarr
      caveat with pack_to_int16: Zarr attrs are JSON and cannot hold a
      float32 scale_factor, so readers decode the packed values to float64 -
      equal to the NetCDF float32 decode within float32 rounding, not
      bit-identical.
    """

    # ---------------------------
    # local helpers (kept inside for copy/paste simplicity)
    # ---------------------------
    RGBN_REQUIRED = {"blue", "green", "red", "nir"}
    FULL_REQUIRED = {
        "blue", "green", "red", "nir",
        "rededge1", "rededge2", "rededge3",
        "nir08", "swir16", "swir22",
    }

    RGBN_MODEL_ORDER = ["red", "green", "blue", "nir"]
    # Both wrapped models (referencex4 and referencex2) expect Sentinel-2
    # NATURAL order (B02..B12) - see the module-level constants' note.
    FULL_MODEL_ORDER = [
        "blue", "green", "red",
        "rededge1", "rededge2", "rededge3",
        "nir", "nir08", "swir16", "swir22",
    ]
    X2_MODEL_ORDER = list(FULL_MODEL_ORDER)

    def _parse_list_attr(v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except Exception:
                return [x.strip() for x in v.split(",") if x.strip()]
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
        return None

    def _validate_required_bands(available: set[str], required: set[str], mt: str) -> None:
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"model_type='{mt}' requires these bands in the input cube:\n\n"
                f"  required: {', '.join(sorted(required))}\n"
                f"  missing:  {', '.join(missing)}\n"
                f"  present:  {', '.join(sorted(available))}\n\n"
                "Please rebuild your initial data cube including the missing bands."
            )

    def _bands_in_cube_order(da: xr.DataArray, required_set: set[str]) -> list[str]:
        cube_order = [str(b) for b in da.coords["band"].values]
        return [b for b in cube_order if b in required_set]

    def _resolve_model_path(candidates: list[str]) -> str:
        search = []
        # 0) explicit model_dir param: accept both "<dir>/model/<name>" and "<dir>/<name>"
        if model_dir:
            for p in candidates:
                search.append(os.path.join(model_dir, p))
                search.append(os.path.join(model_dir, os.path.basename(p)))
        # 1) relative to cwd
        search.extend(candidates)
        # 2) relative to this file (module dir)
        base = os.path.join(os.path.dirname(__file__), "")
        search.extend(os.path.join(base, p) for p in candidates)

        for cand in search:
            if os.path.exists(cand):
                return cand

        # Not found: fail clearly instead of handing a bad relative path to mlstac,
        # which otherwise raises a confusing "Unsupported scheme: snippet" error.
        raise FileNotFoundError(
            f"Could not find the SEN2SRLite model {candidates}. Looked under "
            + (f"model_dir={model_dir!r}, " if model_dir else "")
            + "the current working directory, and the stac2cube package directory. "
            "Set model_dir to the folder that contains the 'model' directory "
            "(e.g. the 'interactive' folder)."
        )

    # ---------------------------
    # output path
    # ---------------------------
    if output_path is None:
        if isinstance(input_path, xr.DataArray):
            raise ValueError("Provide output_path when input_path is a DataArray.")
        base, ext = os.path.splitext(input_path)
        if ext == "":
            ext = ".nc"
        output_path = f"{base}_sr{ext}"

    if isinstance(input_path, xr.DataArray):
        raise ValueError(
            "This version expects a NetCDF/Zarr cube path so it can read CF georef from metadata."
        )

    ds_in = open_cube(input_path)
    # open_cube already migrated a legacy time-series name; map a legacy
    # var_name typed by an older script forward too, so both keep working.
    var_name = resolve_stack_var(ds_in, var_name)
    dataarray = ds_in[var_name]

    crs_wkt, tf = _extract_cf_crs_and_geotransform(ds_in, var_name)
    if crs_wkt is None:
        raise ValueError("Could not extract CRS WKT from CF grid mapping (spatial_ref).")
    if tf is None:
        raise ValueError("Could not extract/build transform from CF metadata or x/y coords.")

    # attributes
    indices = _parse_list_attr(dataarray.attrs.get("indices", None))
    spectral_bands_attr = _parse_list_attr(dataarray.attrs.get("spectral_bands", None))

    # band coordinate set (what truly exists in the data)
    band_coord_set = {str(b) for b in dataarray.coords["band"].values}

    # spectral_set is what you want to use for detection/validation
    if spectral_bands_attr and len(spectral_bands_attr) > 0:
        spectral_set = {str(b) for b in spectral_bands_attr}
    else:
        # fallback: treat all "band" entries except indices as spectral
        idx_set = set(indices) if indices else set()
        spectral_set = band_coord_set - idx_set

    # sanity: spectral bands listed must exist in band coordinate
    missing_from_coord = sorted(spectral_set - band_coord_set)
    if missing_from_coord:
        raise ValueError(
            "Your data cube attrs['spectral_bands'] lists bands that are not present in the 'band' coordinate:\n"
            f"  missing in coord: {', '.join(missing_from_coord)}\n"
            "Please fix the cube metadata or rebuild the cube."
        )

    # ---------------------------
    # decide model type
    # ---------------------------
    if model_type is None:
        if spectral_set.issubset(RGBN_REQUIRED):
            model_type_used = "rgbn"
        else:
            model_type_used = "full_spectral"
    else:
        mt = str(model_type).strip().lower()
        if mt not in ("rgbn", "full_spectral", "20to10"):
            raise ValueError("model_type must be one of: None, 'rgbn', 'full_spectral', '20to10'")
        model_type_used = mt

    # ---------------------------
    # select bands + model input order + model path + resolutions
    # ---------------------------
    # The 2.5-m modes upscale the grid 4x; edge_crop_px=8 output px = 2 input
    # px cropped to remove boundary artifacts. "20to10" keeps the grid size
    # (the six 20-m bands gain detail on the existing 10-m grid), so the same
    # 2-input-px crop is edge_crop_px=2.
    old_res = 10.0
    if model_type_used == "20to10":
        new_res = 10.0
        edge_crop_px = 2
        default_nan_buffer = 2
    else:
        new_res = 2.5
        edge_crop_px = 8
        default_nan_buffer = 8
    if nan_pixel_buffer is None:
        nan_pixel_buffer = default_nan_buffer

    if model_type_used == "rgbn":
        _validate_required_bands(spectral_set, RGBN_REQUIRED, model_type_used)
        bands_to_use = _bands_in_cube_order(dataarray, RGBN_REQUIRED)  # cube order for restore
        model_band_order = RGBN_MODEL_ORDER
        model_path = _resolve_model_path([
            "model/SEN2SRLite-RGBN",
            "model/SEN2SRLite_RGBN",
        ])
    elif model_type_used == "20to10":
        # needs the four 10-m natives as reference plus the six 20-m bands
        _validate_required_bands(spectral_set, FULL_REQUIRED, model_type_used)
        bands_to_use = _bands_in_cube_order(dataarray, FULL_REQUIRED)  # cube order for restore
        model_band_order = X2_MODEL_ORDER
        model_path = _resolve_model_path([
            "model/SEN2SRLite_Reference_RSWIR_x2",
        ])
    else:
        _validate_required_bands(spectral_set, FULL_REQUIRED, model_type_used)
        bands_to_use = _bands_in_cube_order(dataarray, FULL_REQUIRED)  # cube order for restore
        model_band_order = FULL_MODEL_ORDER
        model_path = _resolve_model_path([
            "model/SEN2SRLite",
        ])

    dataarray_sub = dataarray.sel(band=bands_to_use)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # inference performance setup
    # ---------------------------
    autocast_dtype = None
    if device.type == "cuda":
        # every forward pass sees the same (B, C, 128, 128) shape, so let
        # cuDNN benchmark kernel choices once and reuse them
        torch.backends.cudnn.benchmark = True

        prec = str(precision).strip().lower()
        if prec == "fp16":
            autocast_dtype = torch.float16
        elif prec == "bf16":
            autocast_dtype = torch.bfloat16
        elif prec not in ("auto", "fp32"):
            raise ValueError("precision must be one of: 'auto', 'fp32', 'fp16', 'bf16'")
        # "auto" runs full fp32: fp16 autocast overflows to NaN inside the
        # model's FFT on very bright patches (snow, clouds), punching holes in
        # the output. The speed win comes from batching + inference_mode, not
        # from half precision.

    if batch_size is None:
        if device.type == "cuda":
            # rough per-patch activation budget; OOM retry in the predictor
            # shrinks the batch if this guess is too optimistic
            free_bytes, _ = torch.cuda.mem_get_info()
            per_patch = 192 * 1024**2 if autocast_dtype is None else 128 * 1024**2
            batch_size = int(min(32, max(1, free_bytes // (2 * per_patch))))
        else:
            # on CPU, intra-op threading already uses all cores at batch 1;
            # bigger batches only add memory pressure
            batch_size = 1

    model = mlstac.load(model_path).compiled_model(device=device)

    if device.type == "cuda":
        print(
            f"CUDA acceleration is detected! Super-resolution will run on GPU "
            f"({torch.cuda.get_device_name(0)}).",
            flush=True,
        )
    else:
        print(
            "No CUDA GPU detected - super-resolution will run on CPU (slower).",
            flush=True,
        )

    # ===========================
    # SUPER-RESOLVE (time or single)
    # ===========================
    if "time" in dataarray_sub.dims:
        super_list = []
        for t in tqdm(
            dataarray_sub.time.values,
            desc=f"Super-resolving time steps ({model_type_used})",
            unit="date",
            file=sys.stdout,
            dynamic_ncols=False,
        ):
            da_t = dataarray_sub.sel(time=t)

            da_sr_t = superresolve_single_time(
                da=da_t,
                crs_wkt=crs_wkt,
                transform=tf,
                model=model,
                device=device,
                bands_to_use=bands_to_use,
                model_band_order=model_band_order,  # <-- requires your updated single_time
                old_res=old_res,
                new_res=new_res,
                nan_pixel_buffer=nan_pixel_buffer,
                edge_crop_px=edge_crop_px,
                batch_size=batch_size,
                autocast_dtype=autocast_dtype,
            )
            super_list.append(da_sr_t)

        da_super_all = xr.concat(super_list, dim="time", coords="minimal", compat="override")
        da_super_all = _copy_time_coords(dataarray, da_super_all)

        tf0 = super_list[0].rio.transform()
        da_super_all.rio.set_spatial_dims("x", "y", inplace=True)
        da_super_all.rio.write_crs(crs_wkt, inplace=True)
        da_super_all.rio.write_transform(tf0, inplace=True)

    else:
        # A single, time-less image (e.g. a median composite). Wrap the one call
        # in tqdm too so the GUI/log shows the same progress bar as the time
        # series case, just counting to 1 (0/1 -> 1/1).
        for _ in tqdm(
            range(1),
            desc=f"Super-resolving image ({model_type_used})",
            unit="image",
            file=sys.stdout,
            dynamic_ncols=False,
        ):
            da_super_all = superresolve_single_time(
                da=dataarray_sub,
                crs_wkt=crs_wkt,
                transform=tf,
                model=model,
                device=device,
                bands_to_use=bands_to_use,
                model_band_order=model_band_order,  # <-- requires your updated single_time
                old_res=old_res,
                new_res=new_res,
                nan_pixel_buffer=nan_pixel_buffer,
                edge_crop_px=edge_crop_px,
                batch_size=batch_size,
                autocast_dtype=autocast_dtype,
            )
        # tf0 is also used by the indices block below; without this line a
        # time-less input (e.g. a median composite) with indices attrs raised
        # NameError because tf0 was only set in the time-series branch.
        tf0 = da_super_all.rio.transform()

    print("Super-resolving is done, now exporting...", flush=True)

    # ===========================
    # OPTIONAL: compute indices (only if provided in attrs)
    # ===========================
    if indices and len(indices) > 0:
        stac_idx = calculate_spectral_index(da_super_all, mission="s2", indices=indices)

        if isinstance(stac_idx, xr.Dataset):
            stac_idx = stac_idx.to_array(dim="band")
            stac_idx = stac_idx.assign_coords(band=list(stac_idx["band"].values))
        elif isinstance(stac_idx, xr.DataArray):
            if "band" not in stac_idx.dims:
                for d in stac_idx.dims:
                    if d not in ("time", "x", "y"):
                        stac_idx = stac_idx.rename({d: "band"})
                        break
            if "band" not in stac_idx.dims:
                idx_name = stac_idx.name or "index"
                stac_idx = stac_idx.expand_dims(band=[idx_name])

        da_super_all = xr.concat([da_super_all, stac_idx], dim="band")
        da_super_all.rio.set_spatial_dims("x", "y", inplace=True)
        da_super_all.rio.write_crs(crs_wkt, inplace=True)
        da_super_all.rio.write_transform(tf0, inplace=True)
        da_super_all.attrs["indices"] = indices

    # ===========================
    # FINALIZE + WRITE NETCDF
    # ===========================
    da_super_all.name = var_name
    #da_super_all.attrs["status"] = f"super_resolved_{model_type_used}"

    ds_out = da_super_all.to_dataset(name=var_name)
    tf_out_final = da_super_all.rio.transform()
    ds_out.rio.set_spatial_dims("x", "y", inplace=True)
    ds_out.rio.write_crs(crs_wkt, inplace=True)
    ds_out.rio.write_transform(tf_out_final, inplace=True)

    ds_out[var_name].attrs.pop("grid_mapping", None)
    ds_out[var_name].encoding["grid_mapping"] = "spatial_ref"

    try:
        from rasterio.crs import CRS
        epsg = CRS.from_wkt(crs_wkt).to_epsg()
        if epsg is not None:
            crs_epsg = f"EPSG:{epsg}"
            ds_out.attrs["crs"] = crs_epsg
            ds_out[var_name].attrs["crs"] = crs_epsg
    except Exception:
        pass

    if "spatial_ref" in ds_out.variables:
        tf_out = ds_out.rio.transform()
        ds_out["spatial_ref"].attrs["GeoTransform"] = (
            f"{tf_out.c} {tf_out.a} {tf_out.b} {tf_out.f} {tf_out.d} {tf_out.e}"
        )

    tf_out = ds_out.rio.transform()
    transform9 = [
        float(tf_out.a),
        float(tf_out.b),
        float(tf_out.c),
        float(tf_out.d),
        float(tf_out.e),
        float(tf_out.f),
        0.0, 0.0, 1.0,
    ]
    ds_out.attrs["transform"] = transform9
    ds_out[var_name].attrs["transform"] = transform9

    # ---------------------------
    # output encoding (compression / int16 packing)
    # ---------------------------
    # The output extension picks the container: *.zarr -> Zarr store, else
    # NetCDF (unchanged). zlib/chunksizes are NetCDF/HDF5-only knobs, so
    # `compress` is a no-op on the Zarr side (mirrors export_stac); Zarr always
    # compresses, and when `pack_to_int16` is on _write_zarr additionally
    # switches that variable to a shuffling codec, which is what makes a packed
    # store competitive with the NetCDF one (see _set_zarr_shuffle).
    write_zarr = is_zarr_path(output_path)
    var_encoding = {}
    if not write_zarr and (compress or pack_to_int16):
        # chunk per band/time so partial reads stay cheap and chunks compress well
        var_encoding["chunksizes"] = tuple(
            min(512, ds_out.sizes[d]) if d in ("y", "x") else 1
            for d in ds_out[var_name].dims
        )
    if not write_zarr and compress:
        var_encoding.update({"zlib": True, "complevel": 4})

    if pack_to_int16:
        vals = ds_out[var_name].values
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        # 65534 int16 steps must resolve at least 1e-4 (S2's native precision)
        if np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) <= 6.5:
            scale = max((vmax - vmin) / (2**16 - 2), 1e-12)
            offset = (vmax + vmin) / 2.0
            fill = np.int16(-32768)
            # Pack to int16 here, slab by slab, instead of via encoding
            # dtype/scale_factor: xarray's encode-time packing materializes
            # two extra full-size float copies of the cube (peak ~3.5x cube
            # size), which OOM-kills large scenes at export. Slab packing
            # peaks at ~1.5x (float cube + int16 copy) and the backend then
            # writes half the bytes.
            packed = np.empty(vals.shape, np.int16)
            slab_elems = 32 * 1024 * 1024  # ~128 MB float32 temp per slab
            rows_per_slab = max(1, slab_elems // max(1, vals.shape[-1]))
            for idx in np.ndindex(vals.shape[:-2]):
                for r in range(0, vals.shape[-2], rows_per_slab):
                    s = (*idx, slice(r, r + rows_per_slab))
                    tmp = (vals[s] - np.float32(offset)) / np.float32(scale)
                    np.rint(tmp, out=tmp)
                    nan_mask = np.isnan(tmp)
                    tmp[nan_mask] = 0.0
                    np.clip(tmp, -32767, 32767, out=tmp)
                    ints = tmp.astype(np.int16)
                    ints[nan_mask] = fill
                    packed[s] = ints
            packed_da = ds_out[var_name].copy(data=packed)
            # CF unpacking attrs (float32 so readers decode to float32, not
            # float64). _FillValue goes in attrs, not encoding: the netCDF4
            # backend pops it at variable creation, while encoding._FillValue
            # would trigger a mask/fill pass that copies the whole cube again.
            packed_da.attrs["scale_factor"] = np.float32(scale)
            packed_da.attrs["add_offset"] = np.float32(offset)
            packed_da.attrs["_FillValue"] = fill
            # a leftover encoding._FillValue (e.g. inherited from an input
            # file) would clash with the attrs one and abort the write
            packed_da.encoding.pop("_FillValue", None)
            ds_out[var_name] = packed_da
            ds_out[var_name].encoding["grid_mapping"] = "spatial_ref"
        else:
            print(
                "Note: data range too wide (or non-finite) for int16 packing "
                "at 1e-4 precision; writing float32 instead."
            )

    # Merge into the variable's existing .encoding (which holds grid_mapping=
    # "spatial_ref") and write with a plain to_netcdf. Passing encoding={var_name:
    # ...} to to_netcdf would override var.encoding and drop grid_mapping, leaving
    # the output ungeoreferenced. Eager (non-dask) write on purpose: writing a
    # dask-backed variable to netCDF4/HDF5 is not thread-safe (causes "NetCDF: HDF
    # error") and is far slower here.
    if write_zarr:
        # Same writer as export_stac: strips NetCDF-only encoding keys (keeps
        # grid_mapping), rechunks per (time, band) slice and streams via dask.
        # The int16 packing above is container-independent: the store holds
        # int16 + CF scale/offset attrs and readers unpack transparently.
        # Caveat (Zarr only): attributes are JSON, which cannot express a
        # float32 scale_factor, so CF decoding yields float64 values - equal
        # to the NetCDF float32 decode within float32 rounding (<= 1 ulp),
        # not bit-identical. NaN holes (_FillValue) restore exactly.
        _write_zarr(ds_out, output_path, overwrite=True)
        from pathlib import Path

        size_mb = sum(
            f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
        ) / 1e6
    else:
        if var_encoding:
            ds_out[var_name].encoding.update(var_encoding)
        ds_out.to_netcdf(output_path)
        size_mb = os.path.getsize(output_path) / 1e6
    target_str = "10-meters" if model_type_used == "20to10" else "2.5-meters"
    print(
        f"Data cube is super-resolved to {target_str}! "
        f"(model_type={model_type_used}, {size_mb:.1f} MB)"
    )
