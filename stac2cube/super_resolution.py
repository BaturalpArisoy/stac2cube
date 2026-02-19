import os
import io
import sys
import math
import ast
import contextlib
import mlstac
import torch
import torch.nn.functional as F
import xarray as xr
import numpy as np
import sen2sr
import rioxarray
from affine import Affine
from tqdm.auto import tqdm

from .get_spectral_indices import calculate_spectral_index


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
RGBN_MODEL_ORDER = ["red", "green", "blue", "nir"]
FULL_MODEL_ORDER = [
    "red", "green", "blue", "nir",
    "rededge1", "rededge2", "rededge3",
    "nir08", "swir16", "swir22",
]


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
):
    """
    Super-resolve a SINGLE time slice.

    Order:
      1) SR full image
      2) Crop edges in HR (edge_crop_px)
      3) Apply NaN buffer in HR (nan_pixel_buffer)
    """
    da = da.sel(band=bands_to_use).rio.set_spatial_dims("x", "y", inplace=False)

    # This is the "cube order" for the subset you selected (used to restore at the end)
    orig_band_order = da.band.values

    orig_attrs = dict(da.attrs)
    orig_attrs.pop("transform", None)
    orig_attrs.pop("grid_mapping", None)

    time_coord = da.coords.get("time", None)

    # --- reorder into SEN2SR-required input order ---
    new_order = list(model_band_order)
    da_reordered = da.sel(band=new_order)

    # --- pad to square for SEN2SR ---
    ny, nx = da_reordered.sizes["y"], da_reordered.sizes["x"]
    new_side = math.ceil(max(ny, nx) / 128) * 128

    pad_y = new_side - ny
    pad_x = new_side - nx
    pad_dict = {
        "y": (pad_y // 2, pad_y - pad_y // 2),
        "x": (pad_x // 2, pad_x - pad_x // 2),
    }
    da_square = da_reordered.pad(pad_dict, constant_values=0)

    # ============================================================
    # 1) BUILD LR NAN/CLOUD MASK
    # ============================================================
    mask_lr = da_square.isnull().any(dim="band").compute().to_numpy()

    # ============================================================
    # 2) INFERENCE INPUT
    # ============================================================
    X = torch.from_numpy(da_square.compute().to_numpy().astype("float32")).to(device)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    with suppress_tqdm():
        superX = sen2sr.predict_large(model=model, X=X, overlap=32)

    # ============================================================
    # 3) UPSAMPLE LR MASK TO HR
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
        mask_hr_np = mask_hr.detach().cpu().numpy()
        mask_hr_np = dilate_mask_2d(mask_hr_np, radius_px=int(nan_pixel_buffer))
        mask_hr = torch.from_numpy(mask_hr_np).to(device)

    superX[:, mask_hr.bool()] = float("nan")

    # ============================================================
    # 8) BUILD FINAL XARRAY OUTPUT (already cropped)
    # ============================================================
    arr = superX.detach().cpu().numpy()
    var_name = da.name or "Spectral_Temporal_Stack"

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

    ds_tmp = (
        da_hr.to_dataset(name=var_name)
        .rio.set_spatial_dims("x", "y", inplace=False)
        .rio.write_crs(crs_wkt, inplace=False)
        .rio.write_transform(cropped_tf, inplace=False)
    )

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

    da_super = ds_tmp[var_name].rio.set_spatial_dims("x", "y", inplace=False)
    da_super = da_super.rio.write_crs(crs_wkt, inplace=False).rio.write_transform(
        cropped_tf, inplace=False
    )

    da_super.attrs.update(orig_attrs)
    #da_super.attrs["status"] = "super-resolved"

    return da_super


def super_resolve_cube(
    input_path,
    output_path: str | None = None,
    var_name="Spectral_Temporal_Stack",
    nan_pixel_buffer: int = 8,
    model_type: str | None = None,  # NEW: None | "rgbn" | "full_spectral"
):
    """
    Super-resolve full cube.

    model_type:
      - None (default): auto-detect using attrs["spectral_bands"]
          * if spectral_bands are ONLY within [blue, green, red, nir] -> rgbn
          * otherwise -> full_spectral
      - "rgbn": use SEN2SRLite-RGBN
      - "full_spectral": use SEN2SRLite (requires all 10 bands)
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
    FULL_MODEL_ORDER = [
        "red", "green", "blue", "nir",
        "rededge1", "rededge2", "rededge3",
        "nir08", "swir16", "swir22",
    ]

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
        # Try: relative to cwd
        for p in candidates:
            if os.path.exists(p):
                return p
        # Try: relative to this file (module dir)
        base = os.path.join(os.path.dirname(__file__), "")
        for p in candidates:
            p2 = os.path.join(base, p)
            if os.path.exists(p2):
                return p2
        # fallback to first candidate
        return candidates[0]

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

    old_res = 10.0
    new_res = 2.5
    edge_crop_px = 8

    if isinstance(input_path, xr.DataArray):
        raise ValueError(
            "This version expects a NetCDF file path so it can read CF georef from metadata."
        )

    ds_in = xr.open_dataset(input_path)
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
        if mt not in ("rgbn", "full_spectral"):
            raise ValueError("model_type must be one of: None, 'rgbn', 'full_spectral'")
        model_type_used = mt

    # ---------------------------
    # select bands + model input order + model path
    # ---------------------------
    if model_type_used == "rgbn":
        _validate_required_bands(spectral_set, RGBN_REQUIRED, model_type_used)
        bands_to_use = _bands_in_cube_order(dataarray, RGBN_REQUIRED)  # cube order for restore
        model_band_order = RGBN_MODEL_ORDER
        model_path = _resolve_model_path([
            "model/SEN2SRLite-RGBN",
            "model/SEN2SRLite_RGBN",
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
    model = mlstac.load(model_path).compiled_model(device=device)

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
            )
            super_list.append(da_sr_t)

        da_super_all = xr.concat(super_list, dim="time", coords="minimal", compat="override")
        da_super_all = _copy_time_coords(dataarray, da_super_all)

        tf0 = super_list[0].rio.transform()
        da_super_all = (
            da_super_all.rio.set_spatial_dims("x", "y", inplace=False)
            .rio.write_crs(crs_wkt, inplace=False)
            .rio.write_transform(tf0, inplace=False)
        )

    else:
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
        )

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
        da_super_all = (
            da_super_all.rio.set_spatial_dims("x", "y", inplace=False)
            .rio.write_crs(crs_wkt, inplace=False)
            .rio.write_transform(da_super_all.rio.transform(), inplace=False)
        )
        da_super_all.attrs["indices"] = indices

    # ===========================
    # FINALIZE + WRITE NETCDF
    # ===========================
    da_super_all.name = var_name
    #da_super_all.attrs["status"] = f"super_resolved_{model_type_used}"

    ds_out = da_super_all.to_dataset(name=var_name)
    ds_out = ds_out.rio.set_spatial_dims("x", "y", inplace=False)
    ds_out = ds_out.rio.write_crs(crs_wkt, inplace=False)
    ds_out = ds_out.rio.write_transform(da_super_all.rio.transform(), inplace=False)

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

    ds_out.to_netcdf(output_path)
    print(f"Data cube is super-resolved to 2.5-meters! (model_type={model_type_used})")
