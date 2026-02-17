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
    old_res=10.0,
    new_res=2.5,
    nan_pixel_buffer=8,
    edge_crop_px=8,
):
    """
    Super-resolve a SINGLE time slice.

    Correct order (as requested):
      1) SR full image
      2) Crop edges in HR (edge_crop_px)
      3) Apply NaN buffer in HR (nan_pixel_buffer)
    """
    da = da.sel(band=bands_to_use).rio.set_spatial_dims("x", "y", inplace=False)

    orig_band_order = da.band.values
    orig_attrs = dict(da.attrs)
    orig_attrs.pop("transform", None)
    orig_attrs.pop("grid_mapping", None)

    time_coord = da.coords.get("time", None)

    new_order = ["red", "green", "blue", "nir"]
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
    mask_lr = (
        da_square.isnull().any(dim="band").compute().to_numpy()
    )  # (y_lr, x_lr) bool

    # ============================================================
    # 2) INFERENCE INPUT
    # ============================================================
    X = torch.from_numpy(da_square.compute().to_numpy().astype("float32")).to(device)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    with suppress_tqdm():
        superX = sen2sr.predict_large(
            model=model, X=X, overlap=32
        )  # (band, y_hr_full, x_hr_full)

    # ============================================================
    # 3) UPSAMPLE LR MASK TO HR
    # ============================================================
    scale = int(round(old_res / new_res))  # 10m -> 2.5m => 4
    mask_lr_t = torch.from_numpy(mask_lr.astype(np.float32))[None, None, :, :].to(
        device
    )
    mask_hr_t = F.interpolate(
        mask_lr_t, scale_factor=scale, mode="nearest"
    )  # (1,1,y_hr_full,x_hr_full)
    mask_hr = mask_hr_t[0, 0].bool()  # (y_hr_full, x_hr_full)

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

    superX = superX[:, y0:y1, x0:x1]  # (band, y_hr, x_hr)
    mask_hr = mask_hr[y0:y1, x0:x1]  # (y_hr, x_hr)
    cropped_tf = full_tf * Affine.translation(x0, y0)

    # ============================================================
    # 6) EDGE CROP IN HR (BEFORE NAN BUFFER)
    # ============================================================
    if edge_crop_px > 0:
        H = superX.shape[1]
        W = superX.shape[2]

        # safety: avoid empty arrays
        if 2 * edge_crop_px < H and 2 * edge_crop_px < W:
            superX = superX[:, edge_crop_px:-edge_crop_px, edge_crop_px:-edge_crop_px]
            mask_hr = mask_hr[edge_crop_px:-edge_crop_px, edge_crop_px:-edge_crop_px]

            # update transform due to cropping from top-left
            cropped_tf = cropped_tf * Affine.translation(edge_crop_px, edge_crop_px)

    # ============================================================
    # 7) APPLY NAN BUFFER IN HR (AFTER EDGE CROP)
    # ============================================================
    if nan_pixel_buffer > 0:
        mask_hr_np = mask_hr.detach().cpu().numpy()
        mask_hr_np = dilate_mask_2d(mask_hr_np, radius_px=int(nan_pixel_buffer))
        mask_hr = torch.from_numpy(mask_hr_np).to(device)

    # apply buffered mask as NaN
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

    # restore original band order
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
    da_super.attrs["status"] = "super-resolved"
    da_super.attrs["nan_pixel_buffer"] = int(nan_pixel_buffer)
    da_super.attrs["edge_crop_px"] = int(edge_crop_px)

    return da_super


def super_resolve_cube(
    input_path,
    output_path: str | None = None,
    var_name="Spectral_Temporal_Stack",
    nan_pixel_buffer: int = 8,
):
    """
    Super-resolve full cube.

    - Crops 8 SR pixels from edges AFTER SR (inside superresolve_single_time)
    - Then applies nan_pixel_buffer AFTER that crop
    """
    if output_path is None:
        if isinstance(input_path, xr.DataArray):
            raise ValueError("Provide output_path when input_path is a DataArray.")
        base, ext = os.path.splitext(input_path)
        if ext == "":
            ext = ".nc"
        output_path = f"{base}_sr{ext}"

    bands_to_use = ["blue", "green", "red", "nir"]
    model_path = "model/SEN2SRLite_RGBN"
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
        raise ValueError(
            "Could not extract CRS WKT from CF grid mapping (spatial_ref)."
        )
    if tf is None:
        raise ValueError(
            "Could not extract/build transform from CF metadata or x/y coords."
        )

    indices = dataarray.attrs.get("indices", None)

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
            desc="Super-resolving time steps",
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
                old_res=old_res,
                new_res=new_res,
                nan_pixel_buffer=nan_pixel_buffer,
                edge_crop_px=edge_crop_px,
            )
            super_list.append(da_sr_t)

        da_super_all = xr.concat(
            super_list, dim="time", coords="minimal", compat="override"
        )
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
            old_res=old_res,
            new_res=new_res,
            nan_pixel_buffer=nan_pixel_buffer,
            edge_crop_px=edge_crop_px,
        )

    # ===========================
    # OPTIONAL: compute indices (only if provided in attrs)
    # ===========================
    if isinstance(indices, str):
        try:
            indices = ast.literal_eval(indices)
        except Exception:
            indices = [x.strip() for x in indices.split(",") if x.strip()]
    elif isinstance(indices, (tuple, list, np.ndarray)):
        indices = list(indices)

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
    da_super_all.attrs["status"] = "super-resolved"

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
        ds_out["spatial_ref"].attrs[
            "GeoTransform"
        ] = f"{tf_out.c} {tf_out.a} {tf_out.b} {tf_out.f} {tf_out.d} {tf_out.e}"

    tf_out = ds_out.rio.transform()
    transform9 = [
        float(tf_out.a),
        float(tf_out.b),
        float(tf_out.c),
        float(tf_out.d),
        float(tf_out.e),
        float(tf_out.f),
        0.0,
        0.0,
        1.0,
    ]
    ds_out.attrs["transform"] = transform9
    ds_out[var_name].attrs["transform"] = transform9

    ds_out.to_netcdf(output_path)
    print("Data cube is super-resolved to 2.5-meters!")
