import os
import io
import sys
import math
import contextlib
import mlstac
import torch
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
            # attach the full vector to dst
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

    # --- Find the grid mapping variable name
    gm_name = da.attrs.get("grid_mapping", None)
    if gm_name is None and "spatial_ref" in ds.variables:
        gm_name = "spatial_ref"

    crs_wkt = None
    geotransform = None

    if gm_name is not None and gm_name in ds.variables:
        gm = ds[gm_name]

        # CRS WKT could be in different keys depending on writer
        for k in ("spatial_ref", "crs_wkt", "WKT", "proj_wkt", "esri_pe_string"):
            v = gm.attrs.get(k, None)
            if isinstance(v, str) and v.strip():
                crs_wkt = v
                break

        # GDAL often stores GeoTransform as a string of 6 numbers
        gt = gm.attrs.get("GeoTransform", None) or gm.attrs.get("geotransform", None)
        if isinstance(gt, str):
            parts = [p for p in gt.replace(",", " ").split() if p]
            if len(parts) == 6:
                geotransform = [float(p) for p in parts]
        elif isinstance(gt, (list, tuple)) and len(gt) == 6:
            geotransform = [float(p) for p in gt]

    # Fallback CRS from attrs if needed
    if crs_wkt is None:
        crs_wkt = da.attrs.get("crs_wkt", None) or da.attrs.get("spatial_ref", None)

    # Build Affine transform
    tf = None
    if geotransform is not None:
        # GDAL GT: [c, a, b, f, d, e]  (top-left x, w-e px, rot, top-left y, rot, n-s px)
        c, a, b, f, d, e = geotransform
        tf = Affine(a, b, c, d, e, f)
    else:
        # From x/y coords (pixel centers)
        if "x" in da.coords and "y" in da.coords:
            tf = affine_from_xy_centers(da["x"].values, da["y"].values)

    return crs_wkt, tf


def superresolve_single_time(
    da,
    crs_wkt,
    transform,
    indices,
    model,
    device,
    bands_to_use,
    old_res=10.0,
    new_res=2.5,
):
    da = da.sel(band=bands_to_use).rio.set_spatial_dims("x", "y", inplace=False)

    orig_band_order = da.band.values
    orig_attrs = dict(da.attrs)
    orig_attrs.pop("transform", None)
    orig_attrs.pop("grid_mapping", None)

    time_coord = da.coords.get("time", None)

    new_order = ["red", "green", "blue", "nir"]
    da_reordered = da.sel(band=new_order)

    ny, nx = da_reordered.sizes["y"], da_reordered.sizes["x"]
    new_side = math.ceil(max(ny, nx) / 128) * 128
    pad_y = new_side - ny
    pad_x = new_side - nx
    pad_dict = {"y": (pad_y // 2, pad_y - pad_y // 2), "x": (pad_x // 2, pad_x - pad_x // 2)}
    da_square = da_reordered.pad(pad_dict, constant_values=0)

    X = torch.from_numpy(da_square.compute().to_numpy().astype("float32")).float().to(device)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    with suppress_tqdm():
        superX = sen2sr.predict_large(model=model, X=X, overlap=32)

    scale = int(round(old_res / new_res))
    pad_y_top_hr = pad_dict["y"][0] * scale
    pad_x_left_hr = pad_dict["x"][0] * scale

    # transform for padded HR
    scaled_tf = transform * Affine.scale(1 / scale, 1 / scale)
    full_tf = scaled_tf * Affine.translation(-pad_x_left_hr, -pad_y_top_hr)

    arr = superX.cpu().numpy()
    var_name = da.name or "Spectral_Temporal_Stack"

    da_hr = xr.DataArray(
        arr,
        dims=("band", "y", "x"),
        coords={"band": new_order, "y": np.arange(arr.shape[1]), "x": np.arange(arr.shape[2])},
        name=var_name,
    )
    da_hr.attrs = orig_attrs
    if time_coord is not None:
        da_hr = da_hr.assign_coords(time=time_coord)

    ds_tmp = (
        da_hr.to_dataset(name=var_name)
        .rio.set_spatial_dims("x", "y", inplace=False)
        .rio.write_crs(crs_wkt, inplace=False)
        .rio.write_transform(full_tf, inplace=False)
    )

    # crop to original HR extent
    orig_h = da.sizes["y"]
    orig_w = da.sizes["x"]
    y0, y1 = pad_y_top_hr, pad_y_top_hr + orig_h * scale
    x0, x1 = pad_x_left_hr, pad_x_left_hr + orig_w * scale
    ds_cropped = ds_tmp.isel(y=slice(y0, y1), x=slice(x0, x1))

    cropped_tf = full_tf * Affine.translation(x0, y0)
    ds_cropped = ds_cropped.rio.write_transform(cropped_tf, inplace=False)

    ds_cropped = ds_cropped.sel(band=orig_band_order)

    # assign world x/y centers from cropped_tf
    W = ds_cropped.sizes["x"]
    H = ds_cropped.sizes["y"]
    xs = cropped_tf.c + cropped_tf.a * (np.arange(W) + 0.5)
    ys = cropped_tf.f + cropped_tf.e * (np.arange(H) + 0.5)
    ds_cropped = ds_cropped.assign_coords(x=("x", xs.astype(np.float64)), y=("y", ys.astype(np.float64)))

    da_super = ds_cropped[var_name].rio.set_spatial_dims("x", "y", inplace=False)
    da_super = da_super.rio.write_crs(crs_wkt, inplace=False).rio.write_transform(cropped_tf, inplace=False)
    da_super.attrs["status"] = "super-resolved"

    if indices:
        stac_indices = calculate_spectral_index(da_super, mission="s2", indices=indices)
        da_super = xr.concat([da_super, stac_indices], dim="band")
        da_super = da_super.rio.set_spatial_dims("x", "y", inplace=False)
        da_super = da_super.rio.write_crs(crs_wkt, inplace=False).rio.write_transform(cropped_tf, inplace=False)
        da_super.attrs.update(orig_attrs)
        da_super.attrs["indices"] = indices
        da_super.attrs["status"] = "super-resolved"

    return da_super


def super_resolve_cube(input_path, output_path: str | None = None, var_name="Spectral_Temporal_Stack"):
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

    if isinstance(input_path, xr.DataArray):
        raise ValueError("This version expects a NetCDF file path so it can read CF georef from metadata.")

    ds_in = xr.open_dataset(input_path)
    dataarray = ds_in[var_name]

    crs_wkt, tf = _extract_cf_crs_and_geotransform(ds_in, var_name)
    if crs_wkt is None:
        raise ValueError("Could not extract CRS WKT from CF grid mapping (spatial_ref).")
    if tf is None:
        raise ValueError("Could not extract/build transform from CF metadata or x/y coords.")

    indices = dataarray.attrs.get("indices", None)
    dataarray_sub = dataarray.sel(band=bands_to_use)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlstac.load(model_path).compiled_model(device=device)

    if "time" in dataarray_sub.dims:
        super_list = []
        for t in tqdm(dataarray_sub.time.values, desc="Super-resolving time steps", unit="date", file=sys.stdout, dynamic_ncols=False):
            da_t = dataarray_sub.sel(time=t)
            da_sr_t = superresolve_single_time(
                da=da_t,
                crs_wkt=crs_wkt,
                transform=tf,
                indices=indices,
                model=model,
                device=device,
                bands_to_use=bands_to_use,
                old_res=old_res,
                new_res=new_res,
            )
            super_list.append(da_sr_t)

        da_super_all = xr.concat(super_list, dim="time", coords="minimal", compat="override")

        # ADD THIS LINE (copies cloud_percentage etc. back onto the result)
        da_super_all = _copy_time_coords(dataarray, da_super_all)

        # Re-assert rio metadata after concat
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
            indices=indices,
            model=model,
            device=device,
            bands_to_use=bands_to_use,
            old_res=old_res,
            new_res=new_res,
        )

    da_super_all.name = var_name
    da_super_all.attrs["status"] = "super-resolved"

    # --- Write NetCDF (CRS must survive reload) ---
    ds_out = da_super_all.to_dataset(name=var_name)
    ds_out = ds_out.rio.set_spatial_dims("x", "y", inplace=False)
    ds_out = ds_out.rio.write_crs(crs_wkt, inplace=False)
    ds_out = ds_out.rio.write_transform(da_super_all.rio.transform(), inplace=False)

    # IMPORTANT:
    # 1) remove grid_mapping from attrs (xarray reserves it for CF encoding)
    ds_out[var_name].attrs.pop("grid_mapping", None)

    # 2) set grid_mapping in ENCODING (this is what xarray wants)
    ds_out[var_name].encoding["grid_mapping"] = "spatial_ref"

    # Optional: keep CRS as plain metadata too (as you requested)
    #ds_out.attrs["crs_wkt"] = crs_wkt
    #ds_out[var_name].attrs["crs_wkt"] = crs_wkt
    try:
        from rasterio.crs import CRS
        epsg = CRS.from_wkt(crs_wkt).to_epsg()
        if epsg is not None:
            crs_epsg = f"EPSG:{epsg}"
            ds_out.attrs["crs"] = crs_epsg
            ds_out[var_name].attrs["crs"] = crs_epsg
    except Exception:
        pass

    # keep GeoTransform on spatial_ref (QGIS-friendly)
    if "spatial_ref" in ds_out.variables:
        tf = ds_out.rio.transform()
        ds_out["spatial_ref"].attrs["GeoTransform"] = f"{tf.c} {tf.a} {tf.b} {tf.f} {tf.d} {tf.e}"

    # store transform as attribute too (your request)
    tf = ds_out.rio.transform()
    transform9 = [
        float(tf.a), float(tf.b), float(tf.c),
        float(tf.d), float(tf.e), float(tf.f),
        0.0, 0.0, 1.0,
    ]
    ds_out.attrs["transform"] = transform9
    ds_out[var_name].attrs["transform"] = transform9

    ds_out.to_netcdf(output_path)
    print("Data cube is super-resolved to 2.5-meters!")