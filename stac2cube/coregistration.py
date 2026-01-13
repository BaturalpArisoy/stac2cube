import os
import io
import numpy as np
import xarray as xr
import rioxarray
from arosics import COREG
from geoarray import GeoArray
from rasterio.transform import Affine
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


def _load_coreg_input(input_obj, stack_name="Spectral_Temporal_Stack"):
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
        ds = xr.open_dataset(input_obj)
        if stack_name not in ds:
            raise KeyError(f"Dataset has no variable '{stack_name}'. Found: {list(ds.data_vars)}")
        stack = ds[stack_name]
        input_path_str = input_obj

    elif isinstance(input_obj, xr.Dataset):
        ds = input_obj
        if stack_name not in ds:
            raise KeyError(f"Dataset has no variable '{stack_name}'. Found: {list(ds.data_vars)}")
        stack = ds[stack_name]
        input_path_str = None

    elif isinstance(input_obj, xr.DataArray):
        ds = None
        stack = input_obj
        input_path_str = None

    else:
        raise TypeError("input_path must be one of: str (netcdf path), xarray.Dataset, xarray.DataArray")

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
        if ds is not None and "spatial_ref" in ds.variables and hasattr(ds.spatial_ref, "GeoTransform"):
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
        if ds is not None and "spatial_ref" in ds.variables and hasattr(ds.spatial_ref, "crs_wkt"):
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


def _roi_to_geom_and_projected_bbox(roi, roi_crs="EPSG:4326", target_crs_wkt=None):
    """
    Returns (bbox_in_target_crs, target_crs_str_for_debug)

    bbox_in_target_crs: (xmin, ymin, xmax, ymax) in the stack CRS

    roi can be:
      - bbox list/tuple: [xmin, ymin, xmax, ymax] (assumed roi_crs)
      - gpkg path (str ending in .gpkg)
      - geojson geometry dict (has "type" and "coordinates")
    """
    from pyproj import CRS, Transformer
    from shapely.geometry import box, shape
    import pathlib

    if target_crs_wkt is None:
        raise ValueError("target_crs_wkt is required to project ROI into stack CRS.")

    target_crs = CRS.from_wkt(target_crs_wkt)
    src_crs = CRS.from_user_input(roi_crs)

    if isinstance(roi, (list, tuple)) and len(roi) == 4:
        xmin, ymin, xmax, ymax = map(float, roi)
        geom = box(xmin, ymin, xmax, ymax)

    elif isinstance(roi, dict) and "type" in roi and "coordinates" in roi:
        geom = shape(roi)

    elif isinstance(roi, str) and pathlib.Path(roi).suffix.lower() == ".gpkg":
        import geopandas as gpd
        gdf = gpd.read_file(roi)
        if gdf.empty:
            raise ValueError("GPKG ROI is empty.")
        if gdf.crs is None:
            raise ValueError("GPKG has no CRS. Please assign one before using it as ROI.")
        geom = gdf.geometry.unary_union
        src_crs = CRS.from_user_input(gdf.crs)

    else:
        raise TypeError(
            "roi must be one of: bbox [xmin,ymin,xmax,ymax], geojson geometry dict, or .gpkg path"
        )

    if src_crs == target_crs:
        xmin, ymin, xmax, ymax = geom.bounds
        return (float(xmin), float(ymin), float(xmax), float(ymax)), str(target_crs)

    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)

    xmin, ymin, xmax, ymax = geom.bounds
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    xs, ys = [], []
    for x, y in corners:
        X, Y = transformer.transform(x, y)
        xs.append(X)
        ys.append(Y)

    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))), str(target_crs)


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
# Main function (sliding-grid)
# ----------------------------------------------------------------------
def coregister_cube(
    input_path,  # str | xr.Dataset | xr.DataArray
    output_path=None,
    stack_name="Spectral_Temporal_Stack",
    first_scene_mode="composite",
    composite_window_days=30,
    grid_size=3,
    min_reliability_keep=10.0,
    min_reliability_update_ref=50.0,
    max_cloud_update_ref=20.0,
    max_cc=None,
    time_period=None,
):
    stac, masked_stac, cloud_pct_da, input_path_str = _load_coreg_input(input_path, stack_name=stack_name)

    # replace hard-coded test filters
    filtered_data = _apply_time_and_cloud_filters(masked_stac, max_cc=max_cc, time_period=time_period)

    # geo
    crs_wkt = _get_crs_wkt(filtered_data, ds=stac)
    filtered_data = filtered_data.rio.write_crs(crs_wkt, inplace=True)
    geotransform = _get_geotransform(filtered_data, ds=stac)

    times = filtered_data.time.values
    if times.size == 0:
        raise ValueError("No scenes left after applying max_cc/time_period filters.")
    band_names = filtered_data.band.values
    height = filtered_data.sizes["y"]
    width = filtered_data.sizes["x"]

    corrected_images, failed_times = [], []
    current_reference, master_geoArr = None, None
    kept_reliabilities, kept_rel_times = [], []

    if first_scene_mode == "first":
        im_ref = filtered_data.sel(time=times[0]).transpose("y", "x", "band")
        im_ref = im_ref.where(im_ref != 0, np.nan)
        y_coords, x_coords = _compute_coords(geotransform, height, width)
        im_ref = im_ref.assign_coords({"y": ("y", y_coords), "x": ("x", x_coords), "time": times[0]})
        corrected_images.append(im_ref)
        current_reference = im_ref
        start_idx = 1

    elif first_scene_mode == "composite":
        first_time = times[0]
        end_time = first_time + np.timedelta64(composite_window_days, "D")
        subset = filtered_data.sel(time=slice(first_time, end_time))
        if subset.sizes["time"] == 0:
            subset = filtered_data
        master_median = subset.median(dim="time", skipna=True)
        master_ref = master_median.transpose("y", "x", "band").where(master_median.transpose("y", "x", "band") != 0, np.nan)
        master_geoArr = GeoArray(master_ref.values, geotransform=geotransform, projection=crs_wkt)
        start_idx = 0
    else:
        raise ValueError("first_scene_mode must be 'first' or 'composite'")

    indices = list(range(start_idx, len(times)))

    for idx in tqdm(indices, total=len(indices), desc="Co-registering scenes", unit="scene"):
        t = times[idx]
        im_target = filtered_data.sel(time=t).transpose("y", "x", "band")

        if first_scene_mode == "composite" and current_reference is None:
            ref_geoArr = master_geoArr
        else:
            if current_reference is None:
                raise RuntimeError("No valid reference available for chained mode.")
            ref_geoArr = GeoArray(current_reference.values, geotransform=geotransform, projection=crs_wkt)

        tgt_geoArr = GeoArray(im_target.values, geotransform=geotransform, projection=crs_wkt)

        # sliding-grid candidates
        height_target, width_target, _ = im_target.shape
        left, bottom, right, top = _get_bounds_from_gt(geotransform, height_target, width_target)

        margin = 1.0 / (grid_size + 1)
        frac_vals = np.linspace(margin, 1.0 - margin, grid_size)

        manual_wps = []
        for iy, fy in enumerate(frac_vals):
            for ix, fx in enumerate(frac_vals):
                x_wp = left + fx * (right - left)
                y_wp = bottom + fy * (top - bottom)
                manual_wps.append((f"g{grid_size}x{grid_size}_r{iy}_c{ix}", (x_wp, y_wp)))

        candidates = [("auto", None)] + manual_wps
        successful_matches = []

        with _suppress_arosics_warnings():
            for label, wp in candidates:
                try:
                    if wp is None:
                        CR_try = COREG(ref_geoArr, tgt_geoArr, align_grids=True, q=True)
                    else:
                        CR_try = COREG(ref_geoArr, tgt_geoArr, align_grids=True, q=True, wp=wp)

                    CR_try.calculate_spatial_shifts()
                    result_try = CR_try.correct_shifts()
                    reliability_try = getattr(CR_try, "shift_reliability", None)
                    successful_matches.append({"label": label, "CR": CR_try, "result": result_try, "reliability": reliability_try})

                except (RuntimeError, ValueError, AssertionError, AttributeError):
                    continue

        if not successful_matches:
            failed_times.append(t)
            continue

        best_match = max(
            successful_matches,
            key=lambda m: -np.inf if m["reliability"] is None else float(m["reliability"]),
        )

        CR = best_match["CR"]
        result = best_match["result"]
        reliability = best_match["reliability"]

        if (min_reliability_keep is not None) and ((reliability is None) or (reliability < min_reliability_keep)):
            failed_times.append(t)
            continue

        out_geoArr = GeoArray(result["arr_shifted"], result["updated geotransform"], result["updated projection"])
        arr_corr = out_geoArr[:].transpose(2, 0, 1)
        arr_corr = np.where(arr_corr == 0, np.nan, arr_corr)

        updated_gt = result["updated geotransform"]
        h2, w2 = arr_corr.shape[1], arr_corr.shape[2]
        y2, x2 = _compute_coords(updated_gt, h2, w2)

        da_corr = xr.DataArray(
            arr_corr,
            dims=("band", "y", "x"),
            coords={"band": range(1, arr_corr.shape[0] + 1), "y": ("y", y2), "x": ("x", x2)},
        )
        da_corr = da_corr.rio.write_transform(Affine.from_gdal(*updated_gt))
        da_corr = da_corr.rio.write_crs(result["updated projection"])
        da_corr = da_corr.assign_coords(band=("band", band_names)).transpose("y", "x", "band").assign_coords(time=t)

        corrected_images.append(da_corr)

        if reliability is not None:
            kept_reliabilities.append(float(reliability))
            kept_rel_times.append(t)

        # reference update rules
        cp_t = None
        if cloud_pct_da is not None:
            try:
                cp_t = float(cloud_pct_da.sel(time=t)) if "time" in cloud_pct_da.dims else float(cloud_pct_da)
            except Exception:
                cp_t = None

        update_ref = True
        if (min_reliability_update_ref is not None) and ((reliability is None) or (reliability < min_reliability_update_ref)):
            update_ref = False
        if (max_cloud_update_ref is not None) and (cp_t is not None) and (cp_t > max_cloud_update_ref):
            update_ref = False

        if update_ref:
            current_reference = da_corr

    if not corrected_images:
        raise RuntimeError("No scenes were kept. Output stack would be empty.")

    corrected_stack = xr.concat(corrected_images, dim="time").transpose("time", "band", "y", "x")
    corrected_stack = corrected_stack.rio.write_crs(crs_wkt, inplace=True)
    corrected_stack.name = "Spectral_Temporal_Stack"

    out_ds = xr.Dataset({"Spectral_Temporal_Stack": corrected_stack})
    if stac is not None and "spatial_ref" in stac.variables:
        out_ds["spatial_ref"] = stac["spatial_ref"]

    if stac is not None:
        if "cloud_percentage" in stac.coords:
            out_ds = out_ds.assign_coords(cloud_percentage=stac.cloud_percentage)
        elif "cloud_percentage" in stac:
            out_ds = out_ds.assign_coords(cloud_percentage=stac["cloud_percentage"])

    # report
    times_out = corrected_stack.time.values
    print("\nCo-registration summary")
    print("-----------------------")
    print(f"Original (after max_cc/time_period): {len(times)} scenes from {np.datetime_as_string(times[0], 'D')} to {np.datetime_as_string(times[-1], 'D')}")
    print("Scenes excluded after co-registration (overlap / tie points / low reliability):", len(failed_times))
    print(f"Scenes remaining in the co-registered cube: {len(times_out)}")

    if failed_times:
        excluded_entries = []
        for ts in failed_times:
            ds_ = np.datetime_as_string(ts, unit="D")
            if cloud_pct_da is not None:
                try:
                    cp = float(cloud_pct_da.sel(time=ts)) if "time" in cloud_pct_da.dims else float(cloud_pct_da)
                    excluded_entries.append(f"{ds_} ({cp:.1f}%)")
                except Exception:
                    excluded_entries.append(ds_)
            else:
                excluded_entries.append(ds_)
        print("Excluded dates (cloud percentage): " + ", ".join(excluded_entries))

    if kept_reliabilities:
        mean_rel = float(np.mean(kept_reliabilities))
        min_idx = int(np.argmin(kept_reliabilities))
        print(f"\nMean match reliability of kept scenes: {mean_rel:.1f} %")
        print(
            f"Minimum match reliability of kept scenes: {kept_reliabilities[min_idx]:.1f} % "
            f"(date: {np.datetime_as_string(kept_rel_times[min_idx], 'D')})"
        )

    print("\nS2 co-registration is completed!")

    # export
    if output_path is None and input_path_str is not None:
        output_path = _auto_output_path(input_path_str, suffix="_cr")

    if output_path is not None:
        out_ds.to_netcdf(output_path)
        print(f"\nCo-registered cube written to: {output_path}")
    else:
        print("\nNo output_path provided and input was not a file path -> skipping NetCDF export.")

    return out_ds, output_path


# ----------------------------------------------------------------------
# ROI-based co-registration (no sliding windows)
# ----------------------------------------------------------------------
def coregister_cube_roi(
    input_path,   # str | xr.Dataset | xr.DataArray
    roi,          # bbox [xmin,ymin,xmax,ymax] OR geojson geom dict OR .gpkg path
    roi_crs="EPSG:4326",
    output_path=None,
    stack_name="Spectral_Temporal_Stack",
    first_scene_mode="composite",
    composite_window_days=30,
    min_reliability_keep=10.0,
    min_reliability_update_ref=50.0,
    max_cloud_update_ref=20.0,
    roi_ws_min_px=64,
    roi_ws_max_px=2048,
    # NEW:
    max_cc=None,
    time_period=None,
):
    """
    Same chaining logic, BUT matching is forced into a user ROI by using:
      - wp = ROI centroid (map coords)
      - ws = ROI bbox size converted to pixels (bounded by roi_ws_min_px/max_px)
    """
    stac, masked_stac, cloud_pct_da, input_path_str = _load_coreg_input(input_path, stack_name=stack_name)

    # replace hard-coded test filters
    filtered_data = _apply_time_and_cloud_filters(masked_stac, max_cc=max_cc, time_period=time_period)

    # geo
    crs_wkt = _get_crs_wkt(filtered_data, ds=stac)
    filtered_data = filtered_data.rio.write_crs(crs_wkt, inplace=True)
    geotransform = _get_geotransform(filtered_data, ds=stac)

    times = filtered_data.time.values
    if times.size == 0:
        raise ValueError("No scenes left after applying max_cc/time_period filters.")
    band_names = filtered_data.band.values
    height = filtered_data.sizes["y"]
    width = filtered_data.sizes["x"]

    # ROI bbox in stack CRS
    (rxmin, rymin, rxmax, rymax), _ = _roi_to_geom_and_projected_bbox(roi, roi_crs=roi_crs, target_crs_wkt=crs_wkt)

    wp = ((rxmin + rxmax) / 2.0, (rymin + rymax) / 2.0)

    px_w = float(geotransform[1])
    px_h = float(abs(geotransform[5]))
    wsx = int(max(roi_ws_min_px, min(roi_ws_max_px, (rxmax - rxmin) / max(px_w, 1e-12))))
    wsy = int(max(roi_ws_min_px, min(roi_ws_max_px, (rymax - rymin) / max(px_h, 1e-12))))
    ws = (wsx, wsy)

    corrected_images, failed_times = [], []
    current_reference, master_geoArr = None, None
    kept_reliabilities, kept_rel_times = [], []

    if first_scene_mode == "first":
        im_ref = filtered_data.sel(time=times[0]).transpose("y", "x", "band")
        im_ref = im_ref.where(im_ref != 0, np.nan)
        y_coords, x_coords = _compute_coords(geotransform, height, width)
        im_ref = im_ref.assign_coords({"y": ("y", y_coords), "x": ("x", x_coords), "time": times[0]})
        corrected_images.append(im_ref)
        current_reference = im_ref
        start_idx = 1

    elif first_scene_mode == "composite":
        first_time = times[0]
        end_time = first_time + np.timedelta64(composite_window_days, "D")
        subset = filtered_data.sel(time=slice(first_time, end_time))
        if subset.sizes["time"] == 0:
            subset = filtered_data
        master_median = subset.median(dim="time", skipna=True)
        master_ref = master_median.transpose("y", "x", "band").where(master_median.transpose("y", "x", "band") != 0, np.nan)
        master_geoArr = GeoArray(master_ref.values, geotransform=geotransform, projection=crs_wkt)
        start_idx = 0
    else:
        raise ValueError("first_scene_mode must be 'first' or 'composite'")

    indices = list(range(start_idx, len(times)))

    for idx in tqdm(indices, total=len(indices), desc="Co-registering scenes (ROI)", unit="scene"):
        t = times[idx]
        im_target = filtered_data.sel(time=t).transpose("y", "x", "band")

        if first_scene_mode == "composite" and current_reference is None:
            ref_geoArr = master_geoArr
        else:
            if current_reference is None:
                raise RuntimeError("No valid reference available for chained mode.")
            ref_geoArr = GeoArray(current_reference.values, geotransform=geotransform, projection=crs_wkt)

        tgt_geoArr = GeoArray(im_target.values, geotransform=geotransform, projection=crs_wkt)

        with _suppress_arosics_warnings():
            try:
                CR = COREG(ref_geoArr, tgt_geoArr, align_grids=True, q=True, wp=wp, ws=ws)
                CR.calculate_spatial_shifts()
                result = CR.correct_shifts()
                reliability = getattr(CR, "shift_reliability", None)
            except (RuntimeError, ValueError, AssertionError, AttributeError):
                failed_times.append(t)
                continue

        if (min_reliability_keep is not None) and ((reliability is None) or (reliability < min_reliability_keep)):
            failed_times.append(t)
            continue

        out_geoArr = GeoArray(result["arr_shifted"], result["updated geotransform"], result["updated projection"])
        arr_corr = out_geoArr[:].transpose(2, 0, 1)
        arr_corr = np.where(arr_corr == 0, np.nan, arr_corr)

        updated_gt = result["updated geotransform"]
        h2, w2 = arr_corr.shape[1], arr_corr.shape[2]
        y2, x2 = _compute_coords(updated_gt, h2, w2)

        da_corr = xr.DataArray(
            arr_corr,
            dims=("band", "y", "x"),
            coords={"band": range(1, arr_corr.shape[0] + 1), "y": ("y", y2), "x": ("x", x2)},
        )
        da_corr = da_corr.rio.write_transform(Affine.from_gdal(*updated_gt))
        da_corr = da_corr.rio.write_crs(result["updated projection"])
        da_corr = da_corr.assign_coords(band=("band", band_names)).transpose("y", "x", "band").assign_coords(time=t)

        corrected_images.append(da_corr)

        if reliability is not None:
            kept_reliabilities.append(float(reliability))
            kept_rel_times.append(t)

        cp_t = None
        if cloud_pct_da is not None:
            try:
                cp_t = float(cloud_pct_da.sel(time=t)) if "time" in cloud_pct_da.dims else float(cloud_pct_da)
            except Exception:
                cp_t = None

        update_ref = True
        if (min_reliability_update_ref is not None) and ((reliability is None) or (reliability < min_reliability_update_ref)):
            update_ref = False
        if (max_cloud_update_ref is not None) and (cp_t is not None) and (cp_t > max_cloud_update_ref):
            update_ref = False

        if update_ref:
            current_reference = da_corr

    if not corrected_images:
        raise RuntimeError("No scenes were kept. Output stack would be empty.")

    corrected_stack = xr.concat(corrected_images, dim="time").transpose("time", "band", "y", "x")
    corrected_stack = corrected_stack.rio.write_crs(crs_wkt, inplace=True)
    corrected_stack.name = "Spectral_Temporal_Stack"

    out_ds = xr.Dataset({"Spectral_Temporal_Stack": corrected_stack})
    if stac is not None and "spatial_ref" in stac.variables:
        out_ds["spatial_ref"] = stac["spatial_ref"]

    if stac is not None:
        if "cloud_percentage" in stac.coords:
            out_ds = out_ds.assign_coords(cloud_percentage=stac.cloud_percentage)
        elif "cloud_percentage" in stac:
            out_ds = out_ds.assign_coords(cloud_percentage=stac["cloud_percentage"])

    # report
    times_out = corrected_stack.time.values
    print("\nCo-registration summary (ROI)")
    print("-----------------------------")
    print(f"ROI wp={wp}, ws(px)={ws}")
    print(f"Original (after max_cc/time_period): {len(times)} scenes from {np.datetime_as_string(times[0], 'D')} to {np.datetime_as_string(times[-1], 'D')}")
    print("Scenes excluded after co-registration (overlap / tie points / low reliability):", len(failed_times))
    print(f"Scenes remaining in the co-registered cube: {len(times_out)}")

    if failed_times:
        excluded_entries = []
        for ts in failed_times:
            ds_ = np.datetime_as_string(ts, unit="D")
            if cloud_pct_da is not None:
                try:
                    cp = float(cloud_pct_da.sel(time=ts)) if "time" in cloud_pct_da.dims else float(cloud_pct_da)
                    excluded_entries.append(f"{ds_} ({cp:.1f}%)")
                except Exception:
                    excluded_entries.append(ds_)
            else:
                excluded_entries.append(ds_)
        print("Excluded dates (cloud percentage): " + ", ".join(excluded_entries))

    if kept_reliabilities:
        mean_rel = float(np.mean(kept_reliabilities))
        min_idx = int(np.argmin(kept_reliabilities))
        print(f"\nMean match reliability of kept scenes: {mean_rel:.1f} %")
        print(
            f"Minimum match reliability of kept scenes: {kept_reliabilities[min_idx]:.1f} % "
            f"(date: {np.datetime_as_string(kept_rel_times[min_idx], 'D')})"
        )

    print("\nS2 co-registration is completed!")

    # export
    if output_path is None and input_path_str is not None:
        output_path = _auto_output_path(input_path_str, suffix="_cr")

    if output_path is not None:
        out_ds.to_netcdf(output_path)
        print(f"\nCo-registered cube written to: {output_path}")
    else:
        print("\nNo output_path provided and input was not a file path -> skipping NetCDF export.")

    return out_ds
