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
            raise KeyError(
                f"Dataset has no variable '{stack_name}'. Found: {list(ds.data_vars)}"
            )
        stack = ds[stack_name]
        input_path_str = input_obj

    elif isinstance(input_obj, xr.Dataset):
        ds = input_obj
        if stack_name not in ds:
            raise KeyError(
                f"Dataset has no variable '{stack_name}'. Found: {list(ds.data_vars)}"
            )
        stack = ds[stack_name]
        input_path_str = None

    elif isinstance(input_obj, xr.DataArray):
        ds = None
        stack = input_obj
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
            raise ValueError(
                "GPKG has no CRS. Please assign one before using it as ROI."
            )
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

    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))), str(
        target_crs
    )


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
    # NEW:
    iteration=1,
):
    # -----------------------------
    # NEW: validate iteration
    # -----------------------------
    if isinstance(iteration, bool) or not isinstance(iteration, (int, np.integer)):
        raise TypeError("iteration must be an integer >= 1.")
    iteration = int(iteration)
    if iteration < 1:
        raise ValueError("iteration must be an integer >= 1 (cannot be 0).")

    # Keep original input path string for auto-export on final iteration
    _orig_input_path_str = input_path if isinstance(input_path, str) else None

    def _run_once(
        _input_obj,
        _output_path,
        _first_scene_mode,
        do_export=True,
    ):
        stac, masked_stac, cloud_pct_da, input_path_str = _load_coreg_input(
            _input_obj, stack_name=stack_name
        )
        input_crs_attr = masked_stac.attrs.get("crs", None)
        # replace hard-coded test filters
        filtered_data = _apply_time_and_cloud_filters(
            masked_stac, max_cc=max_cc, time_period=time_period
        )

        # geo
        crs_wkt = _get_crs_wkt(filtered_data, ds=stac)
        filtered_data = filtered_data.rio.write_crs(crs_wkt, inplace=True)
        geotransform = _get_geotransform(filtered_data, ds=stac)

        times = filtered_data.time.values
        if times.size == 0:
            raise ValueError(
                "No scenes left after applying max_cc/time_period filters."
            )
        band_names = filtered_data.band.values
        height = filtered_data.sizes["y"]
        width = filtered_data.sizes["x"]

        corrected_images, failed_times = [], []
        current_reference, master_geoArr = None, None
        kept_reliabilities, kept_rel_times = [], []

        if _first_scene_mode == "first":
            im_ref = filtered_data.sel(time=times[0]).transpose("y", "x", "band")
            im_ref = im_ref.where(im_ref != 0, np.nan)
            y_coords, x_coords = _compute_coords(geotransform, height, width)
            im_ref = im_ref.assign_coords(
                {"y": ("y", y_coords), "x": ("x", x_coords), "time": times[0]}
            )
            corrected_images.append(im_ref)
            current_reference = im_ref
            start_idx = 1

        elif _first_scene_mode == "composite":
            first_time = times[0]
            end_time = first_time + np.timedelta64(composite_window_days, "D")
            subset = filtered_data.sel(time=slice(first_time, end_time))
            if subset.sizes["time"] == 0:
                subset = filtered_data
            master_median = subset.median(dim="time", skipna=True)
            master_ref = master_median.transpose("y", "x", "band").where(
                master_median.transpose("y", "x", "band") != 0, np.nan
            )
            master_geoArr = GeoArray(
                master_ref.values, geotransform=geotransform, projection=crs_wkt
            )
            start_idx = 0
        else:
            raise ValueError("first_scene_mode must be 'first' or 'composite'")

        indices = range(start_idx, len(times))

        for idx in tqdm(
            indices,
            total=len(times),  # show full count (e.g., 1/23 in "first" mode)
            initial=start_idx,
            desc="Co-registering scenes",
            unit="scene",
        ):
            t = times[idx]
            im_target = filtered_data.sel(time=t).transpose("y", "x", "band")

            if _first_scene_mode == "composite" and current_reference is None:
                ref_geoArr = master_geoArr
            else:
                if current_reference is None:
                    raise RuntimeError("No valid reference available for chained mode.")
                ref_geoArr = GeoArray(
                    current_reference.values,
                    geotransform=geotransform,
                    projection=crs_wkt,
                )

            tgt_geoArr = GeoArray(
                im_target.values, geotransform=geotransform, projection=crs_wkt
            )

            # sliding-grid candidates
            height_target, width_target, _ = im_target.shape
            left, bottom, right, top = _get_bounds_from_gt(
                geotransform, height_target, width_target
            )

            margin = 1.0 / (grid_size + 1)
            frac_vals = np.linspace(margin, 1.0 - margin, grid_size)

            manual_wps = []
            for iy, fy in enumerate(frac_vals):
                for ix, fx in enumerate(frac_vals):
                    x_wp = left + fx * (right - left)
                    y_wp = bottom + fy * (top - bottom)
                    manual_wps.append(
                        (f"g{grid_size}x{grid_size}_r{iy}_c{ix}", (x_wp, y_wp))
                    )

            candidates = [("auto", None)] + manual_wps
            successful_matches = []

            with _suppress_arosics_warnings():
                for label, wp in candidates:
                    try:
                        if wp is None:
                            CR_try = COREG(
                                ref_geoArr, tgt_geoArr, align_grids=True, q=True
                            )
                        else:
                            CR_try = COREG(
                                ref_geoArr, tgt_geoArr, align_grids=True, q=True, wp=wp
                            )

                        CR_try.calculate_spatial_shifts()
                        result_try = CR_try.correct_shifts()
                        reliability_try = getattr(CR_try, "shift_reliability", None)
                        successful_matches.append(
                            {
                                "label": label,
                                "CR": CR_try,
                                "result": result_try,
                                "reliability": reliability_try,
                            }
                        )

                    except (RuntimeError, ValueError, AssertionError, AttributeError):
                        continue

            if not successful_matches:
                failed_times.append(t)
                continue

            best_match = max(
                successful_matches,
                key=lambda m: (
                    -np.inf if m["reliability"] is None else float(m["reliability"])
                ),
            )

            CR = best_match["CR"]
            result = best_match["result"]
            reliability = best_match["reliability"]

            if (min_reliability_keep is not None) and (
                (reliability is None) or (reliability < min_reliability_keep)
            ):
                failed_times.append(t)
                continue

            out_geoArr = GeoArray(
                result["arr_shifted"],
                result["updated geotransform"],
                result["updated projection"],
            )
            arr_corr = out_geoArr[:].transpose(2, 0, 1)
            arr_corr = np.where(arr_corr == 0, np.nan, arr_corr)

            updated_gt = result["updated geotransform"]
            h2, w2 = arr_corr.shape[1], arr_corr.shape[2]
            y2, x2 = _compute_coords(updated_gt, h2, w2)

            da_corr = xr.DataArray(
                arr_corr,
                dims=("band", "y", "x"),
                coords={
                    "band": range(1, arr_corr.shape[0] + 1),
                    "y": ("y", y2),
                    "x": ("x", x2),
                },
            )
            da_corr = da_corr.rio.write_transform(Affine.from_gdal(*updated_gt))
            da_corr = da_corr.rio.write_crs(result["updated projection"])
            da_corr = (
                da_corr.assign_coords(band=("band", band_names))
                .transpose("y", "x", "band")
                .assign_coords(time=t)
            )

            corrected_images.append(da_corr)

            if reliability is not None:
                kept_reliabilities.append(float(reliability))
                kept_rel_times.append(t)

            # reference update rules
            cp_t = None
            if cloud_pct_da is not None:
                try:
                    cp_t = (
                        float(cloud_pct_da.sel(time=t))
                        if "time" in cloud_pct_da.dims
                        else float(cloud_pct_da)
                    )
                except Exception:
                    cp_t = None

            update_ref = True
            if (min_reliability_update_ref is not None) and (
                (reliability is None) or (reliability < min_reliability_update_ref)
            ):
                update_ref = False
            if (
                (max_cloud_update_ref is not None)
                and (cp_t is not None)
                and (cp_t > max_cloud_update_ref)
            ):
                update_ref = False

            if update_ref:
                current_reference = da_corr

        if not corrected_images:
            raise RuntimeError("No scenes were kept. Output stack would be empty.")

        corrected_stack = xr.concat(corrected_images, dim="time").transpose(
            "time", "band", "y", "x"
        )
        corrected_stack = corrected_stack.rio.write_crs(crs_wkt, inplace=True)
        if input_crs_attr is not None:
            corrected_stack.attrs["crs"] = input_crs_attr
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
        print(
            f"Original (after max_cc/time_period): {len(times)} scenes from "
            f"{np.datetime_as_string(times[0], 'D')} to {np.datetime_as_string(times[-1], 'D')}"
        )
        print(
            "Scenes excluded after co-registration (overlap / tie points / low reliability):",
            len(failed_times),
        )
        print(f"Scenes remaining in the co-registered cube: {len(times_out)}")

        if failed_times:
            excluded_entries = []
            for ts in failed_times:
                ds_ = np.datetime_as_string(ts, unit="D")
                if cloud_pct_da is not None:
                    try:
                        cp = (
                            float(cloud_pct_da.sel(time=ts))
                            if "time" in cloud_pct_da.dims
                            else float(cloud_pct_da)
                        )
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

        # export (ONLY if requested by wrapper)
        final_out_path = _output_path
        if do_export:
            if final_out_path is None and input_path_str is not None:
                final_out_path = _auto_output_path(input_path_str, suffix="_cr")

            if final_out_path is not None:
                out_ds.to_netcdf(final_out_path)
                print(f"\nCo-registered cube written to: {final_out_path}")
            else:
                print(
                    "\nNo output_path provided and input was not a file path -> skipping NetCDF export."
                )

        return out_ds, final_out_path

    # -----------------------------
    # NEW: iteration wrapper
    # -----------------------------
    if iteration == 1:
        return _run_once(input_path, output_path, first_scene_mode, do_export=True)

    # Multi-iteration: no intermediate exports; export only at the end.
    # If user didn't specify output_path but the ORIGINAL input was a file path, keep the old auto-export behavior.
    final_target_path = output_path
    if final_target_path is None and _orig_input_path_str is not None:
        final_target_path = _auto_output_path(_orig_input_path_str, suffix="_cr")

    current_input = input_path
    current_mode = first_scene_mode

    out_ds_final, out_path_final = None, None

    for it in range(1, iteration + 1):
        is_last = it == iteration
        print(
            f"\n=== Iteration {it}/{iteration} (first_scene_mode='{current_mode}') ==="
        )

        out_ds_it, out_path_it = _run_once(
            current_input,
            final_target_path if is_last else None,
            current_mode,
            do_export=is_last,
        )

        current_input = out_ds_it  # feed next iteration with already co-registered cube
        out_ds_final, out_path_final = out_ds_it, out_path_it

        # composite only on iteration 1; then switch to 'first'
        if first_scene_mode == "composite" and it == 1:
            current_mode = "first"

    return out_ds_final, out_path_final


# ----------------------------------------------------------------------
# ROI-based co-registration (no sliding windows)
# ----------------------------------------------------------------------
def coregister_cube_roi(
    input_path,  # str | xr.Dataset | xr.DataArray
    roi,  # bbox [xmin,ymin,xmax,ymax] OR geojson geom dict OR .gpkg path
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
    max_cc=None,
    time_period=None,
    # NEW:
    iteration=1,
):
    if isinstance(iteration, bool) or not isinstance(iteration, (int, np.integer)):
        raise TypeError("iteration must be an integer >= 1.")
    iteration = int(iteration)
    if iteration < 1:
        raise ValueError("iteration must be an integer >= 1 (cannot be 0).")

    _orig_input_path_str = input_path if isinstance(input_path, str) else None

    def _run_once(
        _input_obj,
        _output_path,
        _first_scene_mode,
        do_export=True,
    ):
        stac, masked_stac, cloud_pct_da, input_path_str = _load_coreg_input(
            _input_obj, stack_name=stack_name
        )
        input_crs_attr = masked_stac.attrs.get("crs", None)
        filtered_data = _apply_time_and_cloud_filters(
            masked_stac, max_cc=max_cc, time_period=time_period
        )

        crs_wkt = _get_crs_wkt(filtered_data, ds=stac)
        filtered_data = filtered_data.rio.write_crs(crs_wkt, inplace=True)
        geotransform = _get_geotransform(filtered_data, ds=stac)

        times = filtered_data.time.values
        if times.size == 0:
            raise ValueError(
                "No scenes left after applying max_cc/time_period filters."
            )
        band_names = filtered_data.band.values
        height = filtered_data.sizes["y"]
        width = filtered_data.sizes["x"]

        (rxmin, rymin, rxmax, rymax), _ = _roi_to_geom_and_projected_bbox(
            roi, roi_crs=roi_crs, target_crs_wkt=crs_wkt
        )
        wp = ((rxmin + rxmax) / 2.0, (rymin + rymax) / 2.0)

        px_w = float(geotransform[1])
        px_h = float(abs(geotransform[5]))
        wsx = int(
            max(roi_ws_min_px, min(roi_ws_max_px, (rxmax - rxmin) / max(px_w, 1e-12)))
        )
        wsy = int(
            max(roi_ws_min_px, min(roi_ws_max_px, (rymax - rymin) / max(px_h, 1e-12)))
        )
        ws = (wsx, wsy)

        corrected_images, failed_times = [], []
        current_reference, master_geoArr = None, None
        kept_reliabilities, kept_rel_times = [], []

        if _first_scene_mode == "first":
            im_ref = filtered_data.sel(time=times[0]).transpose("y", "x", "band")
            im_ref = im_ref.where(im_ref != 0, np.nan)
            y_coords, x_coords = _compute_coords(geotransform, height, width)
            im_ref = im_ref.assign_coords(
                {"y": ("y", y_coords), "x": ("x", x_coords), "time": times[0]}
            )
            corrected_images.append(im_ref)
            current_reference = im_ref
            start_idx = 1

        elif _first_scene_mode == "composite":
            first_time = times[0]
            end_time = first_time + np.timedelta64(composite_window_days, "D")
            subset = filtered_data.sel(time=slice(first_time, end_time))
            if subset.sizes["time"] == 0:
                subset = filtered_data
            master_median = subset.median(dim="time", skipna=True)
            master_ref = master_median.transpose("y", "x", "band").where(
                master_median.transpose("y", "x", "band") != 0, np.nan
            )
            master_geoArr = GeoArray(
                master_ref.values, geotransform=geotransform, projection=crs_wkt
            )
            start_idx = 0
        else:
            raise ValueError("first_scene_mode must be 'first' or 'composite'")

        indices = range(start_idx, len(times))

        for idx in tqdm(
            indices,
            total=len(times),
            initial=start_idx,
            desc="Co-registering scenes (ROI)",
            unit="scene",
        ):
            t = times[idx]
            im_target = filtered_data.sel(time=t).transpose("y", "x", "band")

            if _first_scene_mode == "composite" and current_reference is None:
                ref_geoArr = master_geoArr
            else:
                if current_reference is None:
                    raise RuntimeError("No valid reference available for chained mode.")
                ref_geoArr = GeoArray(
                    current_reference.values,
                    geotransform=geotransform,
                    projection=crs_wkt,
                )

            tgt_geoArr = GeoArray(
                im_target.values, geotransform=geotransform, projection=crs_wkt
            )

            with _suppress_arosics_warnings():
                try:
                    CR = COREG(
                        ref_geoArr, tgt_geoArr, align_grids=True, q=True, wp=wp, ws=ws
                    )
                    CR.calculate_spatial_shifts()
                    result = CR.correct_shifts()
                    reliability = getattr(CR, "shift_reliability", None)
                except (RuntimeError, ValueError, AssertionError, AttributeError):
                    failed_times.append(t)
                    continue

            if (min_reliability_keep is not None) and (
                (reliability is None) or (reliability < min_reliability_keep)
            ):
                failed_times.append(t)
                continue

            out_geoArr = GeoArray(
                result["arr_shifted"],
                result["updated geotransform"],
                result["updated projection"],
            )
            arr_corr = out_geoArr[:].transpose(2, 0, 1)
            arr_corr = np.where(arr_corr == 0, np.nan, arr_corr)

            updated_gt = result["updated geotransform"]
            h2, w2 = arr_corr.shape[1], arr_corr.shape[2]
            y2, x2 = _compute_coords(updated_gt, h2, w2)

            da_corr = xr.DataArray(
                arr_corr,
                dims=("band", "y", "x"),
                coords={
                    "band": range(1, arr_corr.shape[0] + 1),
                    "y": ("y", y2),
                    "x": ("x", x2),
                },
            )
            da_corr = da_corr.rio.write_transform(Affine.from_gdal(*updated_gt))
            da_corr = da_corr.rio.write_crs(result["updated projection"])
            da_corr = (
                da_corr.assign_coords(band=("band", band_names))
                .transpose("y", "x", "band")
                .assign_coords(time=t)
            )

            corrected_images.append(da_corr)

            if reliability is not None:
                kept_reliabilities.append(float(reliability))
                kept_rel_times.append(t)

            cp_t = None
            if cloud_pct_da is not None:
                try:
                    cp_t = (
                        float(cloud_pct_da.sel(time=t))
                        if "time" in cloud_pct_da.dims
                        else float(cloud_pct_da)
                    )
                except Exception:
                    cp_t = None

            update_ref = True
            if (min_reliability_update_ref is not None) and (
                (reliability is None) or (reliability < min_reliability_update_ref)
            ):
                update_ref = False
            if (
                (max_cloud_update_ref is not None)
                and (cp_t is not None)
                and (cp_t > max_cloud_update_ref)
            ):
                update_ref = False

            if update_ref:
                current_reference = da_corr

        if not corrected_images:
            raise RuntimeError("No scenes were kept. Output stack would be empty.")

        corrected_stack = xr.concat(corrected_images, dim="time").transpose(
            "time", "band", "y", "x"
        )
        corrected_stack = corrected_stack.rio.write_crs(crs_wkt, inplace=True)
        if input_crs_attr is not None:
            corrected_stack.attrs["crs"] = input_crs_attr
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
        print(
            f"Original (after max_cc/time_period): {len(times)} scenes from "
            f"{np.datetime_as_string(times[0], 'D')} to {np.datetime_as_string(times[-1], 'D')}"
        )
        print(
            "Scenes excluded after co-registration (overlap / tie points / low reliability):",
            len(failed_times),
        )
        print(f"Scenes remaining in the co-registered cube: {len(times_out)}")

        if failed_times:
            excluded_entries = []
            for ts in failed_times:
                ds_ = np.datetime_as_string(ts, unit="D")
                if cloud_pct_da is not None:
                    try:
                        cp = (
                            float(cloud_pct_da.sel(time=ts))
                            if "time" in cloud_pct_da.dims
                            else float(cloud_pct_da)
                        )
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

        final_out_path = _output_path
        if do_export:
            if final_out_path is None and input_path_str is not None:
                final_out_path = _auto_output_path(input_path_str, suffix="_cr")

            if final_out_path is not None:
                out_ds.to_netcdf(final_out_path)
                print(f"\nCo-registered cube written to: {final_out_path}")
            else:
                print(
                    "\nNo output_path provided and input was not a file path -> skipping NetCDF export."
                )

        return out_ds, final_out_path

    if iteration == 1:
        return _run_once(input_path, output_path, first_scene_mode, do_export=True)

    final_target_path = output_path
    if final_target_path is None and _orig_input_path_str is not None:
        final_target_path = _auto_output_path(_orig_input_path_str, suffix="_cr")

    current_input = input_path
    current_mode = first_scene_mode
    out_ds_final, out_path_final = None, None

    for it in range(1, iteration + 1):
        is_last = it == iteration
        print(
            f"\n=== Iteration {it}/{iteration} (first_scene_mode='{current_mode}') ==="
        )

        out_ds_it, out_path_it = _run_once(
            current_input,
            final_target_path if is_last else None,
            current_mode,
            do_export=is_last,
        )

        current_input = out_ds_it
        out_ds_final, out_path_final = out_ds_it, out_path_it

        if first_scene_mode == "composite" and it == 1:
            current_mode = "first"

    return out_ds_final


# ----------------------------------------------------------------------
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go


def _load_stac(path, stack_name="Spectral_Temporal_Stack"):
    with xr.open_dataset(path) as ds:
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
    before_path, after_path, stack_name="Spectral_Temporal_Stack", rgb_time="first"
):
    stac_b = _load_stac(before_path, stack_name)
    stac_a = _load_stac(after_path, stack_name)

    ndvi_b = _band_label(stac_b, "ndvi")
    ndvi_a = _band_label(stac_a, "ndvi")

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
            text="NDVI Spectral Profile",
            x=0.5,
            xanchor="center",
            pad=dict(t=10, b=20),
        ),
        xaxis=dict(
            title="time",
            tickangle=-45,   # <-- incline labels so they don't overlap
            type="category", # <-- treat x as categorical strings, not numbers
        ),
        yaxis_title="NDVI",
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

        s_b = stac_b.sel(band=ndvi_b).sel(x=x0, y=y0, method="nearest")
        s_a = stac_a.sel(band=ndvi_a).sel(x=x0, y=y0, method="nearest")

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

    ui = widgets.VBox([widgets.HBox([fig_img, fig_ts]), out])
    display(ui)
    return ui
