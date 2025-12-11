# DOES NOT WORK IF THERE IS ONLY ONE BAND!
import os
import io
import sys
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
# Helper to suppress noisy AROSICS warnings
# ----------------------------------------------------------------------
@contextmanager
def _suppress_arosics_warnings():
    """
    Suppress Python warnings *and* any print/log output from arosics/geoarray
    during co-registration calls (e.g. 'Automatically detected nodata value ...').
    """
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            yield


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _compute_coords(gt, height, width):
    """
    Compute y/x coordinates for pixel centers from a GDAL-style geotransform.
    gt = (origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height)
    """
    origin_x, pixel_width, _, origin_y, _, pixel_height = gt
    x_coords = origin_x + pixel_width * (np.arange(width) + 0.5)
    y_coords = origin_y + pixel_height * (np.arange(height) + 0.5)
    return y_coords, x_coords


def _get_bounds_from_gt(gt, height, width):
    """
    Compute bounding box (left, bottom, right, top) in map units
    from geotransform and image shape.
    """
    y_coords, x_coords = _compute_coords(gt, height, width)
    left = np.min(x_coords)
    right = np.max(x_coords)
    bottom = np.min(y_coords)
    top = np.max(y_coords)
    return left, bottom, right, top


# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------
def coregister_stack(
    input_path,
    output_path=None,
    first_scene_mode="composite",      # "composite" or "first"
    composite_window_days=30,          # only used if first_scene_mode == "composite"
    grid_size=3,                       # e.g. 3 -> 3x3 grid, 5 -> 5x5 grid
    min_reliability_keep=10.0,         # below this: scene dropped
    min_reliability_update_ref=50.0,   # below this: kept but not used as new ref
):
    """
    Scene-by-scene co-registration of a Sentinel-2 spectral-temporal stack using AROSICS.

    Parameters
    ----------
    input_path : str
        Path to input NetCDF with 'Spectral_Temporal_Stack' DataArray (time, band, y, x).
    output_path : str or None, optional
        Path to write the co-registered NetCDF.
        If None, writes to same folder as input with '_coregistered' suffix.
    first_scene_mode : {"composite", "first"}, optional
        How to choose initial reference:
        - "composite": median composite of scenes within composite_window_days of first date.
        - "first":     use the first scene as reference (no shift).
    composite_window_days : int, optional
        Time span (days) after first scene used to compute the initial composite.
    grid_size : int, optional
        Number of grid cells per side for manual window positions (>= 2 recommended).
    min_reliability_keep : float or None, optional
        Minimum match reliability (in %) required to KEEP a scene in the output cube.
        Scenes below this are dropped completely.
    min_reliability_update_ref : float or None, optional
        Minimum match reliability (in %) required to UPDATE the reference.
        Scenes below this (but above keep threshold) are kept, but previous
        reference is reused for the next step.

    Returns
    -------
    out_ds : xarray.Dataset
        Output dataset containing the co-registered 'Spectral_Temporal_Stack' and 'spatial_ref'.
    output_path : str
        Path where the NetCDF file was written.
    """

    # ------------------------------------------------------------------
    # Load data (no manual cloud filtering here)
    # ------------------------------------------------------------------
    stac = xr.open_dataset(input_path)
    masked_stac = stac.Spectral_Temporal_Stack  # (time, band, y, x)

    filtered_data = masked_stac
    filtered_data = filtered_data.rio.write_crs(masked_stac.crs, inplace=True)

    # Get geotransform (assumed consistent for all dates)
    geotransform = filtered_data.spatial_ref.GeoTransform
    geotransform = [float(x) for x in geotransform.split()]

    # Get CRS WKT
    times = filtered_data.time.values
    sample_time = times[0]
    sample_im = filtered_data.sel(time=sample_time)
    crs_wkt = sample_im.spatial_ref.crs_wkt

    band_names = filtered_data.band.values
    height = filtered_data.sizes["y"]
    width = filtered_data.sizes["x"]

    # ------------------------------------------------------------------
    # Initial reference setup
    # ------------------------------------------------------------------
    corrected_images = []
    failed_times = []
    current_reference = None
    master_geoArr = None

    # for reliability statistics of kept scenes
    kept_reliabilities = []   # list of floats
    kept_rel_times = []       # corresponding times

    if first_scene_mode == "first":
        # Use the first scene as unshifted reference (classic chaining)
        im_ref = filtered_data.sel(time=times[0]).transpose("y", "x", "band")
        im_ref = im_ref.where(im_ref != 0, np.nan)

        y_coords, x_coords = _compute_coords(geotransform, height, width)
        im_ref = im_ref.assign_coords(
            {"y": ("y", y_coords), "x": ("x", x_coords), "time": times[0]}
        )

        corrected_images.append(im_ref)
        current_reference = im_ref
        start_idx = 1  # start from second scene

    elif first_scene_mode == "composite":
        # Build a median composite from all scenes within a window after first date
        first_time = times[0]
        end_time = first_time + np.timedelta64(composite_window_days, "D")

        subset = filtered_data.sel(time=slice(first_time, end_time))
        if subset.sizes["time"] == 0:
            subset = filtered_data  # fallback: use all times

        master_median = subset.median(dim="time", skipna=True)
        master_ref = master_median.transpose("y", "x", "band")
        master_ref = master_ref.where(master_ref != 0, np.nan)

        master_geoArr = GeoArray(
            master_ref.values,
            geotransform=geotransform,
            projection=crs_wkt,
        )

        # We start by registering *all* scenes, first against composite, then chained
        start_idx = 0

    else:
        raise ValueError("first_scene_mode must be 'first' or 'composite'")

    indices = list(range(start_idx, len(times)))
    total_to_process = len(indices)

    # ------------------------------------------------------------------
    # Main loop (chained reference, with dual thresholds)
    # ------------------------------------------------------------------
    for idx in tqdm(indices, total=total_to_process,
                    desc="Co-registering scenes", unit="scene"):
        t = times[idx]
        t_str = np.datetime_as_string(t, unit="D")

        im_target = filtered_data.sel(time=t).transpose("y", "x", "band")

        # Choose reference for this step
        if first_scene_mode == "composite" and current_reference is None:
            # Until we have our first "strong" scene, use composite master
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
            im_target.values,
            geotransform=geotransform,
            projection=crs_wkt,
        )

        # Candidate window positions (grid in bbox)
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
                label = f"g{grid_size}x{grid_size}_r{iy}_c{ix}"
                manual_wps.append((label, (x_wp, y_wp)))

        candidates = [("auto", None)] + manual_wps

        successful_matches = []

        with _suppress_arosics_warnings():
            for label, wp in candidates:
                try:
                    if wp is None:
                        CR_try = COREG(ref_geoArr, tgt_geoArr,
                                       align_grids=True, q=True)
                    else:
                        CR_try = COREG(
                            ref_geoArr, tgt_geoArr,
                            align_grids=True, q=True, wp=wp
                        )

                    CR_try.calculate_spatial_shifts()
                    result_try = CR_try.correct_shifts()
                    reliability_try = getattr(CR_try, "shift_reliability", None)
                    dx_px_try = getattr(CR_try, "x_shift_px", np.nan)
                    dy_px_try = getattr(CR_try, "y_shift_px", np.nan)
                    dx_map_try = getattr(CR_try, "x_shift_map", np.nan)
                    dy_map_try = getattr(CR_try, "y_shift_map", np.nan)

                    successful_matches.append(
                        {
                            "label": label,
                            "wp": wp,
                            "CR": CR_try,
                            "result": result_try,
                            "reliability": reliability_try,
                            "dx_px": dx_px_try,
                            "dy_px": dy_px_try,
                            "dx_map": dx_map_try,
                            "dy_map": dy_map_try,
                        }
                    )
                except (RuntimeError, ValueError, AssertionError):
                    continue

        # No candidate worked at all: drop scene
        if not successful_matches:
            failed_times.append(t)
            continue

        # Select candidate with highest reliability
        def rel_key(m):
            r = m["reliability"]
            return -np.inf if r is None else float(r)

        best_match = max(successful_matches, key=rel_key)

        CR = best_match["CR"]
        result = best_match["result"]
        reliability = best_match["reliability"]

        # 1) KEEP THRESHOLD: if too low, drop scene entirely
        if (min_reliability_keep is not None) and (
            (reliability is None) or (reliability < min_reliability_keep)
        ):
            failed_times.append(t)
            continue

        # Build corrected DataArray
        out_geoArr = GeoArray(
            result["arr_shifted"],
            result["updated geotransform"],
            result["updated projection"],
        )

        arr_corr = out_geoArr[:]
        arr_corr = arr_corr.transpose(2, 0, 1)
        arr_corr = np.where(arr_corr == 0, np.nan, arr_corr)

        updated_gt = result["updated geotransform"]
        height_corr, width_corr = arr_corr.shape[1], arr_corr.shape[2]
        y_coords_corr, x_coords_corr = _compute_coords(updated_gt, height_corr, width_corr)

        da_corr = xr.DataArray(
            arr_corr,
            dims=("band", "y", "x"),
            coords={
                "band": range(1, arr_corr.shape[0] + 1),
                "y": ("y", y_coords_corr),
                "x": ("x", x_coords_corr),
            },
        )
        affine_transform = Affine.from_gdal(*updated_gt)
        da_corr = da_corr.rio.write_transform(affine_transform)
        da_corr = da_corr.rio.write_crs(result["updated projection"])

        da_corr = da_corr.assign_coords(band=("band", band_names))
        da_corr = da_corr.transpose("y", "x", "band")
        da_corr = da_corr.assign_coords(time=t)

        corrected_images.append(da_corr)

        # Collect reliability stats for kept scenes
        if reliability is not None:
            kept_reliabilities.append(float(reliability))
            kept_rel_times.append(t)

        # 2) REFERENCE UPDATE THRESHOLD:
        #    - if reliability is high -> this scene becomes new reference
        #    - if not -> keep previous reference
        if (min_reliability_update_ref is None) or (
            (reliability is not None)
            and (reliability >= min_reliability_update_ref)
        ):
            current_reference = da_corr
        else:
            # reference remains unchanged
            pass

    # ------------------------------------------------------------------
    # Combine into corrected stack and wrap as Dataset
    # ------------------------------------------------------------------
    corrected_stack = xr.concat(corrected_images, dim="time")
    corrected_stack = corrected_stack.transpose("time", "band", "y", "x")
    corrected_stack = corrected_stack.rio.write_crs(crs_wkt, inplace=True)
    corrected_stack.name = "Spectral_Temporal_Stack"

    out_ds = xr.Dataset(
        {
            "Spectral_Temporal_Stack": corrected_stack,
            "spatial_ref": stac.spatial_ref,
        }
    )

    # Re-attach coords / labels similar to original
    out_ds = out_ds.assign_coords(
        band=stac.Spectral_Temporal_Stack.band,
        x=corrected_stack.x,
        y=corrected_stack.y,
        time=corrected_stack.time,
    )

    # Keep cloud_percentage if present
    if "cloud_percentage" in stac.coords:
        out_ds = out_ds.assign_coords(cloud_percentage=stac.cloud_percentage)
    elif "cloud_percentage" in stac:
        out_ds = out_ds.assign_coords(cloud_percentage=stac["cloud_percentage"])

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    n_total = len(times)
    n_failed = len(failed_times)
    n_final = len(corrected_images)

    t_start = np.datetime_as_string(times[0], unit="D")
    t_end = np.datetime_as_string(times[-1], unit="D")

    print("\nCo-registration summary")
    print("-----------------------")
    print(f"Original time series: {n_total} scenes from {t_start} to {t_end}")
    print(
        "Scenes excluded after co-registration "
        "(clouds / overlap issues / low reliability): ",
        n_failed,
    )
    print(f"Scenes remaining in the co-registered cube: {n_final}")

    if failed_times:
        failed_str = ", ".join(
            np.datetime_as_string(ts, unit="D") for ts in failed_times
        )
        print(f"Excluded dates: {failed_str}")

    # Reliability stats for kept scenes
    if kept_reliabilities:
        mean_rel = float(np.mean(kept_reliabilities))
        min_idx = int(np.argmin(kept_reliabilities))
        min_rel = kept_reliabilities[min_idx]
        min_rel_time = kept_rel_times[min_idx]
        min_rel_time_str = np.datetime_as_string(min_rel_time, unit="D")

        print(f"\nMean match reliability of kept scenes: {mean_rel:.1f} %")
        print(
            f"Minimum match reliability of kept scenes: {min_rel:.1f} % "
            f"(date: {min_rel_time_str})"
        )
    else:
        print(
            "\nNo reliability statistics available "
            "(no scenes passed the keep threshold)."
        )

    print("\nS2 co-registration is completed!")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    if output_path is None:
        in_dir, in_name = os.path.split(input_path)
        base, ext = os.path.splitext(in_name)
        if not ext:
            ext = ".nc"
        out_name = f"{base}_coregistered{ext}"
        output_path = os.path.join(in_dir, out_name)

    out_ds.to_netcdf(output_path)
    print(f"\nCo-registered cube written to: {output_path}")

    return out_ds, output_path
