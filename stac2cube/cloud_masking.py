from .main import get_stac_layers
from .get_update import get_stac_parameters
from .get_update import find_missing_times
from s2cloudless import S2PixelCloudDetector
import numpy as np
import xarray as xr
import sys
from .export_cfg import export_stac
import rioxarray as rio
import cv2
import os
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def get_cloud_layers(
    polygon=None,
    daterange=None,
    output_clouds=None,
    output_masked=None,
    output=None,
    threshold=None,
    clip_raster=None,
    masking=None,
    update=None,
    slurm_timer=None,
):
    if output_clouds is None and output is not None:
        output_clouds = output

    # If we are called from an existing cube (seasonal mode), we must use its exact time list
    reference_times = None

    if masking:
        stac_parameters = get_stac_parameters(masking)
        polygon = stac_parameters["polygon"]
        daterange = stac_parameters["daterange"]

        # Use the exact seasonal timestamps from the initial cube
        with xr.open_dataset(masking) as ds:
            if "Spectral_Temporal_Stack" in ds:
                reference_times = ds["Spectral_Temporal_Stack"].time.values
            else:
                reference_times = ds["time"].values

    if update:
        stac_parameters = get_stac_parameters(update)
        polygon = stac_parameters["polygon"]
        if daterange is None:
            daterange = stac_parameters.get("daterange")

        # In update mode, restrict to the times already present in the existing cloud cube
        with xr.open_dataset(update) as ds:
            if "Cloud_Stack" in ds:
                reference_times = ds["Cloud_Stack"].time.values
            else:
                reference_times = ds["time"].values

    if not daterange:
        raise ValueError("Error: Please select a daterange.")
    if not polygon:
        raise ValueError("Error: Please select a polygon or bbox list with geographic coordinates.")

    # --- STAC Retrieval ---
    max_cc = 100
    mission = "sentinel_2_l1c"
    bands = [
        "coastal", "blue", "red", "rededge1", "nir", "nir08", "nir09",
        "cirrus", "swir16", "swir22",
    ]

    def _filter_to_reference_times(stac_da: xr.DataArray, ref_times) -> xr.DataArray:
        st = np.asarray(stac_da.time.values).astype("datetime64[ns]")
        rt = np.asarray(ref_times).astype("datetime64[ns]")

        # 1) exact timestamp match
        if np.all(np.isin(rt, st)):
            order = [int(np.where(st == t)[0][0]) for t in rt]
            out = stac_da.isel(time=order).assign_coords(time=ref_times)
            return out

        # 2) fallback: day-level matching (handles duplicates in the reference list)
        st_d = st.astype("datetime64[D]")
        rt_d = rt.astype("datetime64[D]")

        from collections import defaultdict
        pos = defaultdict(list)
        for i, d in enumerate(st_d):
            pos[d].append(i)

        used = defaultdict(int)
        order = []
        missing = []
        for d in rt_d:
            k = used[d]
            if d not in pos or k >= len(pos[d]):
                missing.append(d)
            else:
                order.append(pos[d][k])
                used[d] += 1

        if missing:
            ex = ", ".join(np.datetime_as_string(m, unit="D") for m in missing[:5])
            raise ValueError(
                "Cloud STAC retrieval is missing some reference dates. "
                f"Missing (first up to 5): {ex}"
            )

        out = stac_da.isel(time=order)
        # Keep the reference time coordinate for alignment
        if out.sizes["time"] == len(ref_times):
            out = out.assign_coords(time=ref_times)
        return out

    stac = get_stac_layers(
        mission=mission,
        polygon=polygon,
        daterange=daterange,
        bands=bands,
        max_cc=max_cc,
        clip_raster=clip_raster,
        q=True,
    )

    if reference_times is not None:
        stac = _filter_to_reference_times(stac, reference_times)

    crs = stac.crs
    transform = stac.transform
    bbox = stac.bbox

    if update:
        with xr.open_dataset(update) as ds:
            stac_existing = ds["Cloud_Stack"].load()
        stac, missing_times = find_missing_times(stac_existing, stac)
        if not missing_times:
            raise ValueError("The probability map is up to date. Nothing to update!")

    # --- Cloud Probability Calculation ---
    # Set the parameters for the cloud detector.
    # Default threshold (0.7) for computing cloud probability.
    average_over = 4
    dilation_size = 2
    default_threshold = 0.7
    cloud_detector = S2PixelCloudDetector(
        threshold=default_threshold,
        average_over=average_over,
        dilation_size=dilation_size,
        all_bands=False,
    )

    cloud_prob_results = []
    times = []  # To store the time coordinate for each processed slice
    total = len(stac.time)

    if slurm_timer:
        import time

        slurm_timer = slurm_timer * 3600
        start_time = time.time()

    for i, t in enumerate(stac.time.values, start=1):
        # Retrieve and compute the current time slice.
        img = stac.sel(time=t).compute()
        times.append(t)

        # Transpose to (y, x, band) for s2cloudless and add a batch dimension.
        img_transposed = img.transpose("y", "x", "band")
        img_np = img_transposed.to_numpy()[np.newaxis, ...]

        # Compute the cloud probability maps (3D with shape: (batch, y, x)).
        cp_3d = cloud_detector.get_cloud_probability_maps(img_np)
        cp = cp_3d[0]
        cloud_prob_results.append(cp)

        t_str = np.datetime_as_string(t, unit="D")  # -> "YYYY-MM-DD"
        print(f"Processed time slice: {i}/{total} (time: {t_str})", flush=True)
        del img, img_transposed, img_np

        if slurm_timer:
            # Calculate the elapsed time
            elapsed = time.time() - start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            # Check if the elapsed time has reached or exceeded the threshold
            if elapsed >= slurm_timer:
                print(
                    "Time threshold reached! Exiting loop and exporting the collected cloud maps..."
                )
                break

    # Assemble the cloud probability DataArray.
    cp_stack = np.stack(cloud_prob_results, axis=0)  # shape: (time, y, x)
    cp_da = xr.DataArray(
        cp_stack,
        dims=["time", "y", "x"],
        coords={"time": times, "y": stac.y, "x": stac.x},
    )
    cp_da = cp_da.expand_dims(dim={"band": ["cloud_prob"]})

    cp_da.name = "Cloud_Stack"

    def update_prob_maps(stac_existing, cloud_only_stack):
        # keep band dimension with correct label
        stac_existing = stac_existing.sel(band=["cloud_prob"])
        cloud_only_stack = cloud_only_stack.sel(band=["cloud_prob"])

        out = xr.concat([stac_existing, cloud_only_stack], dim="time")
        out = out.sortby("time")
        return out

    # --- Determine Output Based on 'threshold' Parameter ---
    # If no threshold(s) are provided, return only the probability layer.

    # Always build probability layer first (uint8 0-100)
    cloud_prob_uint8 = (cp_da.sel(band="cloud_prob") * 100).astype(np.uint8)

    # Create a proper 4D stack: (time, band, y, x) with band label "cloud_prob"
    cloud_only_stack = cloud_prob_uint8.expand_dims(band=["cloud_prob"]).transpose("time", "band", "y", "x")
    cloud_only_stack.name = "Cloud_Stack"

    # If update: merge new probability dates into existing stack
    if update:
        cloud_only_stack = update_prob_maps(stac_existing, cloud_only_stack)

    # If threshold(s) are provided (NON-update only): compute masks and concat
    if threshold is not None:
        mask_da = mask_from_probability(
            cloud_only_stack.sel(band="cloud_prob"),
            threshold=threshold,
            average_over=average_over,
            dilation_size=dilation_size,
        )
        cloud_only_stack = xr.concat([cloud_only_stack, mask_da], dim="band").transpose("time", "band", "y", "x")

    # ---- attrs: set ALWAYS (update or not) ----
    cloud_only_stack.attrs["bbox"] = bbox
    cloud_only_stack.attrs["crs"] = crs
    cloud_only_stack.attrs["transform"] = transform

    # ---- export: do NOT hide behind `if not update` ----
    if output_clouds is not None:
        export_stac(cloud_only_stack, output_clouds, crs, transform)
    

    # ---- Masking (kept as before; typically not combined with update) ----
    if masking:
        if threshold is None:
            raise ValueError("Error: 'threshold' must be set when 'masking' is used.")
        if isinstance(threshold, list):
            raise ValueError("Error: 'masking' supports only a single threshold (not a list).")

        thr = int(threshold)
        mask_layer = f"cloud_mask_{thr}"

        if output_masked is None:
            dirname, filename = os.path.split(masking)
            name, ext = os.path.splitext(filename)
            output_masked = os.path.join(dirname, f"{name}_masked_{thr}{ext}")

        return mask_stac_clouds(masking, cloud_only_stack, mask_layer, output_masked)

    # Always return in-memory stack
    return cloud_only_stack


def mask_stac_clouds(stac, cloud, mask_layer, output=None):
    if isinstance(stac, (str, os.PathLike)):
        stac = xr.open_dataset(stac)
        stac = stac.Spectral_Temporal_Stack

    if isinstance(cloud, (str, os.PathLike)):
        cloud = xr.open_dataset(cloud)
        cloud = cloud.Cloud_Stack

    if isinstance(stac, xr.Dataset):
        stac = stac.Spectral_Temporal_Stack

    if isinstance(cloud, xr.Dataset):
        cloud = cloud.Cloud_Stack

    cloud_mask = cloud.sel(band=mask_layer)
    masked_stac = stac.where(cloud_mask == 0)

    # Calculate cloud percentage per time slice.
    null_count_per_time = masked_stac.isnull().sum(dim=["band", "y", "x"])
    total_elements = (
        masked_stac.sizes["band"] * masked_stac.sizes["y"] * masked_stac.sizes["x"]
    )
    cloud_percentage_int = ((null_count_per_time / total_elements) * 100).astype(int)
    masked_stac = masked_stac.assign_coords(
        cloud_percentage=("time", cloud_percentage_int.data)
    )

    #export_stac(masked_stac, output)
    if output is not None:
        export_stac(masked_stac, output)
        return output  # return path (old code returned None anyway)
    else:
        return masked_stac  # return in-memory masked cube


def mask_from_probability(
    cloud_probability, threshold=0.7, average_over=4, dilation_size=2
):

    if not isinstance(threshold, list):
        thresholds = [threshold]
    else:
        thresholds = threshold

    # Normalize probabilities to [0, 1] if necessary.
    if cloud_probability.max() > 1:
        prob_da = cloud_probability / 100.0
    else:
        prob_da = cloud_probability

    band_dataarrays = []

    for t_val in thresholds:
        # Scale the threshold from 0-100 to 0-1.
        scaled_threshold = t_val / 100.0
        cloud_detector = S2PixelCloudDetector(
            threshold=scaled_threshold,
            average_over=average_over,
            dilation_size=dilation_size,
        )
        mask_list = []

        for t in prob_da.time.values:
            prob_slice = prob_da.sel(time=t)
            prob_np = prob_slice.to_numpy()[np.newaxis, ...]
            cm = cloud_detector.get_mask_from_prob(prob_np, threshold=scaled_threshold)
            mask_da = xr.DataArray(
                cm[0], dims=["y", "x"], coords={"y": prob_slice.y, "x": prob_slice.x}
            )
            mask_list.append(mask_da)

        threshold_mask_da = xr.concat(mask_list, dim="time")
        threshold_mask_da = threshold_mask_da.assign_coords(time=prob_da.time)
        band_label = f"cloud_mask_{int(t_val)}"
        threshold_mask_da = threshold_mask_da.expand_dims(dim={"band": [band_label]})
        band_dataarrays.append(threshold_mask_da)

    final_mask_da = xr.concat(band_dataarrays, dim="band")
    final_mask_da = final_mask_da.transpose("time", "band", "y", "x")
    final_mask_da.name = "Cloud_Stack"

    return final_mask_da


def cloud_filter(inp, max_cloud):
    """
    Keep only time steps where cloud_percentage <= max_cloud.

    - if inp is a netcdf path (str): open it, take ds["Spectral_Temporal_Stack"], filter
    - if inp is an xr.Dataset: take ds["Spectral_Temporal_Stack"], filter
    - if inp is an xr.DataArray: filter directly
    """
    if isinstance(inp, str):
        da = xr.open_dataset(inp)["Spectral_Temporal_Stack"]
    elif isinstance(inp, xr.Dataset):
        da = inp["Spectral_Temporal_Stack"]
    else:  # assume xr.DataArray
        da = inp

    # da = da.where(da["cloud_percentage"] <= int(max_cloud), drop=True)
    # da = da.rio.write_crs(da.crs)

    return da.where(da["cloud_percentage"] <= int(max_cloud), drop=True)
