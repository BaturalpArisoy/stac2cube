import os
import xarray as xr
import pyproj
import numpy as np

# from .vector_refiner import proj_2_geo


def get_stac_parameters(stac_existing):

    if isinstance(stac_existing, (str, os.PathLike)):
        stac_existing = xr.open_dataset(stac_existing)
        if "Spectral_Temporal_Stack" in list(stac_existing.data_vars):
            stac_existing = stac_existing.Spectral_Temporal_Stack
        elif "Cloud_Stack" in list(stac_existing.data_vars):
            stac_existing = stac_existing.Cloud_Stack

    # Polygon
    bbox = stac_existing.bbox
    bbox = bbox.tolist()
    # Full geometry # update-clip raster
    # if hasattr(stac_existing, "geometry"):
    #   geometry = stac_existing.geometry
    # else:
    #   geometry = []
    # Daterange
    daterange = [min(stac_existing.time.values), max(stac_existing.time.values)]
    daterange = [np.datetime_as_string(dt, unit="D") for dt in daterange]

    if stac_existing.name == "Cloud_Stack":
        mission = "sentinel_2_l1c"
        spectral_bands = [
            "coastal",
            "blue",
            "red",
            "rededge1",
            "nir",
            "nir08",
            "nir09",
            "cirrus",
            "swir16",
            "swir22",
        ]
        indices = []
        resolution = 10
    else:
        # Mission
        mission = stac_existing.mission
        # Resolution
        resolution = abs(stac_existing.y.resolution).item()
        # Spectral bands
        spectral_bands = stac_existing.spectral_bands
        # Indices
        indices = stac_existing.indices

    stac_parameters = {
        "mission": mission,
        "resolution": resolution,
        "polygon": bbox,
        #  "geometry": geometry, # update-clip raster
        "spectral_bands": spectral_bands,
        "indices": indices,
        "daterange": daterange,
    }

    return stac_parameters


def update_stac(stac_existing, stac_updated):

    stac_existing = xr.open_dataset(stac_existing)
    if "Spectral_Temporal_Stack" in list(stac_existing.data_vars):
        stac_existing = stac_existing.Spectral_Temporal_Stack
    elif "Cloud_Stack" in list(stac_existing.data_vars):
        stac_existing = stac_existing.Cloud_Stack
    # stac_existing = stac_existing.Spectral_Temporal_Stack

    stac_missing, missing_times = find_missing_times(stac_existing, stac_updated)

    # Compute the missing slices (only these will be computed now)
    computed_missing = stac_missing.compute()

    # Merge the computed missing data with the existing dataarray along the time dimension
    updated = xr.concat([stac_existing, computed_missing], dim="time")
    updated = updated.sortby("time")

    num_added = len(missing_times)
    # Generate a detailed report about the update with formatted dates (date only, no time)
    print("Update Report:")
    print("-------------------------")
    print(
        f"{num_added} new date{'s' if num_added != 1 else ''} have been integrated into the dataset."
    )
    print("The following dates were added:")

    # Format the dates to display only the date part using numpy's datetime_as_string
    for dt in missing_times:
        formatted_date = np.datetime_as_string(dt, unit="D")
        print(f" - {formatted_date}")
    print("-------------------------")

    return updated


def find_missing_times(stac_old, stac_new):

    existing_times = set(stac_old.time.values)
    updating_times = set(stac_new.time.values)

    # Identify the missing times (i.e., dates present in the lazy array but not in the computed one)
    missing_times = sorted(list(updating_times - existing_times))

    # Select only the missing dates from the lazy array
    stac_missing = stac_new.sel(time=missing_times)

    num_added = len(missing_times)
    print(f"{num_added} new date{'s' if num_added != 1 else ''} found!")

    return stac_missing, missing_times
