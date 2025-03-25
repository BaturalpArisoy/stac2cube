import os
import xarray as xr
import pyproj
import numpy as np
#from .vector_refiner import proj_2_geo



def get_stac_parameters(stac_existing):

    if isinstance(stac_existing, (str, os.PathLike)):
        stac_existing = xr.open_dataset(stac_existing)
        stac_existing = stac_existing.Spectral_Temporal_Stack
    if isinstance(stac_existing, xr.Dataset):
        stac_existing = stac_existing.Spectral_Temporal_Stack

    # Mission
    mission = stac_existing.mission
    # Resolution
    resolution = abs(stac_existing.y.resolution).item()
    # Polygon
    bbox = stac_existing.bbox
    bbox = bbox.tolist()
    # Get projected coordinates, slightly different polygon. Deactivated for now.
    '''
    x_min = stac_existing.coords["x"].min().item()
    x_max = stac_existing.coords["x"].max().item()
    y_min = stac_existing.coords["y"].min().item()
    y_max = stac_existing.coords["y"].max().item()
    bbox = [x_min, y_min, x_max, y_max]
    print(bbox)
    proj_str = stac_existing.crs
    crs_obj = pyproj.CRS.from_string(proj_str)
    source_epsg = crs_obj.to_epsg()
    bbox = proj_2_geo(polygon=bbox, source_epsg=source_epsg)
    '''
    # Spectral bands
    spectral_bands = stac_existing.spectral_bands
    # Indices
    indices = stac_existing.indices


    stac_parameters = {
        "mission": mission,
        "resolution": resolution,
        "polygon": bbox,
        "spectral_bands": spectral_bands,
        "indices": indices
    }

    return stac_parameters
    


def update_stac(stac_existing, stac_updated):

    stac_existing = xr.open_dataset(stac_existing)
    stac_existing = stac_existing.Spectral_Temporal_Stack
    
    # Compare the time coordinates to determine missing dates
    existing_times = set(stac_existing.time.values)
    updating_times = set(stac_updated.time.values)

    # Identify the missing times (i.e., dates present in the lazy array but not in the computed one)
    missing_times = sorted(list(updating_times - existing_times))

    # Select only the missing dates from the lazy array
    stac_missing = stac_updated.sel(time=missing_times)

    # Compute the missing slices (only these will be computed now)
    computed_missing = stac_missing.compute()

    # Merge the computed missing data with the existing dataarray along the time dimension
    updated = xr.concat([stac_existing, computed_missing], dim="time")
    updated = updated.sortby("time")

    # Generate a detailed report about the update with formatted dates (date only, no time)
    num_added = len(missing_times)
    print("Update Report:")
    print("-------------------------")
    print(f"{num_added} new date{'s' if num_added != 1 else ''} have been integrated into the dataset.")
    print("The following dates were added:")

    # Format the dates to display only the date part using numpy's datetime_as_string
    for dt in missing_times:
        formatted_date = np.datetime_as_string(dt, unit='D')
        print(f" - {formatted_date}")
    print("-------------------------")
    print("\nUpdated STAC DataArray summary:")
    #print(updated)

    return updated