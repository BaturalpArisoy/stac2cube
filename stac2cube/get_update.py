import os
import xarray as xr
import pyproj
import numpy as np

from .export_cfg import open_cube

# from .vector_refiner import proj_2_geo


def get_stac_parameters(stac_existing):

    if isinstance(stac_existing, (str, os.PathLike)):
        stac_existing = open_cube(stac_existing)
        if "Spectral_Temporal_Stack" in list(stac_existing.data_vars):
            stac_existing = stac_existing.Spectral_Temporal_Stack
        elif "Cloud_Stack" in list(stac_existing.data_vars):
            stac_existing = stac_existing.Cloud_Stack

    # Polygon. np.asarray: NetCDF stores the attr as an ndarray, Zarr (JSON
    # attrs) returns a plain list - normalize before .tolist().
    bbox = np.asarray(stac_existing.bbox)
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
        # Resolution (float(): the attr is a numpy scalar from NetCDF but a
        # plain Python float from Zarr's JSON attrs, which has no .item())
        resolution = float(abs(stac_existing.y.resolution))
        # Spectral bands
        spectral_bands = stac_existing.spectral_bands
        # Indices
        if hasattr(stac_existing, "indices"):
            indices = stac_existing.indices
        else:
            indices = None

    # SCL cloud strategy. Legacy cubes predate the cloud_status attribute:
    # fall back on the cloud_percentage coordinate - present means clouds were
    # assessed (treat as masked), absent means clouds were never detected.
    cloud_status = stac_existing.attrs.get("cloud_status")
    if cloud_status is None:
        cloud_status = (
            "scl_masked"
            if "cloud_percentage" in stac_existing.coords
            else "clouds_not_detected"
        )

    stac_parameters = {
        "mission": mission,
        "resolution": resolution,
        "polygon": bbox,
        #  "geometry": geometry, # update-clip raster
        "spectral_bands": spectral_bands,
        "indices": indices,
        "daterange": daterange,
        "stac_api": stac_existing.attrs.get("stac_api", "element84"),
        "resampling": stac_existing.attrs.get("resampling", "bilinear"),
        # The cube's own projection, so update mode can pin it and put the new
        # scenes on the SAME grid instead of re-deriving a target CRS. Fresh
        # cubes store "EPSG:<code>"; legacy ones a CRS name, which normalises.
        "crs": stac_existing.attrs.get("crs"),
        # Scene-metadata coordinate selection (None on cubes built without
        # it). Restored by update mode so new scenes carry the same coords.
        "scene_metadata": stac_existing.attrs.get("scene_metadata"),
        # How AOI-straddling tiles were handled ("mosaic"/"separate"). Update
        # mode refuses to touch a "separate" cube (sub-day per-tile times).
        "tile_handling": stac_existing.attrs.get("tile_handling", "mosaic"),
        "cloud_status": cloud_status,
        # Shadow projection params (present only on scl_shadow_masked cubes).
        "nir_dark_threshold": stac_existing.attrs.get("nir_dark_threshold"),
        "shadow_proj_distance": stac_existing.attrs.get("shadow_proj_distance"),
        # Full stored time axis (day precision) - lets update mode subset the
        # fresh query to the missing dates before any masking work.
        "times": np.asarray(stac_existing.time.values),
    }

    return stac_parameters


def update_stac(stac_existing, stac_updated, new_bands=None):

    stac_existing = open_cube(stac_existing)
    if "Spectral_Temporal_Stack" in list(stac_existing.data_vars):
        stac_existing = stac_existing.Spectral_Temporal_Stack
    elif "Cloud_Stack" in list(stac_existing.data_vars):
        stac_existing = stac_existing.Cloud_Stack
    # stac_existing = stac_existing.Spectral_Temporal_Stack

    # Floor existing times to day up front so both the band merge and the time
    # concat align cleanly with the new data (main.py floors stac_updated).
    stac_existing = stac_existing.assign_coords(
        time=stac_existing.time.astype("datetime64[D]").astype("datetime64[ns]")
    )

    stac_missing, missing_times = find_missing_times(stac_existing, stac_updated)

    # ---- Band addition -------------------------------------------------------
    # new_bands: bands in the fresh query that the existing cube lacks (see the
    # update block in main.py). Only these bands are computed for the dates
    # already stored and appended along the band dimension. Masking consistency
    # is inherent: the fresh query was built with the cube's restored
    # cloud/shadow strategy, so the new bands arrive masked (or not) exactly
    # like the stored ones.
    added_bands = []
    nan_filled_dates = []
    if new_bands:
        added_bands = [str(b) for b in new_bands]
        stac_new = stac_updated.assign_coords(
            time=stac_updated.time.astype("datetime64[D]").astype("datetime64[ns]")
        )

        # The band merge is only valid on an identical spatial grid.
        if not (
            np.array_equal(stac_new.x.values, stac_existing.x.values)
            and np.array_equal(stac_new.y.values, stac_existing.y.values)
        ):
            raise ValueError(
                "Band addition aborted: the fresh query's spatial grid does "
                "not match the existing cube's grid (was the cube clipped "
                "with a polygon, coregistered or resampled after creation?). "
                "Bands can only be added on the cube's original bbox grid."
            )

        existing_times = stac_existing.time.values
        new_times = set(stac_new.time.values)
        common = [t for t in existing_times if t in new_times]
        nan_filled_dates = [t for t in existing_times if t not in new_times]

        # Compute ONLY the new bands on the already-stored dates.
        band_slab = stac_new.sel(time=common, band=added_bands).compute()
        if nan_filled_dates:
            # Dates in the cube that the fresh query did not return (e.g.
            # scenes gone from the catalogue): the added bands cannot be
            # fetched for them and stay NaN (reported below).
            band_slab = band_slab.reindex(time=existing_times)
        # Per-scene coords (cloud_percentage) were recomputed by the fresh
        # query; the existing cube's values win, so drop them from the slab
        # to avoid a conflict in the band concat.
        _extra = [
            c for c in band_slab.coords if c not in ("time", "band", "y", "x")
        ]
        band_slab = band_slab.drop_vars(_extra, errors="ignore")

        stac_existing = xr.concat(
            [stac_existing, band_slab], dim="band", coords="minimal"
        )

        # Keep the fresh-build band convention: spectral bands first, indices
        # last (the concat above appended the new bands after the indices).
        _idx_attr = stac_existing.attrs.get("indices")
        _idx = (
            []
            if _idx_attr is None
            else [str(i) for i in np.asarray(_idx_attr).ravel()]
        )
        _order = [str(b) for b in stac_existing.band.values]
        final_order = (
            [b for b in _order if b not in _idx and b not in added_bands]
            + added_bands
            + [b for b in _order if b in _idx]
        )
        stac_existing = stac_existing.sel(band=final_order)

    if missing_times:
        # Compute the missing slices (only these will be computed now)
        computed_missing = stac_missing.compute()
        if added_bands:
            # Match the (band-extended) existing cube's band order, otherwise
            # the time concat's outer join silently re-sorts the bands
            # alphabetically.
            computed_missing = computed_missing.sel(
                band=[str(b) for b in stac_existing.band.values]
            )

        # Merge the computed missing data with the existing dataarray along the time dimension
        updated = xr.concat([stac_existing, computed_missing], dim="time")
        updated = updated.sortby("time")
    else:
        # Nothing to append along time. Skipping the concat matters: with a
        # dask-backed cube (Zarr) concatenating a zero-length slab crashes in
        # dask's auto-chunking.
        updated = stac_existing

    if added_bands:
        _sb = updated.attrs.get("spectral_bands", [])
        _sb = [str(b) for b in np.asarray(_sb).ravel()]
        updated.attrs["spectral_bands"] = _sb + added_bands
        # The band coord inherited the source file's fixed-width string
        # encoding (e.g. '<U5' when all stored names are <=5 chars); to_netcdf
        # honors it and would silently truncate longer added names such as
        # 'swir22' -> 'swir2'. Drop it so the writer re-infers the width.
        updated["band"].encoding = {}

    num_added = len(missing_times)
    # Generate a detailed report about the update with formatted dates (date only, no time)
    print("Update Report:")
    print("-------------------------")
    print(
        f"{num_added} new date{'s' if num_added != 1 else ''} have been integrated into the dataset."
    )
    if missing_times:
        print("The following dates were added:")

        # Format the dates to display only the date part using numpy's datetime_as_string
        for dt in missing_times:
            formatted_date = np.datetime_as_string(dt, unit="D")
            print(f" - {formatted_date}")
    if added_bands:
        print(
            f"{len(added_bands)} new band{'s' if len(added_bands) != 1 else ''} "
            "have been integrated into the dataset:"
        )
        for b in added_bands:
            print(f" - {b}")
        if nan_filled_dates:
            print(
                "WARNING: the fresh query did not return the following stored "
                "dates; the added bands are NaN there:"
            )
            for dt in nan_filled_dates:
                print(f" - {np.datetime_as_string(dt, unit='D')}")
    print("-------------------------")

    return updated


def find_missing_times(stac_old, stac_new):

    # Floor both to day-precision so timestamps like 10:23:45 vs 00:00:00
    # on the same date are treated as equal (main.py floors stac_new already).
    stac_old = stac_old.assign_coords(
        time=stac_old.time.astype("datetime64[D]").astype("datetime64[ns]")
    )
    stac_new = stac_new.assign_coords(
        time=stac_new.time.astype("datetime64[D]").astype("datetime64[ns]")
    )

    existing_times = set(stac_old.time.values)
    updating_times = set(stac_new.time.values)

    # Identify the missing times (i.e., dates present in the lazy array but not in the computed one)
    missing_times = sorted(list(updating_times - existing_times))

    # Select only the missing dates from the lazy array
    stac_missing = stac_new.sel(time=missing_times)

    num_added = len(missing_times)
    print(f"{num_added} new date{'s' if num_added != 1 else ''} found!")

    return stac_missing, missing_times
