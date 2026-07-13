import os

from .get_data import get_stac, _S2_BNUM_TO_COMMON
from .vector_refiner import (
    proj_check,
    polygon_2_bbox,
    read_polygon_file,
    polygon_2_features,
)
from .stac_processing import scale_factor, cloud_mask, build_scl_mask_cube
from .get_spectral_indices import calculate_spectral_index
from .export_cfg import export_stac

# from .get_topo import calculate_topo
# from .time_series_tools import generate_animation
from .clip import clip_stac, compute_cloud_percentage
from .get_statistics import calculate_statistics
from .get_update import get_stac_parameters, update_stac

import xarray as xr
import rioxarray as rio
import pandas as pd
import numpy as np


def _human_size(nbytes):
    """Human-readable size for a (logical) byte count."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(nbytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def _mask_new_scenes_s2cloudless(
    stac, threshold, shadow, nir_dark_threshold, proj_distance
):
    """Mask an in-memory (unmasked) L2A cube with s2cloudless for update mode.

    ``stac`` holds only the newly added dates, so the L1C download + detector
    (and optional shadow projection) run on those scenes alone. Returns the
    masked Spectral_Temporal_Stack. Imported lazily: cloud_masking imports this
    module, so a top-level import would be circular at package load.
    """
    if shadow:
        from .shadow_masking import get_shadow_layers

        _stack, masked = get_shadow_layers(
            input_cube=stac,
            cloud_source="s2cloudless",
            threshold=threshold,
            nir_dark_threshold=nir_dark_threshold,
            proj_distance=proj_distance,
            masking=True,
        )
        return masked

    from .cloud_masking import get_cloud_layers

    return get_cloud_layers(masking=stac, threshold=threshold, output_masked=None)


def get_stac_layers(
    mission=None,
    polygon=None,
    resolution=None,
    daterange=None,
    bands=None,
    max_cc=None,
    clip_raster=None,
    cloud_masking=None,
    keep_clouds=None,
    shadow_masking=None,
    nir_dark_threshold=0.18,
    shadow_proj_distance=1.0,
    cloud_mask_output=None,
    return_cloud_mask=False,
    indices=None,
    output=None,
    aggregator=None,
    stats=None,
    topographic_features=None,
    animation=None,
    update=None,
    source=None,
    resampling_method=None,
    q=None,
    compress=False,
):

    # Reassign short names
    if mission == "s2":
        mission = "sentinel_2_l2a"
    if mission == "s2_l1c":
        mission = "sentinel_2_l1c"
    if mission == "s1":
        mission = "sentinel_1_rtc"
    if mission == "l_oli":
        mission = "landsat_c2_l2"
    if mission == "cop_dem":
        mission = "cop_dem_glo_30"

    # --- Cloud shadow masking preconditions -----------------------------------
    # Shadows are projected FROM the detected clouds along the anti-solar
    # direction, so shadow masking is meaningless without cloud detection and
    # needs the nir band for the dark-pixel test (GEE s2cloudless approach).
    if shadow_masking:
        if update:
            raise ValueError(
                "shadow_masking is not supported in update mode. Update the cube "
                "first, then apply get_shadow_layers to it."
            )
        if mission != "sentinel_2_l2a":
            raise ValueError(
                "shadow_masking is available for Sentinel-2 L2A cubes only."
            )
        if cloud_masking is not True:
            raise ValueError(
                "shadow_masking requires cloud_masking=True (shadows are "
                "projected from the detected clouds)."
            )
        if aggregator:
            raise ValueError(
                "shadow_masking cannot be combined with an aggregator (it needs "
                "the time dimension and per-scene solar geometry)."
            )
        if not bands or "nir" not in [str(b).lower() for b in bands]:
            raise ValueError(
                "shadow_masking needs the 'nir' band among the requested bands "
                "(dark-pixel test of the GEE method)."
            )

    # --- Native multi-feature batching ---------------------------------------
    # When a polygon FILE containing more than one feature is supplied (i.e. not
    # a bbox list and not update mode), generate one data cube per feature
    # instead of a single cube over the union of all features. Each feature is
    # processed by recursing with a one-row GeoDataFrame as the polygon, which
    # flows through the normal pipeline (search bbox + optional clip) unchanged.
    if not update and not isinstance(polygon, (list, tuple)):
        features = polygon_2_features(polygon)
        if len(features) > 1:
            n = len(features)
            results = []
            masks = []  # per-feature in-memory binary masks (return_cloud_mask)
            for pos, feature_gdf in enumerate(features):
                idx = pos + 1  # human-friendly: count features from 1, not 0
                if not q:
                    print(f"\n=== Feature {idx}/{n} ===", flush=True)

                # One binary cloud-mask file per feature (<stem>_<idx><ext>),
                # mirroring how the main cube is split per feature.
                cmo_i = None
                if cloud_mask_output:
                    _cstem, _cext = os.path.splitext(cloud_mask_output)
                    cmo_i = f"{_cstem}_{idx}{_cext}"

                # Build this feature LAZILY (output=None) so nothing is computed yet.
                res = get_stac_layers(
                    mission=mission,
                    polygon=feature_gdf,
                    resolution=resolution,
                    daterange=daterange,
                    # copy mutable args so per-feature calls don't share/append
                    # to the same list (e.g. cloud-mask band injection in get_stac)
                    bands=list(bands) if bands else bands,
                    max_cc=max_cc,
                    clip_raster=clip_raster,
                    cloud_masking=cloud_masking,
                    keep_clouds=keep_clouds,
                    shadow_masking=shadow_masking,
                    nir_dark_threshold=nir_dark_threshold,
                    shadow_proj_distance=shadow_proj_distance,
                    cloud_mask_output=cmo_i,
                    return_cloud_mask=return_cloud_mask,
                    indices=list(indices) if indices else indices,
                    output=None,  # export below, per feature, so RAM frees each time
                    aggregator=aggregator,
                    stats=stats,
                    topographic_features=topographic_features,
                    animation=animation,
                    update=update,
                    source=source,
                    resampling_method=resampling_method,
                    q=q,
                    compress=compress,
                )

                # When asked for the in-memory mask, each feature returns
                # (cube, mask); split them out and collect the masks.
                if return_cloud_mask:
                    res, _mask_i = res
                    masks.append(_mask_i)

                # Export this feature now. The heavy compute happens INSIDE
                # export_stac and is released when this iteration ends, while `res`
                # stays lazy -- so RAM does not accumulate across features (peak is
                # ~one feature, not the whole batch).
                if output and isinstance(res, (xr.DataArray, xr.Dataset)):
                    stem, ext = os.path.splitext(output)
                    feature_output = f"{stem}_{idx}{ext}"
                    export_stac(res, feature_output, compress=compress)

                # Report each cube's estimated (logical, pre-load) data size.
                if isinstance(res, (xr.DataArray, xr.Dataset)):
                    size = _human_size(int(res.nbytes))
                    res.attrs["estimated_size"] = size
                    if not q:
                        print(
                            f"Feature {idx}/{n} — estimated size: {size}",
                            flush=True,
                        )

                results.append(res)

            if return_cloud_mask:
                return results, masks
            return results

    # s2cloudless update reproduction state (set in the cloud_status restore
    # below). s2cloudless masking is a post-process, not part of the L2A query,
    # so these drive a masking pass applied to the new scenes before the merge.
    _s2c_update = False
    _s2c_threshold = None
    _s2c_shadow = False
    _s2c_nir_dark = nir_dark_threshold
    _s2c_proj = shadow_proj_distance

    if update:
        stac_parameters = get_stac_parameters(update)
        existing_times = stac_parameters["times"]

        mission = stac_parameters["mission"]
        resolution = stac_parameters["resolution"]
        polygon = stac_parameters["polygon"]
        # geometry = stac_parameters["geometry"] # update-clip raster
        user_bands = bands  # bands passed alongside update = bands to ADD
        bands = stac_parameters["spectral_bands"]
        if not isinstance(bands, list):
            # NetCDF stores the attr as an ndarray, Zarr returns a list.
            bands = np.asarray(bands).tolist()
        indices = stac_parameters["indices"]
        if indices is None:
            # Cube built without any spectral indices - nothing to restore.
            indices = []
        elif not isinstance(indices, list):
            indices = indices.tolist()

        # --- Band addition ---------------------------------------------------
        # Bands passed alongside update are bands to ADD to the existing cube.
        # The stored bands are always re-fetched too (the fresh lazy query
        # needs them for the restored cloud/shadow strategy and for any newly
        # found dates); update_stac then appends only the genuinely new bands
        # to the dates already stored.
        add_bands = []
        if user_bands:
            _stored = {str(b).lower() for b in bands}
            _idx = {str(i).lower() for i in indices}
            for b in user_bands:
                bl = str(b).lower()
                if mission == "sentinel_2_l2a":
                    bl = _S2_BNUM_TO_COMMON.get(bl, bl)
                if bl in _stored:
                    if not q:
                        print(f"Band '{bl}' is already in the cube - skipping.")
                elif bl in _idx:
                    if not q:
                        print(
                            f"'{bl}' is a spectral index already in the cube - skipping."
                        )
                elif bl not in add_bands:
                    add_bands.append(bl)
            bands = bands + add_bands

        # --- Daterange ---------------------------------------------------------
        # Default to the cube's own time span. When bands are being added the
        # query MUST cover every existing date (otherwise the new bands would
        # stay NaN on the uncovered dates), so a user-given range is widened
        # to include the stored span.
        stored_dr = stac_parameters["daterange"]
        if daterange is None:
            daterange = list(stored_dr)
        elif add_bands:
            _lo = min(np.datetime64(str(daterange[0])), np.datetime64(str(stored_dr[0])))
            _hi = max(np.datetime64(str(daterange[1])), np.datetime64(str(stored_dr[1])))
            _widened = [
                np.datetime_as_string(_lo, unit="D"),
                np.datetime_as_string(_hi, unit="D"),
            ]
            if _widened != [str(daterange[0]), str(daterange[1])] and not q:
                print(
                    f"Daterange widened to {_widened} so the added bands cover "
                    "every stored date."
                )
            daterange = _widened

        if source is None:
            source = stac_parameters.get("stac_api", "element84")
        if resampling_method is None:
            resampling_method = stac_parameters.get("resampling", "bilinear")
        # Reproduce the original cube's SCL cloud strategy on the new scenes,
        # unless the caller overrode it explicitly. Without this, new scenes
        # would miss the cloud_percentage coordinate (concat fails) or be
        # masked/kept inconsistently with the existing scenes.
        if cloud_masking is None:
            cs = stac_parameters.get("cloud_status", "clouds_not_detected")
            if cs == "clouds_detected":
                cloud_masking = True
                if keep_clouds is None:
                    keep_clouds = True
            elif cs == "scl_masked":
                cloud_masking = True
                if keep_clouds is None:
                    keep_clouds = False
            elif cs == "scl_shadow_masked":
                # Reproduce cloud + shadow masking on the new scenes with the
                # cube's original shadow params. Shadow always implies masking
                # (keep_clouds=False). The precondition block near the top ran
                # before this restore, so re-assert the shadow requirements here.
                cloud_masking = True
                keep_clouds = False
                shadow_masking = True
                _ndt = stac_parameters.get("nir_dark_threshold")
                if _ndt is not None:
                    nir_dark_threshold = float(_ndt)
                _spd = stac_parameters.get("shadow_proj_distance")
                if _spd is not None:
                    shadow_proj_distance = float(_spd)
                if aggregator:
                    raise ValueError(
                        "Cannot update a cloud-shadow-masked cube with an "
                        "aggregator: shadow projection needs the time dimension "
                        "and per-scene solar geometry."
                    )
                if not bands or "nir" not in [str(b).lower() for b in bands]:
                    raise ValueError(
                        "This cube was cloud-shadow-masked but its stored bands "
                        "lack 'nir', which shadow projection requires."
                    )
            elif cs.startswith("cloud_mask_"):
                # s2cloudless-masked cube. Masking is a POST-process (a separate
                # L1C download + detector), not part of the L2A query, so build
                # the new scenes UNMASKED here and mask them with s2cloudless
                # below, before the merge. The threshold is self-encoded in the
                # status name; the shadow params (if stored) mean s2cloudless +
                # shadow was applied.
                if add_bands:
                    raise ValueError(
                        "Band addition is not supported for s2cloudless-masked "
                        "cubes yet - update dates only (clear the band selection)."
                    )
                if aggregator:
                    raise ValueError(
                        "Cannot update an s2cloudless-masked cube with an "
                        "aggregator (masking needs the per-scene time dimension)."
                    )
                try:
                    _s2c_threshold = int(cs.rsplit("_", 1)[1])
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Cannot read the s2cloudless threshold from "
                        f"cloud_status={cs!r} (expected e.g. 'cloud_mask_50')."
                    )
                cloud_masking = False   # unmasked L2A build; masked below
                _s2c_update = True
                _spd = stac_parameters.get("shadow_proj_distance")
                _ndt = stac_parameters.get("nir_dark_threshold")
                _s2c_shadow = _spd is not None
                if _s2c_shadow:
                    _s2c_proj = float(_spd)
                    if _ndt is not None:
                        _s2c_nir_dark = float(_ndt)
                    if not bands or "nir" not in [str(b).lower() for b in bands]:
                        raise ValueError(
                            "This cube was s2cloudless + shadow masked but its "
                            "stored bands lack 'nir', which shadow needs."
                        )
            else:  # clouds_not_detected
                cloud_masking = False
        # NOTE: do NOT force output=update here anymore.
        # This allows update mode to return an in-memory updated cube when output=None.
    else:
        if not mission:
            raise ValueError("Error: Please select a mission.")
        # `polygon is None` rather than `not polygon`: a single-feature
        # GeoDataFrame (passed in by the multi-feature batch loop) has an
        # ambiguous truth value.
        if polygon is None:
            raise ValueError(
                "Error: Please select a polygon or bbox list with geographic coordinates."
            )

    # If projected coords are given, will transform to WGS84 coords
    # if not isinstance(polygon, list):
    #    polygon = proj_check(polygon)

    if resampling_method is None:
        resampling_method = "nearest"

    # Did the user explicitly request the classification layer as a band?
    # cloud_masking auto-appends it for internal use and normally drops it
    # after masking; an explicit request means "keep it in the cube" (e.g. so
    # shadow masking can reuse it without re-downloading). It is loaded with
    # "nearest" resampling regardless of resampling_method (see get_stac).
    _class_layer = {"sentinel_2_l2a": "scl", "landsat_c2_l2": "qa_pixel"}.get(mission)
    user_requested_scl = bool(bands) and _class_layer is not None and any(
        str(b).lower() == _class_layer for b in bands
    )
    # Shadow masking needs the SCL classes on the final (clipped) grid, so the
    # layer rides along through the pipeline and - unless the user explicitly
    # asked for it as a band - is dropped again right after the shadow step.
    keep_class_layer = user_requested_scl or bool(shadow_masking)

    stac, baselines, tiles = get_stac(
        mission, polygon, resolution, daterange, bands, max_cc, cloud_masking,
        source=source or "element84",
        resampling=resampling_method,
    )
    crs = stac.spatial_ref.projected_crs_name
    transform = stac.rio.transform()

    # In update mode, restrict the fresh query to the dates the existing cube
    # is MISSING before any masking / shadow / cloud% work runs. Those steps
    # read pixels eagerly (shadow especially), so without this they would
    # re-download and re-process every already-stored date only to discard it
    # in update_stac. Skipped when adding bands (band addition needs the stored
    # dates in the fresh query). Subsetting is safe for baselines (scale_factor
    # re-aligns via .sel(time=...)) and leaves the new dates' cloud_percentage
    # identical (a pixel is masked/observed per its own scene; the cross-scene
    # footprint never flips a new date's count).
    #
    # _has_new_dates=False means the cube is already up to date. The masking
    # passes below are then SKIPPED entirely: they would recompute stored dates
    # (s2cloudless would re-run the L1C detector on existing scenes) only for
    # update_stac to discard the result and return the cube unchanged.
    _has_new_dates = True
    if update and not add_bands:
        _existing_days = np.asarray(existing_times).astype("datetime64[D]")
        _fetched_days = stac["time"].values.astype("datetime64[D]")
        _is_new = ~np.isin(_fetched_days, _existing_days)
        _has_new_dates = bool(_is_new.any())
        if _has_new_dates and not _is_new.all():
            stac = stac.isel(time=np.nonzero(_is_new)[0])

    # Cloud masking
    cloud_bool = None
    mask_cube = None  # in-memory binary SCL mask (built lazily when requested)
    if cloud_masking is True:
        # keep_clouds=True -> pixels stay intact, only the cloud % is derived from
        # the returned per-pixel cloud boolean (cloud_bool). Default removes them.
        #
        # With shadow masking on, cloud removal is DEFERRED: GEE computes the
        # dark-pixel test on the full (unmasked) image, and dark pixels under
        # thin clouds legitimately support the shadow smoothing. Pixels stay
        # intact here and cloud AND shadow are masked together right after the
        # shadow step (verified bit-identical to get_shadow_layers).
        _defer_cloud_removal = bool(shadow_masking) and not bool(keep_clouds)
        stac, cloud_bool = cloud_mask(
            stac, mission,
            keep_clouds=bool(keep_clouds) or _defer_cloud_removal,
            keep_layer=keep_class_layer,
        )

    # Scale factor
    stac = scale_factor(stac, mission, baselines, source=source or "element84")
    # stac.rio.write_crs(crs, inplace=True)

    # Transform zeros to nan
    # stac = stac.where(stac != 0)

    # Index calculation
    # Add code when only indices are asked without band selection
    if indices:
        stac_indices = calculate_spectral_index(stac, mission, indices)

    # Add animation here
    # if animation is True:
    #    generate_animation(stac)

    #    if mission == 'cop_dem_glo_30':
    #        dem = stac.isel(time=0).dem
    #        dem = dem.expand_dims(dim={'band': ['dem']})
    #        stac_topo_features = calculate_topo(dem, topographic_features)

    # Dataset -> DataArray
    if mission != "cop_dem_glo_30":
        bands = list(stac.data_vars.keys())
        stac = xr.concat([stac[band] for band in bands], dim="band")
        stac = stac.assign_coords(band=bands)

    # DataArray manipulation
    if indices:
        stac = xr.concat([stac, stac_indices], dim="band")
        stac.attrs["indices"] = indices

    if mission == "cop_dem_glo_30":
        stac = xr.concat([dem, stac_topo_features], dim="band")
        stac = stac.rename("Topographic_Features")
    else:
        stac = stac.transpose("time", "band", "y", "x")
        stac = stac.rename("Spectral_Temporal_Stack")

    # Add metadata as attributes
    if not update:
        stac.attrs["spectral_bands"] = bands
        stac.attrs["mission"] = mission
        stac.attrs["resampling"] = resampling_method
        # SCL cloud strategy, recorded so update mode can reproduce the exact
        # same handling on newly added scenes (see get_stac_parameters and the
        # update block above). keep_clouds is None/False -> pixels were masked.
        if cloud_masking is True:
            if shadow_masking and not keep_clouds:
                # Cloud + shadow were projected and masked together. Store the
                # two tunable shadow params numerically so update mode can
                # reproduce the exact same projection on newly added scenes.
                stac.attrs["cloud_status"] = "scl_shadow_masked"
                stac.attrs["nir_dark_threshold"] = nir_dark_threshold
                stac.attrs["shadow_proj_distance"] = shadow_proj_distance
            else:
                stac.attrs["cloud_status"] = (
                    "clouds_detected" if keep_clouds else "scl_masked"
                )
        else:
            stac.attrs["cloud_status"] = "clouds_not_detected"
        _source_aliases = {"e84": "element84", "tb": "terrabyte", "pc": "planetary_computer"}
        stac.attrs["stac_api"] = _source_aliases.get(source or "element84", source or "element84")
        if mission in ("sentinel_2_l2a", "sentinel_2_l1c"):
            tile_list = np.array(tiles, dtype="U10").tolist()
            stac.attrs["tile_id"] = tile_list
        if isinstance(polygon, list):
            bbox = polygon
        else:
            bbox = polygon_2_bbox(polygon)
            # gdf = read_polygon_file(polygon) # update-clip raster
            # geom = list(gdf.iloc[0].geometry.exterior.coords) # update-clip raster
            # stac.attrs['geometry'] = geom # update-clip raster
        stac.attrs["bbox"] = bbox

    # Calculate stats image (optional)
    # Aggregator (optional): collapses time dimension
    if aggregator:
        if aggregator == "mean":
            stac = stac.mean(dim="time", skipna=True)
        elif aggregator == "median":
            stac = stac.median(dim="time", skipna=True)
        else:
            raise ValueError("Invalid aggregator. Please select either 'mean' or 'median'.")

    # Clip netcdf as clip raster
    if clip_raster:
        # if update: # update-clip raster
        # import geopandas as gpd # update-clip raster
        # from shapely.geometry import Polygon # update-clip raster
        # poly = Polygon(geometry) # update-clip raster
        # polygon = gpd.GeoDataFrame(index=[0], geometry=[poly]) # update-clip raster
        # polygon.set_crs(stac.crs, inplace=True) # update-clip raster
        stac = clip_stac(stac, polygon, crs)  # delete write_crs in clip_stac

    # Finalizing
    if not aggregator:
        stac["time"] = stac["time"].dt.floor("D")

        # _has_new_dates guard: in update mode with nothing new to add, these
        # passes (shadow projection, cloud%) read pixels eagerly and would
        # re-download + re-process the stored dates only for update_stac to
        # discard them and return the cube unchanged.
        if cloud_masking is True and _has_new_dates:
            # ---- Cloud shadow detection (GEE s2cloudless approach) ----------
            # Runs on the final (clipped, day-floored) grid: clouds are the SCL
            # cloud boolean, shadows are dark non-water pixels inside each
            # cloud's anti-solar projection (per-scene mean solar azimuth from
            # the STAC metadata). Reads the nir + scl pixels eagerly - the one
            # part of a lazy build that must compute now.
            shadow_bool = None
            cb_aligned = None
            if shadow_masking and cloud_bool is not None:
                # Imported at call time: shadow_masking imports cloud_masking,
                # which imports this module - a top-level import would be
                # circular at package load.
                from .shadow_masking import detect_shadow_stack, solar_azimuths_for_days

                if not q:
                    print("Detecting cloud shadows (reading nir + scl)...", flush=True)

                cb_aligned = cloud_bool.sel(y=stac["y"], x=stac["x"]).assign_coords(
                    time=stac["time"]
                )
                nir_np = np.asarray(
                    stac.sel(band="nir").transpose("time", "y", "x").values,
                    dtype="float32",
                )
                scl_np = np.nan_to_num(
                    stac.sel(band="scl").transpose("time", "y", "x").values, nan=0
                ).astype(np.int16)
                azimuths = solar_azimuths_for_days(
                    polygon, stac["time"].values, source or "element84"
                )
                res_m = float(abs(stac.y.values[1] - stac.y.values[0]))
                shadow_np = detect_shadow_stack(
                    cb_aligned.transpose("time", "y", "x").values,
                    nir_np,
                    scl_np,
                    azimuths,
                    res_m,
                    nir_dark_threshold=nir_dark_threshold,
                    proj_distance=shadow_proj_distance,
                )
                shadow_bool = xr.DataArray(
                    shadow_np.astype(bool),
                    dims=("time", "y", "x"),
                    coords={"time": stac["time"], "y": stac["y"], "x": stac["x"]},
                )
                if not keep_clouds:
                    # Deferred cloud removal (see cloud_mask call above): cloud
                    # and shadow pixels are masked out together, GEE-style.
                    stac = stac.where(~(cb_aligned | shadow_bool))
                stac.attrs["shadow_params"] = (
                    f"cloud_source=scl, nir_dark_threshold={nir_dark_threshold}, "
                    f"proj_distance_km={shadow_proj_distance}"
                )

            # Cloud % is measured against the observable AOI footprint: pixels
            # missing in every scene (incl. anything outside a non-rectangular
            # clip) are excluded from both numerator and denominator, so only
            # real clouds count. The count always comes from the SCL/QA cloud
            # boolean (not from NaN holes) so that per-date swath/tile gaps -
            # genuine NO_DATA, not cloud - are never miscounted as cloud. This
            # holds in both remove-clouds and keep-clouds mode. With shadow
            # masking on, the percentage counts cloud OR shadow (the flagged,
            # unusable fraction).
            _pct_mask = cloud_bool
            if shadow_bool is not None:
                _pct_mask = cb_aligned | shadow_bool
            pct = compute_cloud_percentage(stac, cloud_mask=_pct_mask)
            if pct is not None:
                stac = stac.assign_coords(
                    cloud_percentage=("time", np.asarray(pct.data))
                )

            # Binary SCL cloud-mask time series (1=cloud, 0=clear). Built when
            # either a path is given (write it now - SLURM / NetCDF-during-build)
            # or the caller wants it in memory (return_cloud_mask - the GUI holds
            # it and writes it on export). This is what lets a kept-clouds cube be
            # masked / filtered / co-registered later. With shadow masking on,
            # shadow_mask and cloudshadow_mask bands are appended so the file
            # follows the Cloud_Stack convention of get_shadow_layers.
            if cloud_bool is not None and (cloud_mask_output or return_cloud_mask):
                mask_cube = build_scl_mask_cube(stac, cloud_bool)
                if shadow_bool is not None:
                    _sh = (
                        shadow_bool.astype("uint8")
                        .expand_dims(band=["shadow_mask"])
                        .transpose("time", "band", "y", "x")
                    )
                    _comb = (
                        (cb_aligned | shadow_bool)
                        .astype("uint8")
                        .expand_dims(band=["cloudshadow_mask"])
                        .transpose("time", "band", "y", "x")
                    )
                    # coords="minimal": non-band coords (spatial_ref,
                    # cloud_percentage) exist only on the SCL band's cube and
                    # are carried over instead of failing the concat.
                    mask_cube = xr.concat(
                        [mask_cube, _sh, _comb], dim="band", coords="minimal"
                    )
                    mask_cube.name = "Cloud_Stack"
                # Self-contained georeferencing so it can be exported later.
                mask_cube.attrs["crs"] = crs
                mask_cube.attrs["transform"] = stac.rio.transform()
                if cloud_mask_output:
                    _cmo_dir = os.path.dirname(cloud_mask_output)
                    if _cmo_dir:
                        os.makedirs(_cmo_dir, exist_ok=True)
                    export_stac(
                        mask_cube, cloud_mask_output, crs=crs,
                        transform=stac.rio.transform(), var_name="Cloud_Stack",
                        compress=compress,
                    )
                    if not q:
                        print(f"Binary cloud mask exported: {cloud_mask_output}", flush=True)

            # The scl band rode along only for the shadow step - drop it again
            # unless the user explicitly requested it, and keep the band-list
            # attr truthful either way.
            if (
                shadow_masking
                and not user_requested_scl
                and "band" in stac.dims
                and "scl" in [str(b) for b in stac.band.values]
            ):
                stac = stac.sel(
                    band=[b for b in stac.band.values if str(b) != "scl"]
                )
                stac.attrs["spectral_bands"] = [
                    b for b in stac.attrs.get("spectral_bands", []) if b != "scl"
                ]

    stac.attrs["crs"] = crs
    stac.attrs["transform"] = transform

    # stac = stac.copy()
    stac.attrs.pop("nodata", None)
    try:
        stac = stac.rio.write_nodata(None, inplace=True)
    except Exception:
        pass

    # Update existing cube by integrating only missing dates (optional)
    # Done BEFORE export branching so update can also return in-memory result
    # when output=None.
    if update:
        if _s2c_update and _has_new_dates:
            # Reproduce s2cloudless masking on the freshly built (unmasked) new
            # scenes, then merge. The masking helpers read these attrs, which the
            # update path does not otherwise set (the metadata block is fresh-
            # build only).
            stac.attrs.setdefault("mission", mission)
            stac.attrs["spectral_bands"] = bands
            stac.attrs["bbox"] = (
                polygon if isinstance(polygon, list) else polygon_2_bbox(polygon)
            )
            stac.attrs["crs"] = crs
            stac.attrs["transform"] = transform
            stac = _mask_new_scenes_s2cloudless(
                stac,
                threshold=_s2c_threshold,
                shadow=_s2c_shadow,
                nir_dark_threshold=_s2c_nir_dark,
                proj_distance=_s2c_proj,
            )
        stac = update_stac(stac_existing=update, stac_updated=stac, new_bands=add_bands)

        # Re-attach CRS/transform metadata explicitly (safe after concat/update)
        stac.attrs["crs"] = crs
        stac.attrs["transform"] = transform
        try:
            stac = stac.rio.write_crs(crs, inplace=True)
            stac = stac.rio.write_transform(transform, inplace=True)
        except Exception:
            pass

    if not output:
        stac.rio.write_crs(crs, inplace=True)
        stac.rio.write_transform(transform, inplace=True)
        stac.attrs["crs"] = crs
        stac.attrs["transform"] = transform

        # Optional: add temporal composites/statistics (kept lazy; no computation triggered)
        if stats and (mission != "cop_dem_glo_30") and (not aggregator):
            base_attrs = dict(stac.attrs)
            stac = calculate_statistics(stac, stats)
            stac.attrs.update(base_attrs)
            try:
                stac.rio.write_crs(crs, inplace=True)
                stac.rio.write_transform(transform, inplace=True)
            except Exception:
                pass

        if not q:
            print(stac, flush=True)
        if return_cloud_mask:
            return stac, mask_cube
        return stac  # returns lazy (update mode may compute missing slices internally)

    else:
        # Optional stats/composites (only when time dimension exists)
        if stats and (mission != "cop_dem_glo_30") and (not aggregator):
            stac = calculate_statistics(stac, stats)

        # One consistent debug print for ALL cases (agg on/off, stats on/off)
        if not q:
            print(f"\nExporting to: {output}")
            print(f"  aggregator: {aggregator if aggregator else 'None'}")

            if stats:
                if (mission == "cop_dem_glo_30") or aggregator:
                    print("  stats: ignored (requires time dimension)")
                else:
                    print(f"  stats: {stats}")
            else:
                print("  stats: None")

            print(stac, flush=True)

        img = export_stac(stac, output, crs, transform, compress=compress)
        if return_cloud_mask:
            return img, mask_cube
        return img