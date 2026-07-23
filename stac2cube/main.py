import os

from .get_data import (
    get_stac,
    crs_attr_string,
    validate_target_crs,
    _S2_BNUM_TO_COMMON,
    _SCENE_METADATA_FIELDS,
    export_granule_metadata,
)
from .vector_refiner import (
    proj_check,
    polygon_2_bbox,
    read_polygon_file,
    polygon_2_features,
    polygon_2_gdf,
)
from .stac_processing import scale_factor, cloud_mask, build_scl_mask_cube
from .get_spectral_indices import calculate_spectral_index
from .export_cfg import export_stac, export_to_cogs

# from .get_topo import calculate_topo
# from .time_series_tools import generate_animation
from .clip import (
    clip_stac,
    compute_cloud_percentage,
    compute_scene_coverage,
    compute_scene_coverage_from_imaged,
    drop_partial_scenes,
    _aoi_mask_from_geometries,
)
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


_EXPORT_FORMATS = ("netcdf", "zarr", "cogs")


def _validate_export_format(output, export_format):
    """Normalise and check ``export_format`` against the output path.

    ``None`` (default) keeps the long-standing extension dispatch of
    export_stac ('.zarr' -> Zarr store, anything else -> NetCDF). An explicit
    value is checked against the path, so a mismatch is an error instead of a
    surprise file in the wrong format. 'cogs' makes ``output`` a FOLDER.
    """
    if export_format is None:
        return None

    fmt = str(export_format).strip().lower()
    if fmt not in _EXPORT_FORMATS:
        raise ValueError(
            "export_format must be 'netcdf', 'zarr' or 'cogs' (or None to infer "
            f"it from the output extension), got {export_format!r}."
        )
    if not output:
        raise ValueError(
            f"export_format='{fmt}' needs an output "
            + ("folder for the GeoTIFFs." if fmt == "cogs" else "file path.")
        )

    is_zarr_path = str(output).lower().endswith(".zarr")
    if fmt == "zarr" and not is_zarr_path:
        raise ValueError(
            "export_format='zarr' needs an output path ending in '.zarr' "
            f"(got {output!r})."
        )
    if fmt == "netcdf" and is_zarr_path:
        raise ValueError(
            "export_format='netcdf' cannot write to a '.zarr' path "
            f"(got {output!r}). Use export_format='zarr' or a '.nc' path."
        )
    return fmt


def _export_result(
    stac, output, export_format, crs=None, transform=None, compress=False
):
    """Write the finished cube, dispatching on ``export_format``.

    'cogs' -> one Cloud-Optimized GeoTIFF per timestep into the output FOLDER
    (export_to_cogs, which handles both a DataArray and a stats Dataset).
    Anything else -> export_stac, whose own extension dispatch picks NetCDF vs
    Zarr. Returns what was written / handed over, so callers can keep chaining.
    """
    if export_format == "cogs":
        os.makedirs(output, exist_ok=True)
        export_to_cogs(stac, output_dir=output, prefix="", dtype="float32")
        return stac
    return export_stac(stac, output, crs, transform, compress=compress)


def _normalize_dates(dates):
    """The ``dates`` keep-list as a set of ISO timestamp strings.

    Accepts what the GUI Result date picker holds (``str(numpy.datetime64)``,
    e.g. '2024-04-01T00:00:00.000000000') as well as plain 'YYYY-MM-DD' and
    datetime-like objects; everything is normalised through numpy so both
    spellings match the same scene.
    """
    if dates is None:
        return None
    if isinstance(dates, (str, bytes)):
        dates = [dates]
    try:
        wanted = list(dates)
    except TypeError:
        raise ValueError(
            "dates must be a list of acquisition timestamps (e.g. "
            "['2024-04-01T00:00:00.000000000', ...])."
        )
    if not wanted:
        raise ValueError(
            "dates is empty: that would keep no scene at all. Pass None to keep "
            "every date."
        )
    out = set()
    for d in wanted:
        try:
            out.add(str(np.datetime64(d, "ns")))
        except Exception:
            raise ValueError(
                f"dates entry {d!r} is not a valid acquisition timestamp "
                "(expected e.g. '2024-04-01' or "
                "'2024-04-01T00:00:00.000000000')."
            )
    return out


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
    scene_cloud_coverage=None,
    clip_raster=None,
    cloud_masking=None,
    keep_clouds=None,
    shadow_masking=None,
    nir_dark_threshold=0.18,
    shadow_proj_distance=1.0,
    cloud_mask_output=None,
    return_cloud_mask=False,
    indices=None,
    dates=None,
    output=None,
    export_format=None,
    aggregator=None,
    stats=None,
    topographic_features=None,
    animation=None,
    update=None,
    source=None,
    resampling_method=None,
    scene_metadata=None,
    metadata_output=None,
    crs=None,
    tile_handling="mosaic",
    partial_scene_handling="keep",
    min_scene_coverage=0.9,
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

    # --- Scene-level cloud coverage filter preconditions ----------------------
    # scene_cloud_coverage is the headless twin of the GUI Result panel's
    # "Max cloud %" box: after the cube is built, only scenes whose
    # cloud_percentage is <= the threshold are kept (and exported). Values
    # >= 100 pass everything through, exactly like the GUI box.
    if scene_cloud_coverage is not None:
        try:
            _scc = float(scene_cloud_coverage)
        except (TypeError, ValueError):
            raise ValueError(
                "scene_cloud_coverage must be a number between 0 and 100."
            )
        if not (0.0 <= _scc <= 100.0):
            raise ValueError(
                "scene_cloud_coverage must be between 0 and 100 (percent)."
            )
        if update:
            raise ValueError(
                "scene_cloud_coverage is not supported in update mode: it would "
                "permanently drop already-stored dates from the existing cube. "
                "Update first, then filter with cloud_masking.cloud_filter."
            )
        if cloud_masking is not True:
            raise ValueError(
                "scene_cloud_coverage requires cloud_masking=True: the per-scene "
                "cloud_percentage it filters on is derived from the detected "
                "clouds (keep_clouds=True works too - detection without masking)."
            )
    # --- Explicit date selection preconditions --------------------------------
    # dates is the headless twin of the GUI Result panel's date picker: after
    # the cube is built, only the listed acquisition timestamps are kept. It is
    # a pure time selection on the finished cube, so it is single-cube only -
    # the picker itself is disabled for multi-feature batches (rejected in the
    # batching branch below).
    _dates_keep = _normalize_dates(dates)
    if _dates_keep is not None and update:
        raise ValueError(
            "dates is not supported in update mode: it would permanently drop "
            "already-stored dates from the existing cube. Update first, then "
            "slice the cube with the Data Cube Editor."
        )

    # --- Export format --------------------------------------------------------
    _export_format = _validate_export_format(output, export_format)

    # With a temporal aggregator, the composite must describe the SURVIVING
    # scenes (GUI order: build -> filter -> composite), so the collapse is
    # deferred until after the scene filter instead of running at build time.
    # All three scene filters (cloud %, partial-coverage removal, explicit date
    # selection) trigger the deferral so the composite only ever averages the
    # kept scenes.
    _scc_active = scene_cloud_coverage is not None and float(scene_cloud_coverage) < 100.0
    _remove_partial = partial_scene_handling == "remove"
    _defer_agg = bool(aggregator) and (
        _scc_active or _remove_partial or _dates_keep is not None
    )

    # --- Scene-level metadata coordinate preconditions -------------------------
    # scene_metadata attaches per-scene STAC item properties (solar/viewing
    # geometry, orbit, baseline, full acquisition timestamp) as (time,)
    # coordinates - see _SCENE_METADATA_FIELDS in get_data.py for the canonical
    # names and per-source availability.
    if scene_metadata:
        if isinstance(scene_metadata, str):
            scene_metadata = [scene_metadata]
        _unknown = [
            str(f) for f in scene_metadata
            if str(f) not in _SCENE_METADATA_FIELDS
        ]
        if _unknown:
            raise ValueError(
                f"Unknown scene_metadata field(s) {_unknown}. "
                f"Valid options: {_SCENE_METADATA_FIELDS}."
            )
        if update:
            raise ValueError(
                "scene_metadata cannot be changed in update mode: the cube's "
                "stored selection is restored automatically so the new scenes "
                "carry the same coordinates."
            )
        if mission not in ("sentinel_2_l2a", "sentinel_2_l1c"):
            raise ValueError(
                "scene_metadata is available for Sentinel-2 cubes only."
            )
        if aggregator:
            raise ValueError(
                "scene_metadata cannot be combined with an aggregator: the "
                "composite collapses the time dimension the per-scene "
                "coordinates live on."
            )

    # --- Granule metadata XML export preconditions ------------------------------
    # metadata_output is a directory: the granule metadata XML (MTD_TL.xml) of
    # every scene in the finished cube is downloaded there, one file per
    # granule (see export_granule_metadata). Independent of scene_metadata.
    if metadata_output:
        if aggregator:
            raise ValueError(
                "metadata_output cannot be combined with an aggregator: the "
                "composite has no per-scene time axis to match XMLs against."
            )
        # In update mode the mission is only known after the restore; the
        # helper itself rejects non-Sentinel-2 cubes at call time.
        if not update and mission not in ("sentinel_2_l2a", "sentinel_2_l1c"):
            raise ValueError(
                "metadata_output (granule metadata XML export) is available "
                "for Sentinel-2 cubes only."
            )

    # --- Target projection preconditions --------------------------------------
    # crs= overrides the automatic choice (the projection natively covering most
    # of the area - see _choose_target_crs). Validated up front so a bad CRS
    # fails before any query, and normalised to "EPSG:<code>" so the cube's attrs
    # and the update-mode pin agree on one spelling.
    if crs is not None:
        if update:
            raise ValueError(
                "crs cannot be changed in update mode: the cube's own projection "
                "is restored so the new dates land on the existing grid. Rebuild "
                "in one pass to change the projection."
            )
        crs = validate_target_crs(crs)

    # --- Tile handling preconditions -----------------------------------------
    # tile_handling="separate" keeps AOI-straddling N-S Sentinel-2 tiles as
    # distinct timesteps instead of mosaicing them into one (see get_stac). It
    # keeps full-precision, per-tile acquisition times so two tiles of the same
    # solar day stay unique - which the day-floored update/date logic does not
    # expect, and which the time-collapsing aggregator would erase.
    if tile_handling not in ("mosaic", "separate"):
        raise ValueError(
            "tile_handling must be 'mosaic' (default, current behaviour) or "
            "'separate'."
        )
    if tile_handling == "separate":
        if mission != "sentinel_2_l2a":
            raise ValueError(
                "tile_handling='separate' is available for Sentinel-2 L2A "
                "only (L1C keeps one item per solar day via its processing-"
                "baseline dedup, which would defeat tile separation)."
            )
        if update:
            raise ValueError(
                "tile_handling='separate' is not supported in update mode: it "
                "keeps sub-day per-tile timestamps that the day-precision "
                "update matching does not handle. Build the separated cube in "
                "one pass over the full date range instead."
            )
        if aggregator:
            raise ValueError(
                "tile_handling='separate' cannot be combined with an "
                "aggregator: the composite collapses the time dimension the "
                "separated per-tile scenes live on."
            )

    # --- Partial-scene removal preconditions ----------------------------------
    # partial_scene_handling="remove" drops scenes that image only PART of the
    # AOI (across-track / swath-edge coverage gaps, which load as NaN), keeping
    # only scenes whose coverage of the AOI footprint is >= min_scene_coverage.
    # "keep" (default) is the current behaviour: no filtering.
    if partial_scene_handling not in ("keep", "remove"):
        raise ValueError(
            "partial_scene_handling must be 'keep' (default) or 'remove'."
        )
    _remove_partial = partial_scene_handling == "remove"
    if _remove_partial:
        try:
            _msc = float(min_scene_coverage)
        except (TypeError, ValueError):
            raise ValueError("min_scene_coverage must be a fraction between 0 and 1.")
        if not (0.0 <= _msc <= 1.0):
            raise ValueError(
                "min_scene_coverage must be a fraction in [0, 1] (e.g. 0.9 = a "
                "scene must image at least 90% of the AOI to be kept; 0 keeps "
                "everything)."
            )
        if update:
            raise ValueError(
                "partial_scene_handling='remove' is not supported in update "
                "mode: it would permanently drop already-stored partial scenes. "
                "Build the filtered cube in one pass instead."
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
            # Date selection is a single-cube control (the GUI picker is
            # disabled for batches): every feature has its own time axis, so
            # one shared keep-list cannot describe them all.
            if _dates_keep is not None:
                raise ValueError(
                    f"dates is not supported for batch processing: this polygon "
                    f"file holds {n} features and each cube gets its own time "
                    "axis. Build one feature at a time to select dates."
                )
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

                # One granule-metadata folder per feature (<dir>_<idx>).
                mdo_i = f"{metadata_output}_{idx}" if metadata_output else None

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
                    scene_cloud_coverage=scene_cloud_coverage,
                    clip_raster=clip_raster,
                    cloud_masking=cloud_masking,
                    keep_clouds=keep_clouds,
                    shadow_masking=shadow_masking,
                    nir_dark_threshold=nir_dark_threshold,
                    shadow_proj_distance=shadow_proj_distance,
                    cloud_mask_output=cmo_i,
                    return_cloud_mask=return_cloud_mask,
                    indices=list(indices) if indices else indices,
                    dates=None,  # rejected above for batches
                    output=None,  # export below, per feature, so RAM frees each time
                    export_format=None,  # nothing is written here; see below
                    aggregator=aggregator,
                    stats=stats,
                    topographic_features=topographic_features,
                    animation=animation,
                    update=update,
                    source=source,
                    resampling_method=resampling_method,
                    crs=crs,
                    scene_metadata=(
                        list(scene_metadata) if scene_metadata else scene_metadata
                    ),
                    metadata_output=mdo_i,
                    tile_handling=tile_handling,
                    partial_scene_handling=partial_scene_handling,
                    min_scene_coverage=min_scene_coverage,
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
                    if _export_format == "cogs":
                        # COGs are a FOLDER of per-date GeoTIFFs, so each
                        # feature gets its own subfolder (<folder>/1, /2, ...)
                        # instead of a <stem>_<idx> filename - matching the
                        # GUI's multi-feature COG export.
                        feature_output = os.path.join(output, str(idx))
                    else:
                        stem, ext = os.path.splitext(output)
                        feature_output = f"{stem}_{idx}{ext}"
                    _export_result(
                        res, feature_output, _export_format, compress=compress
                    )

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

    # Target projection handed to get_stac: the user's crs= on a fresh build, the
    # cube's stored projection in update mode (see the restore below), or None to
    # let get_stac choose it from the matched items.
    _pinned_crs = crs

    if update:
        stac_parameters = get_stac_parameters(update)
        existing_times = stac_parameters["times"]

        # A separate-tile cube carries sub-day per-tile timestamps and a `tile`
        # coordinate that the day-precision update matching cannot reconcile;
        # refuse rather than silently corrupt its time axis.
        if stac_parameters.get("tile_handling") == "separate":
            raise ValueError(
                "This cube was built with tile_handling='separate' (per-tile "
                "timesteps) and cannot be updated: rebuild it in one pass over "
                "the extended date range instead."
            )

        mission = stac_parameters["mission"]
        resolution = stac_parameters["resolution"]
        polygon = stac_parameters["polygon"]
        # Pin the cube's OWN projection so the new scenes land on the existing
        # grid. Without this the fresh query re-derives a target CRS, and any
        # change to that choice - a different selection rule, or a multi-CRS area
        # whose ranking shifted - would put the new dates on a different grid and
        # misalign the concat in update_stac. Legacy cubes store a CRS *name*
        # here; crs_attr_string normalises it (see _choose_target_crs).
        _pinned_crs = stac_parameters.get("crs")
        # geometry = stac_parameters["geometry"] # update-clip raster
        user_bands = bands  # bands passed alongside update = bands to ADD
        bands = stac_parameters["spectral_bands"]
        if not isinstance(bands, list):
            # NetCDF stores the attr as an ndarray, Zarr returns a list. A
            # SINGLE-band cube arrives as a scalar string, and tolist() on the
            # 0-d array hands the string back unchanged - which would then be
            # iterated per character ("red" -> "r","e","d"). ravel() makes it
            # a 1-element list first.
            bands = np.asarray(bands).ravel().tolist()
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
        # Restore the cube's scene-metadata coordinate selection so the fresh
        # query builds the SAME (time,) coords on the new scenes - without
        # this the time concat in update_stac fails on the coord mismatch.
        # (The precondition block rejected any user-passed scene_metadata.)
        _sm = stac_parameters.get("scene_metadata")
        if _sm is not None:
            # NetCDF attrs arrive as ndarray (or scalar str for one field),
            # Zarr JSON attrs as list - normalize like the indices attr above.
            scene_metadata = [str(f) for f in np.asarray(_sm).ravel().tolist()]
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
        scene_metadata=scene_metadata,
        tile_handling=tile_handling,
        crs=_pinned_crs,
        q=bool(q),
    )
    # Separate-tile cubes keep full-precision, per-tile timestamps so two tiles
    # of one solar day stay unique; the day-floor below is skipped for them.
    _separate_tiles = (tile_handling == "separate") and (
        "tile" in getattr(stac, "coords", {})
    )
    # On how many solar days the scene-metadata values were reduced from >1
    # STAC item (tile overlap). Grabbed now because the Dataset attrs are lost
    # in the band concat below; re-attached in the metadata block.
    _meta_multiday = stac.attrs.get("scene_metadata_multiday")
    # Target projection chosen in get_stac from the native CRSs of every matched
    # item. Stored as "EPSG:<code>" (see crs_attr_string): the cube's attrs used
    # to hold spatial_ref.projected_crs_name, a display NAME that only round-trips
    # for CRSs in the PROJ name database and does not exist at all for geographic
    # CRSs. Falling back to the loaded grid keeps cubes buildable if a source ever
    # publishes no proj:code.
    crs = stac.attrs.get("target_crs") or crs_attr_string(stac.rio.crs)
    # Native CRSs when something had to be reprojected, with the share of the AOI
    # each one natively covers (provenance for the finished cube; both absent on
    # the normal case of one native projection that the cube also uses).
    _native_crs = stac.attrs.get("native_crs")
    _native_crs_share = stac.attrs.get("native_crs_share")
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
    imaged_bool = None  # per-pixel "was imaged" (SCL/QA != no-data); scene-coverage source
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
        stac, cloud_bool, imaged_bool = cloud_mask(
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
        # How AOI-straddling tiles were handled, recorded for provenance and
        # so a separated cube is self-describing (the `tile` coordinate is the
        # per-timestep identity).
        stac.attrs["tile_handling"] = tile_handling
        # Partial-scene handling (across-track coverage). Recorded for
        # provenance; the scene_coverage coordinate (attached to every freshly
        # built cube above) is the per-scene AOI coverage fraction.
        stac.attrs["partial_scene_handling"] = partial_scene_handling
        if _remove_partial:
            stac.attrs["min_scene_coverage"] = float(min_scene_coverage)
        if scene_metadata:
            # Recorded so update mode can rebuild the same coords on newly
            # added scenes (see get_stac_parameters and the restore above).
            stac.attrs["scene_metadata"] = [str(f) for f in scene_metadata]
            if _meta_multiday is not None:
                stac.attrs["scene_metadata_multiday"] = int(_meta_multiday)
                if int(_meta_multiday) > 0 and not q:
                    print(
                        f"Note: on {int(_meta_multiday)} date(s) the polygon "
                        "is covered by more than one Sentinel-2 granule "
                        "(tile overlap): angle metadata is the per-date mean, "
                        "acq_datetime the earliest acquisition.",
                        flush=True,
                    )
        if mission in ("sentinel_2_l2a", "sentinel_2_l1c"):
            tile_list = np.array(tiles, dtype="U10").tolist()
            stac.attrs["tile_id"] = tile_list
        # Multi-projection provenance: only present when the area's tiles are NOT
        # all in the cube's own CRS, i.e. when some scenes had to be reprojected.
        # Absent on an ordinary single-zone cube.
        if _native_crs:
            stac.attrs["native_crs"] = [str(c) for c in _native_crs]
            # Fraction (0..1) of the AOI each native projection's TILES span.
            # Not to be confused with the scene_coverage coordinate, which is
            # per-timestep and says how much of the AOI one scene imaged.
            if _native_crs_share is not None:
                stac.attrs["native_crs_share"] = [
                    float(s) for s in np.asarray(_native_crs_share).ravel()
                ]
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
    # (deferred when a scene_cloud_coverage filter is active - see _defer_agg)
    if aggregator and not _defer_agg:
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
    if not aggregator or _defer_agg:
        # Separate-tile mode intentionally keeps the full-precision acquisition
        # time so two tiles of the same solar day remain distinct timesteps;
        # flooring them to the day would collide into a duplicate index that
        # the downstream .sel(time=...) / set-based logic cannot handle.
        if not _separate_tiles:
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

            # AOI mask so a non-rectangular polygon clip is respected by both
            # metrics (a plain bbox needs none - the whole grid is the AOI). It
            # also keeps outside-polygon clouds out of the cloud-% footprint.
            _aoi_mask = None
            if not (isinstance(polygon, (list, tuple)) and len(polygon) == 4):
                try:
                    _pproj = polygon_2_gdf(polygon).to_crs(stac.rio.crs)
                    _aoi_mask = _aoi_mask_from_geometries(
                        stac, _pproj.geometry.values
                    )
                except Exception:
                    _aoi_mask = None

            # Cloud % and scene coverage both reduce the SAME SCL/QA read, so
            # materialize them together: dask dedups the shared read and the SCL
            # file is fetched once per date (not twice, and NOT the heavier 10 m
            # blue band the coverage used to read). Coverage uses the SCL
            # "imaged" boolean - reliable even when a swath gap loads as 0 rather
            # than NaN, and cloud-aware for free - so it becomes a ready (eager)
            # scene_coverage coord and the GUI warning needs no further read.
            pct_lazy = compute_cloud_percentage(
                stac, aoi_mask=_aoi_mask, cloud_mask=_pct_mask, lazy=True
            )
            cov_lazy = (
                compute_scene_coverage_from_imaged(
                    stac, imaged_bool, aoi_mask=_aoi_mask
                )
                if imaged_bool is not None
                else None
            )
            import dask
            if cov_lazy is not None:
                pct_c, cov_c = dask.compute(pct_lazy, cov_lazy)
            else:
                (pct_c,) = dask.compute(pct_lazy)
                cov_c = None
            if pct_c is not None:
                stac = stac.assign_coords(
                    cloud_percentage=("time", np.asarray(pct_c.data))
                )
            # scene_coverage attached here (eagerly, from SCL) when cloud
            # detection ran; the lazy band-0 fallback below is skipped then.
            if cov_c is not None:
                stac = stac.assign_coords(
                    scene_coverage=("time", np.asarray(cov_c.data, dtype="float64"))
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

        # ---- Per-scene AOI coverage (default coordinate) --------------------
        # Attach scene_coverage (time,) to EVERY freshly built cube: the
        # fraction (0..1) of the AOI footprint each scene actually images
        # (across-track / swath completeness). Making it a default coord lets a
        # cube be inspected or filtered by coverage later without a rebuild.
        #
        # When cloud detection ran, scene_coverage was ALREADY attached above -
        # eagerly, from the SCL "imaged" boolean, sharing the cloud-% read (the
        # common, preferred path). This block is only the FALLBACK for when there
        # was no cloud detection (no SCL): it attaches a LAZY band-0 coverage so a
        # build that never reads it triggers no extra band read; it materializes
        # only on first use (export, a coverage filter) or when partial-scene
        # removal below reads it. Skipped in update mode, where the newly built
        # dates are concatenated onto an existing cube that may predate the coord.
        _scene_cov = None
        if not update and "scene_coverage" not in stac.coords:
            try:
                _scene_cov = compute_scene_coverage(
                    stac, cloud_mask=cloud_bool, compute=False
                )
            except Exception:
                _scene_cov = None
            if _scene_cov is not None:
                stac = stac.assign_coords(
                    scene_coverage=("time", _scene_cov.data)
                )

        # ---- Partial-scene removal (across-track / swath-edge coverage) -----
        # Drop scenes that image only PART of the AOI (the missing part loads
        # as NaN), keeping those covering >= min_scene_coverage of the AOI
        # footprint and keeping the scene_coverage (time,) coordinate.
        #
        # Runs BEFORE the cloud % filter: at this point no scene has been
        # dropped yet, so cloud_bool is still 1:1 with the time axis (the
        # cloud-aware alignment relies on it) and the footprint is measured over
        # every scene. Both filters are commutative AND-selections on time, so
        # the final surviving set is the same either way, and the deferred
        # composite still runs after both.
        #
        # cloud_bool (when cloud detection ran) makes coverage cloud-aware: on a
        # cloud-MASKED cube the masked clouds are NaN, so without it a fully-
        # imaged but cloudy scene would look partial and be wrongly dropped.
        # Passing the cloud boolean counts cloud-flagged pixels as imaged, so
        # only genuine swath/orbit no-data reduces coverage. None (no detection)
        # -> plain NaN coverage, correct for an unmasked cube.
        if _remove_partial:
            # Reuse the scene_coverage already attached above (SCL-based and
            # eager when cloud detection ran, else the lazy band-0 fallback in
            # _scene_cov) so removal never re-measures coverage.
            _cov_for_drop = (
                stac["scene_coverage"]
                if "scene_coverage" in stac.coords
                else _scene_cov
            )
            stac = drop_partial_scenes(
                stac,
                min_coverage=float(min_scene_coverage),
                cloud_mask=cloud_bool,
                q=q,
                coverage=_cov_for_drop,  # reuse the read above, don't re-measure
            )

        # ---- Scene-level cloud filter (headless "Max cloud %") --------------
        # Same semantics as the GUI Result panel box: keep only timesteps with
        # cloud_percentage <= threshold, via positional time selection (isel),
        # so nothing is computed and the data stays lazy. thr >= 100 was
        # already turned into a no-op by the precondition block. The binary
        # cloud-mask file (cloud_mask_output), written above, keeps ALL dates -
        # matching the GUI, whose held mask is also exported unfiltered.
        if scene_cloud_coverage is not None and float(scene_cloud_coverage) < 100.0:
            if "cloud_percentage" not in stac.coords:
                raise ValueError(
                    "scene_cloud_coverage was requested but the built cube "
                    "carries no cloud_percentage coordinate (cloud detection "
                    "yielded no per-scene percentage for this mission/setup)."
                )
            _thr = float(scene_cloud_coverage)
            _keep = np.asarray((stac["cloud_percentage"] <= _thr).values)
            if not _keep.any():
                _mn = float(np.nanmin(np.asarray(stac["cloud_percentage"].values)))
                raise ValueError(
                    f"scene_cloud_coverage={_thr:g}% keeps no scenes: the least "
                    f"cloudy scene has {_mn:.1f}% cloud cover. Raise the "
                    "threshold (or widen the daterange) and rerun."
                )
            if not _keep.all():
                _n_total = int(_keep.size)
                stac = stac.isel(time=np.flatnonzero(_keep))
                if not q:
                    print(
                        f"scene_cloud_coverage={_thr:g}%: kept "
                        f"{int(_keep.sum())}/{_n_total} scenes.",
                        flush=True,
                    )

        # ---- Explicit date selection (headless "Date Selection") ------------
        # Same semantics as the GUI Result panel's date picker: keep only the
        # listed acquisition timestamps, by positional selection (isel) so
        # nothing is computed and the cube stays lazy. Matching is on the full
        # ISO timestamp string, exactly as the picker does, so it is
        # unambiguous when two scenes share a calendar day (separate tiles) and
        # survives the cloud filter's isel above. Runs AFTER both scene filters
        # and BEFORE the deferred composite / stats, reproducing the GUI order
        # build -> filter -> composite.
        if _dates_keep is not None:
            if "time" not in getattr(stac, "dims", ()):
                raise ValueError(
                    "dates was requested but the built cube has no time "
                    "dimension to select from."
                )
            _tvals = np.asarray(stac["time"].values)
            _keep_d = np.array(
                [str(np.datetime64(t, "ns")) in _dates_keep for t in _tvals]
            )
            if not _keep_d.any():
                raise ValueError(
                    "dates keeps no scene: none of the requested timestamps is "
                    "in the built cube. The cube carries "
                    f"{len(_tvals)} date(s) between "
                    f"{np.datetime_as_string(_tvals.min(), unit='D')} and "
                    f"{np.datetime_as_string(_tvals.max(), unit='D')}."
                )
            _missing = len(_dates_keep) - int(_keep_d.sum())
            if not _keep_d.all():
                stac = stac.isel(time=np.flatnonzero(_keep_d))
            if not q:
                print(
                    f"dates: kept {int(_keep_d.sum())}/{len(_tvals)} scenes"
                    + (
                        f" ({_missing} requested date(s) not in this build)"
                        if _missing > 0
                        else ""
                    ),
                    flush=True,
                )

        # Deferred temporal composite: collapse the SURVIVING scenes (GUI
        # order build -> filter -> composite). keep_attrs preserves the cube
        # metadata through the reduction, as the GUI composite does.
        if _defer_agg:
            if aggregator == "mean":
                stac = stac.mean(dim="time", skipna=True, keep_attrs=True)
            elif aggregator == "median":
                stac = stac.median(dim="time", skipna=True, keep_attrs=True)
            else:
                raise ValueError(
                    "Invalid aggregator. Please select either 'mean' or 'median'."
                )

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

    # Granule metadata XML download (optional). Runs on the FINISHED cube
    # (after scene filters / update merge) so the XMLs match exactly the dates
    # the cube carries. Item-metadata search + small XML downloads only - no
    # pixel compute is triggered.
    if metadata_output:
        export_granule_metadata(stac, metadata_output, q=bool(q))

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

        img = _export_result(
            stac, output, _export_format, crs, transform, compress=compress
        )
        if return_cloud_mask:
            return img, mask_cube
        return img