import os

from .get_data import get_stac
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
    cloud_mask_output=None,
    indices=None,
    output=None,
    aggregator=None,
    stats=None,
    topographic_features=None,
    animation=None,
    update=None,
    source=None,
    q=None,
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
                    cloud_mask_output=cmo_i,
                    indices=list(indices) if indices else indices,
                    output=None,  # export below, per feature, so RAM frees each time
                    aggregator=aggregator,
                    stats=stats,
                    topographic_features=topographic_features,
                    animation=animation,
                    update=update,
                    source=source,
                    q=q,
                )

                # Export this feature now. The heavy compute happens INSIDE
                # export_stac and is released when this iteration ends, while `res`
                # stays lazy -- so RAM does not accumulate across features (peak is
                # ~one feature, not the whole batch).
                if output and isinstance(res, (xr.DataArray, xr.Dataset)):
                    stem, ext = os.path.splitext(output)
                    feature_output = f"{stem}_{idx}{ext}"
                    export_stac(res, feature_output)

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

            return results

    if update:
        stac_parameters = get_stac_parameters(update)

        mission = stac_parameters["mission"]
        resolution = stac_parameters["resolution"]
        polygon = stac_parameters["polygon"]
        # geometry = stac_parameters["geometry"] # update-clip raster
        bands = stac_parameters["spectral_bands"]
        indices = stac_parameters["indices"]
        if not isinstance(indices, list):
            indices = indices.tolist()
        if source is None:
            source = stac_parameters.get("stac_api", "element84")
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

    stac, baselines, tiles = get_stac(
        mission, polygon, resolution, daterange, bands, max_cc, cloud_masking,
        source=source or "element84",
    )
    crs = stac.spatial_ref.projected_crs_name
    transform = stac.rio.transform()

    # Cloud masking
    cloud_bool = None
    if cloud_masking is True:
        # keep_clouds=True -> pixels stay intact, only the cloud % is derived from
        # the returned per-pixel cloud boolean (cloud_bool). Default removes them.
        stac, cloud_bool = cloud_mask(stac, mission, keep_clouds=bool(keep_clouds))

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

        if cloud_masking is True:
            # Cloud % is measured against the observable AOI footprint: pixels
            # missing in every scene (incl. anything outside a non-rectangular
            # clip) are excluded from both numerator and denominator, so only
            # real clouds count. The count always comes from the SCL/QA cloud
            # boolean (not from NaN holes) so that per-date swath/tile gaps -
            # genuine NO_DATA, not cloud - are never miscounted as cloud. This
            # holds in both remove-clouds and keep-clouds mode.
            pct = compute_cloud_percentage(stac, cloud_mask=cloud_bool)
            if pct is not None:
                stac = stac.assign_coords(
                    cloud_percentage=("time", np.asarray(pct.data))
                )

            # On-demand: export the binary SCL cloud-mask time series (1=cloud,
            # 0=clear) as its own NetCDF. This is what lets a kept-clouds cube be
            # masked / filtered / co-registered later, even though the imagery
            # itself kept its clouds. Off by default (cloud_mask_output=None).
            if cloud_mask_output and cloud_bool is not None:
                mask_cube = build_scl_mask_cube(stac, cloud_bool)
                _cmo_dir = os.path.dirname(cloud_mask_output)
                if _cmo_dir:
                    os.makedirs(_cmo_dir, exist_ok=True)
                export_stac(
                    mask_cube, cloud_mask_output, crs=crs,
                    transform=stac.rio.transform(), var_name="Cloud_Stack",
                )
                if not q:
                    print(f"Binary cloud mask exported: {cloud_mask_output}", flush=True)

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
        stac = update_stac(stac_existing=update, stac_updated=stac)

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

        img = export_stac(stac, output, crs, transform)
        return img