from .get_data import get_stac
from .vector_refiner import proj_check, polygon_2_bbox, read_polygon_file
from .stac_processing import scale_factor, cloud_mask
from .get_spectral_indices import calculate_spectral_index
from .export_cfg import export_stac

# from .get_topo import calculate_topo
# from .time_series_tools import generate_animation
from .clip import clip_stac
from .get_statistics import calculate_statistics
from .get_update import get_stac_parameters, update_stac

import xarray as xr
import rioxarray as rio
import pandas as pd
import numpy as np


def get_stac_layers(
    mission=None,
    polygon=None,
    resolution=None,
    daterange=None,
    bands=None,
    max_cc=None,
    clip_raster=None,
    cloud_masking=None,
    indices=None,
    output=None,
    aggregator=None,
    stats=None,
    topographic_features=None,
    animation=None,
    update=None,
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
        # NOTE: do NOT force output=update here anymore.
        # This allows update mode to return an in-memory updated cube when output=None.
    else:
        if not mission:
            raise ValueError("Error: Please select a mission.")
        if not polygon:
            raise ValueError(
                "Error: Please select a polygon or bbox list with geographic coordinates."
            )

    # If projected coords are given, will transform to WGS84 coords
    # if not isinstance(polygon, list):
    #    polygon = proj_check(polygon)

    stac, baselines, tiles = get_stac(
        mission, polygon, resolution, daterange, bands, max_cc, cloud_masking
    )
    crs = stac.spatial_ref.projected_crs_name
    transform = stac.rio.transform()

    # Cloud masking
    if cloud_masking is True:
        stac = cloud_mask(stac, mission)

    # Scale factor
    stac = scale_factor(stac, mission, baselines)
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
            null_count_per_time = stac.isnull().sum(dim=["band", "y", "x"])
            total_elements = stac.sizes["band"] * stac.sizes["y"] * stac.sizes["x"]
            cloud_percentage_int = (
                (null_count_per_time / total_elements) * 100
            ).astype(int)
            stac = stac.assign_coords(
                cloud_percentage=("time", cloud_percentage_int.data)
            )

    stac.attrs["crs"] = crs
    stac.attrs["transform"] = transform

    # stac = stac.copy()
    stac.attrs.pop("nodata", None)
    try:
        stac = stac.rio.write_nodata(None, inplace=False)
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
            stac = stac.rio.write_crs(crs, inplace=False)
            stac = stac.rio.write_transform(transform, inplace=False)
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