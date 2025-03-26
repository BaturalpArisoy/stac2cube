from .get_data import get_stac
from .vector_refiner import proj_check, polygon_2_bbox
from .stac_processing import scale_factor
from .get_spectral_indices import calculate_spectral_index
from .export_cfg import export_stac
from .get_topo import calculate_topo
from .time_series_tools import generate_animation
from .clip import clip_stac
from .get_statistics import calculate_statistics
from .get_update import get_stac_parameters, update_stac

import xarray as xr
import rioxarray as rio
import pandas as pd

def get_stac_layers(
    mission = None,
    polygon = None,
    resolution = None,
    daterange = None,
    bands= None,
    max_cc= None,
    clip_raster= None,
    cloud_masking = None,
    indices = None ,
    output = None,
    aggregator = None,
    stats = None,
    topographic_features = None,
    animation = None,
    update = None):
    
    # Rename short names
    if mission == "s2":
        mission = "sentinel-2-l2a"
    if mission == "s2_l1c":
        mission = "sentinel-2-l1c"
    if mission == "s1":
        mission = "sentinel-1-rtc"
    if mission == "l_oli":
        mission = "landsat_ot_c2_l2"
    if mission == "cop_dem":
        mission = "cop_dem_glo_30"

    if update:
        stac_parameters = get_stac_parameters(update)

        mission = stac_parameters["mission"]
        resolution = stac_parameters["resolution"]
        polygon = stac_parameters["polygon"]
        bands = stac_parameters["spectral_bands"]
        indices = stac_parameters["indices"]
        output = update
    else:
        if not mission:
            raise ValueError("Error: Please select a mission.")
        if not polygon:
            raise ValueError("Error: Please select a polygon or bbox list with geographic coordinates.")

    #if not isinstance(polygon, list):
    #    polygon = proj_check(polygon)

    stac, baselines, tiles = get_stac(mission, polygon, resolution, daterange, bands, max_cc, cloud_masking)
    crs = stac.spatial_ref.projected_crs_name
    transform = stac.rio.transform()
    
    # Cloud masking
#    if cloud_masking is True:
#        stac = cloud_mask(stac, mission)

    # Scale factor
    stac = scale_factor(stac, mission, baselines)
    #stac.rio.write_crs(crs, inplace=True)
    
    # Transform zeros to nan
    stac = stac.where(stac != 0)
    
    # Index calculation
    # Add code when only indices are asked without band selection
    if indices:
        stac_indices = calculate_spectral_index(stac, mission, indices)

    # Add animation here
    #if animation is True:
    #    generate_animation(stac)
    
    if mission == 'cop_dem_glo_30':        
        dem = stac.isel(time=0).dem
        dem = dem.expand_dims(dim={'band': ['dem']})
        stac_topo_features = calculate_topo(dem, topographic_features)
        
    # Dataset -> DataArray
    if mission != 'cop_dem_glo_30':  
        bands = list(stac.data_vars.keys())
        stac = xr.concat([stac[band] for band in bands], dim='band')
        stac = stac.assign_coords(band=bands)

    # DataArray manipulation
    if indices:
        stac = xr.concat([stac, stac_indices], dim='band')
        stac.attrs['indices'] = indices    

    if mission == 'cop_dem_glo_30':
        stac = xr.concat([dem, stac_topo_features], dim='band')
        stac = stac.rename('Topographic_Features')
    else:
        stac = stac.transpose('time', 'band', 'y', 'x')
        stac = stac.rename('Spectral_Temporal_Stack')

    # Add metadata as attributes
    if not update:
        stac.attrs['spectral_bands'] = bands
        stac.attrs['mission'] = mission
        stac.attrs['tile_id'] = tiles
        bbox = polygon_2_bbox(polygon)
        stac.attrs['bbox'] = bbox

    # Calculate stats image (optional)
    if aggregator:
        print(f"stac before {aggregator}:")
        print(stac)
        print("\n-------------------------------------")
        if aggregator == 'mean':
            stac = stac.mean(dim='time', skipna=True) 
        elif aggregator == 'median':
            stac = stac.median(dim='time', skipna=True)
        else:
            raise ValueError("Invalid aggregator. Please select either 'mean' or 'median'.")
    
    # Clip netcdf as clip raster
    if clip_raster:
        stac = clip_stac(stac, polygon, crs) # delete write_crs in clip_stac
            
    # Finalizing
    if not aggregator:
      stac['time'] = stac['time'].dt.floor('D')

    stac.attrs['crs'] = crs
    stac.attrs['transform'] = transform

    if not output:
        stac.rio.write_crs(crs, inplace=True)
        stac.rio.write_transform(transform, inplace=True)
        #if mission == "sentinel-2-l1c":
         #   stac.attrs['crs'] = crs
          #  stac.attrs['transform'] = transform
        stac.attrs['crs'] = crs
        stac.attrs['transform'] = transform
        print(stac, flush=True)
        return stac # returns lazy
    else:
        if update:
            stac = update_stac(stac_existing=update, stac_updated=stac)
        print(stac)
        if mission != 'cop_dem_glo_30':
            if not aggregator:
                print(stac.time.values)
                if stats:
                    stac = calculate_statistics(stac, stats)
        print(stac.band.values)
        img = export_stac(stac, output, crs, transform)
        return img

def missions():

    columns = [
        "name",
        "allias",
        "stac_catalog",
        "default_resolution",
        "bands",
        "indices",
        "topographic_features",
        "max_cc",
        "clip_raster",
        "cloud_masking",
        "output",
        "aggregator",
        "stats",
        "update",
        "animation"
    ]
    
    df = pd.DataFrame(columns=columns)
    
    sentinel_2_l2a = {
        "name": "sentinel-2-l2a",
        "allias": "s2",
        "stac_catalog": "https://earth-search.aws.element84.com/v1/",
        "default_resolution": 10,
        "bands": ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "nir09", "swir16", "swir22"],
        "indices": ["ndvi", "ndwi", "savi"],
        "topographic_features": False,
        "max_cc": 100,
        "clip_raster": [True, False],
        "cloud_masking": [True, False],
        "output": "path/to/output.nc",
        "aggregator": ["mean", "median"],
        "stats": ["mean", "median", "std", "min", "max"],
        "update": "path/to/stac.nc",
        "animation": [True, False]
    }

    sentinel_2_l1c = {
        "name": "sentinel_2_l1c",
        "allias": "s2_l1c",
        "stac_catalog": "https://earth-search.aws.element84.com/v1/",
        "default_resolution": 10,
        "bands": ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "nir09", "cirrus", "swir16", "swir22"],
        "indices": ["ndvi", "ndwi", "savi"],
        "topographic_features": False,
        "max_cc": 100,
        "clip_raster": [True, False],
        "cloud_masking": [True, False],
        "output": "path/to/output.nc",
        "aggregator": ["mean", "median"],
        "stats": ["mean", "median", "std", "min", "max"],
        "update": "path/to/stac.nc",
        "animation": [True, False]
    }

    sentinel_1_rtc = {
        "name": "sentinel_1_rtc",
        "allias": "s1",
        "stac_catalog": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "default_resolution": 10,
        "bands": ["vh", "vv"],
        "indices": ["vh/vv", "vv/vh", "rvi"],
        "topographic_features": False,
        "max_cc": False,
        "clip_raster": [True, False],
        "cloud_masking": False,
        "output": "path/to/output.nc",
        "aggregator": ["mean", "median"],
        "stats": ["mean", "median", "std", "min", "max"],
        "update": "path/to/stac.nc",
        "animation": [True, False]
    }

    landsat_ot_c2_l2 = {
        "name": "landsat-ot-c2-l2",
        "allias": "l_oli",
        "stac_catalog": "https://stac.terrabyte.lrz.de/public/api/",
        "default_resolution": 30,
        "bands": ["coastal", "blue", "green", "red", "nir", "swir1", "swir2", "thermal"],
        "indices": ["ndvi", "ndwi", "savi"],
        "topographic_features": False,
        "max_cc": 100,
        "clip_raster": [True, False],
        "cloud_masking": "path/to/stac.nc",
        "output": "path/to/output.nc",
        "aggregator": ["mean", "median"],
        "stats": ["mean", "median", "std", "min", "max"],
        "update": "path/to/stac.nc",
        "animation": [True, False]
    }

    cop_dem_glo_30 = {
        "name": "cop_dem_glo_30",
        "allias": "cop_dem",
        "stac_catalog": "https://stac.terrabyte.lrz.de/public/api/",
        "default_resolution": False,
        "bands": False,
        "indices": False,
        "topographic_features": ["slope", "aspect", "d_inf_flow_accumulation", "twi"],
        "max_cc": False,
        "clip_raster": [True, False],
        "cloud_masking": False,
        "output": "path/to/output.nc",
        "aggregator": False,
        "stats": False,
        "update": False,
        "animation": False
    }
    
    df = pd.concat([df, pd.DataFrame([sentinel_2_l2a, sentinel_2_l1c, sentinel_1_rtc, landsat_ot_c2_l2, cop_dem_glo_30])], ignore_index=True)
    df.style.set_properties(**{'text-align': 'left'})
    pd.set_option('display.max_colwidth', None)

    return df