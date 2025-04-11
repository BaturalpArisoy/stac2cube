from .vector_refiner import polygon_2_bbox

import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
from pystac_client import Client as pystacclient
from odc.stac import stac_load
import planetary_computer
import os

def get_stac(mission: str, polygon, resolution: int, daterange: list, bands: list, max_cc: int, cloud_masking: bool):

    catalogues = {
        "sentinel_2_l2a": ("https://earth-search.aws.element84.com/v1/", 'sentinel-2-l2a'), #sentinel-2-c1-l2a for terrabyte
        "sentinel_2_l1c": ("https://earth-search.aws.element84.com/v1/", 'sentinel-2-l1c'),
        "cop_dem_glo_30": ("https://stac.terrabyte.lrz.de/public/api/", 'cop-dem-glo-30'),
        "landsat_c2_l2": ("https://planetarycomputer.microsoft.com/api/stac/v1", 'landsat-c2-l2'), #landsat-ot-c2-l2
        "sentinel_1_rtc": ("https://planetarycomputer.microsoft.com/api/stac/v1", 'sentinel-1-rtc') # terrabytes sentinel-1-grd does not provide crs metadata, have to write a code that detects the crs by bbox coordinates
    }
    
    if resolution is not None:     
        resolution = resolution   
    else: 
        resolutions = {
            "sentinel_2_l2a": 10,
            "sentinel_2_l1c": 10,
            "cop_dem_glo_30": None,
            "landsat_c2_l2": 30,
            "sentinel_1_rtc": 10
        }
        resolution = resolutions[mission]

    if isinstance(polygon, list):
        bbox = polygon
    else:
        bbox = polygon_2_bbox(polygon)

    url, collection = catalogues[mission]

    if mission in ('sentinel_1_rtc', 'landsat_c2_l2'):
        catalog = pystacclient.open(url,
            modifier=planetary_computer.sign_inplace,
        )
    else:
        catalog = pystacclient.open(url)
        
    query = {
        'eo:cloud_cover': {
            "gte": 0,
            "lte": max_cc
        }
    }

    if mission in ('cop_dem_glo_30', 'sentinel_1_rtc'):
        query = None
    
    items, crs, stac_mission, tiles = _catalogue_search(catalog, collection, bbox, daterange, query, mission)
    
    band_map = _get_band_map(mission)
    if band_map is not None:
        bands = [band_map.get(band, band) for band in bands]

    if cloud_masking is True:
        if mission == "sentinel_2_l2a":
            bands.append('scl')
        if mission == "landsat_c2_l2":
            bands.append('qa_pixel')

    # Pre-filter duplicate items for sentinel_2_l1c based on processing baseline
    if mission == "sentinel_2_l1c":
        from collections import defaultdict
        grouped = defaultdict(list)
        for item in items:
            # Use date string (first 10 characters) as solar day key
            date_key = item.properties.get('datetime', '')[:10]
            grouped[date_key].append(item)
        filtered_items = []
        for date_key, group in grouped.items():
            # Choose item with highest processing baseline (converted to float)
            best_item = max(group, key=lambda it: float(it.properties.get("s2:processing_baseline", "0")))
            filtered_items.append(best_item)
        items = filtered_items

    stac = stac_load(
        items,
        bands=bands,
        crs=crs,
        resolution=resolution,
        resampling="bilinear",
        chunks={},
        groupby="solar_day",
        bbox=bbox,
    )
    
    if band_map is not None:
        reverse_band_map = {v: k for k, v in band_map.items()}
        if reverse_band_map:
            rename_dict = {band: reverse_band_map.get(band, band) for band in stac.data_vars if band in reverse_band_map}
            stac = stac.rename(rename_dict)
            
    if mission == "sentinel_2_l1c":
        date_list = [item.properties['datetime'] for item in items]
        processing_baseline_list = [item.properties["s2:processing_baseline"] for item in items]
        dates = pd.to_datetime(date_list, format='mixed').to_numpy(dtype='datetime64[ns]')
        baseline_da = xr.DataArray(
            processing_baseline_list,
            dims=["time"],
            coords={"time": dates},
            name="processing_baseline"
        )
        baseline_da_filtered = baseline_da.sel(time=baseline_da.time.isin(stac.time))
        unique_times, counts = np.unique(baseline_da_filtered.time.values, return_counts=True)
        duplicate_times = unique_times[counts > 1]
        stac = stac.sel(time=~np.isin(stac.time, duplicate_times))
        baselines = baseline_da_filtered.sel(time=~np.isin(baseline_da_filtered.time, duplicate_times))
                
        return stac, baselines, tiles
    else:
        if mission == "sentinel_1_rtc":
            from datetime import datetime
            orbit_state_by_day = {}
            for item in items:
                item_date = datetime.fromisoformat(item.properties["datetime"]).date()
                if item_date not in orbit_state_by_day:
                    orbit_state_by_day[item_date] = item.properties["sat:orbit_state"]
            solar_days_in_stac = [pd.Timestamp(t).date() for t in stac.time.values]
            aligned_orbit_states = [orbit_state_by_day.get(day, None) for day in solar_days_in_stac]
            if None in aligned_orbit_states:
                print("Warning: Some dates in the stac dataset did not have a matching orbit state.")
            stac = stac.assign_coords(orbit_state=("time", aligned_orbit_states))
        
        return stac, None, tiles

def _catalogue_search(catalog, collection, bbox, daterange, query, mission):

    results = catalog.search(
        bbox=bbox,
        collections=[collection],
        datetime=daterange,
        query=query,
    )
    
    items = results.item_collection()
    
    if mission == 'sentinel_2_l1c':
        for item in items:
            for asset in item.assets.values():
                asset.href = asset.href.replace('sentinel-s2-l2a', 'sentinel-s2-l1c')
        os.environ['AWS_REQUEST_PAYER'] = 'requester'
        os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
    
    if len(items) < 1:
        raise ValueError("No scenes found by the given parameters. Please check your polygon's geometry, date range or increase max cloud coverage.")
       
    sample_item = items[0]
    crs = sample_item.properties.get('proj:code') or sample_item.properties.get('proj:epsg')
    stac_mission = sample_item.to_dict().get("collection")
    # Get Sentinel tile ID
    if mission in ('sentinel_2_l2a', 'sentinel_2_l1c'):
        gdf = gpd.GeoDataFrame.from_features(items, "epsg:4326")
        gdf["granule"] = (
            gdf["mgrs:utm_zone"].apply(lambda x: f"{x:02d}")
            + gdf["mgrs:latitude_band"]
            + gdf["mgrs:grid_square"]
        )
        tiles = gdf["granule"].unique()
    else:
        tiles = None

    return items, crs, stac_mission, tiles



def _get_band_map(mission: str):
    
    band_maps = {
        "landsat_ot_c2_l2": {
            'coastal': 'B01',
            'blue': 'B02',
            'green': 'B03',
            'red': 'B04',
            'nir': 'B05',
            'swir1': 'B06',
            'swir2': 'B07',
            'thermal': 'B10',
            'qa_temp': 'QA_Temp',
            'qa_pixel': 'QA_Pixel',
            'qa_radsat': 'QA_Radsat',
            'qa_aerosol': 'QA_Aerosol'
        },
        "landsat_c2_l2": {
            'coastal': 'coastal',
            'blue': 'blue',
            'green': 'green',
            'red': 'red',
            'nir': 'nir08',
            'swir1': 'swir16',
            'swir2': 'swir22',
            'thermal': 'lwir11', # SCALE FACTOR FOR THERMAL IS MISSING!
            'qa_pixel': 'qa_pixel',
            'qa_radsat': 'qa_radsat',
            'qa_aerosol': 'qa_aerosol'
        },
        "s2_placeholder": { # Will be activated once switched to terrabyte catalog from Element84.
            'coastal': 'B01',
            'blue': 'B02',
            'green': 'B03',
            'red': 'B04',
            'nir': 'B08',
            'red_edge1': 'B05',
            'red_edge2': 'B06',
            'red_edge3': 'B07',
            'swir1': 'B11',
            'swir2': 'B12',
            'scl': 'SCL'
        }
    }

    return band_maps.get(mission)
