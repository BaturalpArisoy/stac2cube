from .vector_refiner import polygon_2_bbox

import pandas as pd
import xarray as xr
import numpy as np
from pystac_client import Client as pystacclient
from odc.stac import stac_load
import planetary_computer
import os

def get_stac(mission: str, polygon, resolution, daterange: list, bands: list, max_cc: int, cloud_masking: bool):

    catalogues = {
        "s2": ("https://earth-search.aws.element84.com/v1/", 'sentinel-2-l2a'),
        "s2_l1c": ("https://earth-search.aws.element84.com/v1/", 'sentinel-2-l1c'),
        "cop_dem": ("https://stac.terrabyte.lrz.de/public/api/", 'cop-dem-glo-30'),
        "l_oli": ("https://stac.terrabyte.lrz.de/public/api/", 'landsat-ot-c2-l2'),
        "s1": ("https://planetarycomputer.microsoft.com/api/stac/v1", 'sentinel-1-rtc')
    }
    
    if resolution is not None:     
        resolution = resolution   
    else: 
        resolutions = {
            "s2": 10,
            "s2_l1c": 10,
            "cop_dem": None,
            "l_oli": 30,
            "s1": 10
        }
        resolution = resolutions[mission]
    
    if isinstance(polygon, list):
        bbox = polygon
    else:
        bbox = polygon_2_bbox(polygon)

    url, collection = catalogues[mission]

    if mission == 's1':
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

    if mission in ('cop_dem', 's1'):
        query = None
    
    items, crs = _catalogue_search(catalog, collection, bbox, daterange, query)
    
    band_map = _get_band_map(mission)
    if band_map is not None:
        bands = [band_map.get(band, band) for band in bands]

    if cloud_masking is True:
        if mission == "s2":
            bands.append('scl')

    # Pre-filter duplicate items for s2_l1c based on processing baseline
    if mission == "s2_l1c":
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
            
    if mission == "s2_l1c":
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
                
        return stac, baselines
    else:
        return stac, None

def _catalogue_search(catalog, collection, bbox, daterange, query):

    results = catalog.search(
        bbox=bbox,
        collections=[collection],
        datetime=daterange,
        query=query,
    )
    
    items = results.item_collection()
    
    if collection == 'sentinel-2-l1c':
        for item in items:
            for asset in item.assets.values():
                asset.href = asset.href.replace('sentinel-s2-l2a', 'sentinel-s2-l1c')
        os.environ['AWS_REQUEST_PAYER'] = 'requester'
        os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
    
    if len(items) < 1:
        raise ValueError("No scenes found by the given parameters. Please check your polygon's geometry or increase max cloud coverage.")
       
    sample_item = items[0]
    crs = sample_item.properties.get('proj:code') or sample_item.properties.get('proj:epsg')
    return items, crs



def _get_band_map(mission: str):
    
    band_maps = {
        "l_oli": {
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
        "s2_placeholder": { # Will be revoked once switched to terrabyte catalog from Element84.
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
