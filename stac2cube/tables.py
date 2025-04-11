import pandas as pd


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
        "name": "sentinel_2_l2a",
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
        "animation": False
    }

    landsat_c2_l2 = {
        "name": "landsat_c2_l2",
        "allias": "l_oli",
        "stac_catalog": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "default_resolution": 30,
        "bands": ["coastal", "blue", "green", "red", "nir", "swir1", "swir2", "thermal"],
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
    
    df = pd.concat([df, pd.DataFrame([sentinel_2_l2a, sentinel_2_l1c, sentinel_1_rtc, landsat_c2_l2, cop_dem_glo_30])], ignore_index=True)
    df.style.set_properties(**{'text-align': 'left'})
    pd.set_option('display.max_colwidth', None)

    return df



def missions_terrabyte():
    pass