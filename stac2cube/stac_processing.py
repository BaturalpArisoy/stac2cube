import numpy as np
import xarray as xr

'''
def cloud_mask(stac, mission):

    cfg = _mission_cfg(mission)
    layer_name = cfg[0]
    classes = cfg[1]
    
    cloud_mask = getattr(stac, layer_name).isin(classes)

    stac_masked = stac.where(~cloud_mask)

    stac_masked = stac_masked.drop_vars(layer_name)

    return stac_masked
'''

def scale_factor(stac, mission, baselines):

    if mission == "sentinel-2-l1c":
        baselines = baselines.astype(float)
        baselines_aligned = baselines.sel(time=stac.time)
        
        stac = xr.where(baselines_aligned >= 4.00, (stac - 1000) / 10000, stac / 10000)
        return stac
    
    else:
        cfg = _mission_cfg(mission)
        gain = cfg[2]
        offset = cfg[3]

        return (stac + offset) * gain


def _mission_cfg(mission):

    # relevant band (cloud), classifications (cloud), gain (scale), offset (scale) 
    cfg = {
        "sentinel-2-l2a": ('scl', [3, 8, 9, 10], 1e-4, 0),
        "sentinel-2-l1c": (None, None, 1e-4, -1000),
        "landsat_ot_c2_l2": ('qa_pixel', [1, 2, 3, 4], 0.0000275, -0.2),
        "sentinel-1-rtc": (None, None, 1, 0),
        "cop_dem_glo_30": (None, None, 1, 0)
    }

    return cfg[mission]