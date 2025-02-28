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

    if mission == "s2_l1c":
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
        "s2": ('scl', [3, 8, 9, 10], 1e-4, 0),
        "s2_l1c": (None, None, 1e-4, -1000),
        "l_oli": ('qa_pixel', [1, 2, 3, 4], 0.0000275, -0.2),
        "s1": (None, None, 1, 0),
        "cop_dem": (None, None, 1, 0)
    }

    return cfg[mission]